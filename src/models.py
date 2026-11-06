"""
models.py
Model definitions: one baseline CNN trained from scratch and three
ImageNet-pretrained backbones used for transfer learning.

Design notes worth defending in a viva:

* Each backbone gets its OWN `preprocess_input`, baked into the model
  as a Lambda layer. ResNet50 expects caffe-style mean subtraction,
  MobileNetV2 expects [-1, 1], EfficientNet expects raw [0, 255].
  Using the wrong one silently costs several accuracy points - it is
  the single most common bug in student transfer-learning code.
* Augmentation lives INSIDE the model and is active only in training
  mode, so it is automatically disabled at evaluation and inference.
* Training is two-stage: freeze the backbone and train the head, then
  unfreeze the top block at a much lower learning rate. Fine-tuning
  from the start destroys pretrained features with large random
  gradients from the untrained head.
"""

from __future__ import annotations

try:
    from . import config as C
    from .dataset import augmentation_layer
except ImportError:                     # pragma: no cover
    import config as C
    from dataset import augmentation_layer


# ======================================================================
# BACKBONE REGISTRY
# ======================================================================
def _backbone_registry():
    import tensorflow as tf
    ka = tf.keras.applications
    return {
        "mobilenetv2": (ka.MobileNetV2,
                        ka.mobilenet_v2.preprocess_input),
        "resnet50": (ka.ResNet50,
                     ka.resnet.preprocess_input),
        "efficientnetb0": (ka.EfficientNetB0,
                           ka.efficientnet.preprocess_input),
        "densenet121": (ka.DenseNet121,
                        ka.densenet.preprocess_input),
    }


# ======================================================================
# SERIALISABLE PREPROCESSING LAYER
# ======================================================================
# A bare `layers.Lambda(preprocess_input)` trains fine but CANNOT be
# reloaded: Keras 3 saves the function only by name and fails with
# "Could not locate function 'preprocess_input'". Wrapping it in a
# registered Layer that stores the backbone NAME fixes round-tripping,
# so model.save() / load_model() actually work.
_PREPROCESS_LAYER = None


def _make_preprocess_layer():
    global _PREPROCESS_LAYER
    if _PREPROCESS_LAYER is not None:
        return _PREPROCESS_LAYER

    import tensorflow as tf
    try:                                # Keras 3 (TF >= 2.16)
        import keras
        register = keras.saving.register_keras_serializable
    except (ImportError, AttributeError):   # Keras 2 fallback
        register = tf.keras.utils.register_keras_serializable

    @register(package="coral")
    class BackbonePreprocess(tf.keras.layers.Layer):
        """Applies the preprocessing that `backbone` was pretrained with."""

        def __init__(self, backbone: str, **kwargs):
            super().__init__(**kwargs)
            self.backbone = backbone

        def call(self, inputs):
            fn = _backbone_registry()[self.backbone][1]
            return fn(inputs)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            cfg = super().get_config()
            cfg["backbone"] = self.backbone
            return cfg

    _PREPROCESS_LAYER = BackbonePreprocess
    return BackbonePreprocess


# ======================================================================
# BASELINE - PLAIN CNN, NO PRETRAINING
# ======================================================================
def build_baseline_cnn(num_classes: int = C.NUM_CLASSES,
                       img_size: tuple = C.IMG_SIZE,
                       augment: bool = True):
    """A conventional 4-block CNN. This is the control condition: it
    shows how much of the final accuracy comes from transfer learning
    rather than from the dataset being easy."""
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(*img_size, 3), name="input_image")
    x = augmentation_layer()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(C.DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


# ======================================================================
# TRANSFER-LEARNING MODEL
# ======================================================================
def build_transfer_model(backbone: str = "mobilenetv2",
                         num_classes: int = C.NUM_CLASSES,
                         img_size: tuple = C.IMG_SIZE,
                         dropout: float = C.DROPOUT,
                         augment: bool = True,
                         weights: str | None = "imagenet"):
    """Pretrained backbone + global pooling + dropout + softmax head.

    `weights=None` builds the same architecture with random init - useful
    for smoke-testing offline, and as a second control alongside the
    baseline CNN.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    backbone = backbone.lower()
    registry = _backbone_registry()
    if backbone not in registry:
        raise ValueError(f"Unknown backbone '{backbone}'. "
                         f"Choose from {sorted(registry)}")
    ctor, _ = registry[backbone]
    BackbonePreprocess = _make_preprocess_layer()

    inputs = layers.Input(shape=(*img_size, 3), name="input_image")
    x = augmentation_layer()(inputs) if augment else inputs
    x = BackbonePreprocess(backbone, name=f"{backbone}_preprocess")(x)

    base = ctor(include_top=False, weights=weights,
                input_shape=(*img_size, 3))
    base.trainable = False                       # stage 1
    x = base(x, training=False)                  # keep BatchNorm in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{backbone}_transfer")
    model.base_model = base                      # handle for fine-tuning
    return model


# ======================================================================
# COMPILE HELPERS
# ======================================================================
def compile_model(model, lr: float = C.LR_HEAD,
                  label_smoothing: float = C.LABEL_SMOOTHING):
    """Compile with Adam + sparse categorical cross-entropy.

    Note: label smoothing is not supported by the *sparse* loss, so it
    is applied only when labels are one-hot. We keep sparse labels for
    simplicity and pass label_smoothing=0 here; the argument is kept so
    the one-hot variant can be dropped in without changing call sites.
    """
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def unfreeze_for_finetuning(model, fine_tune_at: float = C.FINE_TUNE_AT,
                            lr: float = C.LR_FINETUNE):
    """Stage 2: unfreeze the top (1 - fine_tune_at) fraction of the
    backbone and recompile with a much smaller learning rate.

    BatchNormalization layers stay frozen. Their running statistics were
    estimated on ImageNet over millions of images; re-estimating them on
    a few hundred coral images is unstable and usually hurts.
    """
    import tensorflow as tf

    base = getattr(model, "base_model", None)
    if base is None:
        # A reloaded model loses the python attribute; find the nested
        # backbone by looking for the single sub-Model in the graph.
        nested = [l for l in model.layers if isinstance(l, tf.keras.Model)
                  and l.name != "augmentation"]
        base = nested[0] if nested else None
    if base is None:
        raise AttributeError("Model has no backbone; fine-tuning only "
                             "applies to transfer-learning models.")
    base.trainable = True
    cutoff = int(len(base.layers) * fine_tune_at)
    for i, layer in enumerate(base.layers):
        if i < cutoff or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Fine-tuning: {trainable}/{len(base.layers)} backbone layers "
          f"unfrozen, lr={lr}")
    return compile_model(model, lr=lr)


def load_trained_model(path):
    """Load a saved .keras model.

    ALWAYS use this instead of `tf.keras.models.load_model` directly.
    The custom preprocessing layer is registered lazily, so a fresh
    Python process that has not built a model yet cannot deserialise
    the file. This helper forces registration first.
    """
    import tensorflow as tf
    _make_preprocess_layer()
    return tf.keras.models.load_model(str(path), compile=False,
                                      safe_mode=False)


def build_model(name: str, num_classes: int = C.NUM_CLASSES,
                weights: str | None = "imagenet", **kw):
    """Factory used by train.py so the CLI can say --model baseline."""
    if name.lower() in ("baseline", "cnn", "baseline_cnn"):
        return build_baseline_cnn(num_classes=num_classes, **kw)
    return build_transfer_model(backbone=name, num_classes=num_classes,
                                weights=weights, **kw)
