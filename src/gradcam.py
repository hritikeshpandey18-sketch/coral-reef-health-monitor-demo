"""
gradcam.py
Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

Why this belongs in the project: a marine biologist will not act on a
label from a black box. Grad-CAM shows which pixels drove the decision,
so a reviewer can check the model is looking at coral tissue and not at
the sand, the diver's fin, or a caption burned into the frame. It also
turns your error analysis from guesswork into evidence.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from . import config as C
except ImportError:                     # pragma: no cover
    import config as C


def _output_rank(layer):
    """Rank of a layer's output tensor, or None if it cannot be read."""
    try:
        shape = layer.output.shape          # Keras 3
    except Exception:                       # noqa: BLE001
        shape = getattr(layer, "output_shape", None)
    if shape is None:
        return None
    if isinstance(shape, list):             # multi-output layer
        shape = shape[0]
    return len(shape)


def find_last_conv_index(model) -> int:
    """Index of the deepest layer in `model.layers` that still emits a
    4-D feature map (batch, H, W, C).

    For the transfer-learning models this is the pretrained backbone
    itself, because `include_top=False` with no pooling makes the
    backbone's own output the final convolutional feature map. For the
    baseline CNN it is the last pooling block.
    """
    import tensorflow as tf
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if _output_rank(layer) == 4:
            return i
    raise ValueError("No 4-D feature map found; Grad-CAM needs a "
                     "convolutional layer.")


def compute_heatmap(model, img_array: np.ndarray,
                    class_index: int | None = None,
                    split_index: int | None = None):
    """img_array: (1, H, W, 3) float32 in the SAME range the model expects
    at its input layer (raw 0-255 here, because preprocessing is baked
    into the model as a layer).

    Returns (heatmap HxW in [0,1], predicted_index, confidence).

    The forward pass is replayed layer by layer so the intermediate
    feature map can be watched by the gradient tape. This assumes a
    linear layer stack, which both model families here satisfy; anything
    more exotic falls back to occlusion sensitivity automatically.
    """
    import tensorflow as tf

    try:
        k = split_index if split_index is not None else find_last_conv_index(model)
        layers = model.layers
        start = 1 if isinstance(layers[0], tf.keras.layers.InputLayer) else 0

        with tf.GradientTape() as tape:
            x = tf.convert_to_tensor(img_array, dtype=tf.float32)
            for layer in layers[start:k + 1]:
                x = layer(x, training=False)
            feats = x
            tape.watch(feats)
            for layer in layers[k + 1:]:
                x = layer(x, training=False)
            preds = x
            idx = int(tf.argmax(preds[0])) if class_index is None else class_index
            score = preds[:, idx]

        grads = tape.gradient(score, feats)
        if grads is None:
            raise RuntimeError("gradients did not flow to the feature map")

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(feats[0] * weights, axis=-1)
        cam = tf.nn.relu(cam).numpy()
        if cam.max() <= 0:                  # all-negative activation map
            raise RuntimeError("empty class activation map")
        cam = cam / (cam.max() + 1e-8)
        return cam, idx, float(preds[0][idx])

    except Exception as exc:                # noqa: BLE001
        print(f"Grad-CAM unavailable ({exc}); using occlusion sensitivity.")
        return _fallback_heatmap(model, img_array, class_index)


def _fallback_heatmap(model, img_array, class_index=None):
    """Occlusion sensitivity: slide a grey patch over the image and
    record the drop in the class score. Slower than Grad-CAM but works
    on any architecture, including ones where gradients do not flow
    cleanly through a nested functional model."""
    import tensorflow as tf

    base = model(img_array, training=False).numpy()[0]
    idx = int(base.argmax()) if class_index is None else class_index
    h, w = img_array.shape[1:3]
    step, size = max(h // 12, 8), max(h // 6, 16)
    grid = np.zeros((h // step + 1, w // step + 1), dtype=np.float32)

    for i, y in enumerate(range(0, h, step)):
        batch, coords = [], []
        for j, x in enumerate(range(0, w, step)):
            occ = img_array.copy()
            occ[0, y:y + size, x:x + size, :] = 127.0
            batch.append(occ[0]); coords.append(j)
        probs = model(tf.stack(batch), training=False).numpy()[:, idx]
        for j, p in zip(coords, probs):
            grid[i, j] = max(base[idx] - p, 0.0)

    grid = grid / (grid.max() + 1e-8)
    return grid, idx, float(base[idx])


def overlay_heatmap(original_bgr: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    h, w = original_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 1 - alpha, hm, alpha, 0)


def explain_image(model, image_path: str | Path, class_names: list[str],
                  out_path: str | Path | None = None,
                  enhance: bool = True):
    """End-to-end: read -> (optionally) enhance -> predict -> overlay."""
    try:
        from .preprocessing import enhance_underwater_image
    except ImportError:
        from preprocessing import enhance_underwater_image

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    if enhance:
        img = enhance_underwater_image(img)

    resized = cv2.resize(img, C.IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")
    batch = np.expand_dims(rgb, 0)

    heatmap, idx, conf = compute_heatmap(model, batch)
    overlay = overlay_heatmap(resized, heatmap)

    if out_path:
        strip = np.hstack([resized, overlay])
        cv2.imwrite(str(out_path), strip)
    return {"label": class_names[idx], "confidence": round(conf, 4),
            "overlay_bgr": overlay, "heatmap": heatmap}
