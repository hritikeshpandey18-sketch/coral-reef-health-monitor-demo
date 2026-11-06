"""
dataset.py
Stratified splitting and tf.data input pipelines.

Two things here matter for a defensible result:

1. The split is done ONCE, on disk, with a fixed seed. Re-running
   training does not reshuffle the test set, so reported numbers are
   reproducible and the test set is never seen during model selection.
2. Augmentation is applied to the TRAINING stream only. A common
   student mistake is augmenting validation/test data, which inflates
   or destabilises the reported accuracy.
"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from . import config as C
except ImportError:                     # pragma: no cover
    import config as C

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ======================================================================
# 1. STRATIFIED SPLIT ON DISK
# ======================================================================
def stratified_split(src_root: Path = C.ENHANCED_DIR,
                     dst_root: Path = C.SPLIT_DIR,
                     ratios: tuple = C.SPLIT_RATIOS,
                     seed: int = C.RANDOM_SEED,
                     verbose: bool = True) -> dict:
    """Split src_root/<class>/* into dst_root/{train,val,test}/<class>/*.

    Stratified: each class is split by the same ratios, so rare classes
    are not accidentally absent from the test set.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"

    # The split is permanent: regenerating it mid-study means earlier and
    # later results were measured on different test sets.
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(
            f"{dst_root} already exists and is not empty. Refusing to "
            "overwrite the split; delete it by hand only if every model "
            "will be retrained.")
    for split in ("train", "val", "test"):
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        files = [f for f in sorted(class_dir.iterdir())
                 if f.suffix.lower() in VALID_EXT]
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        chunks = {"train": files[:n_train],
                  "val": files[n_train:n_train + n_val],
                  "test": files[n_train + n_val:]}

        for split, items in chunks.items():
            out_dir = dst_root / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out_dir / f.name)
            counts[split][class_dir.name] = len(items)

    if verbose:
        print(f"{'class':<14}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
        for cls in sorted(counts['train']):
            t, v, s = (counts[k][cls] for k in ("train", "val", "test"))
            print(f"{cls:<14}{t:>8}{v:>8}{s:>8}{t + v + s:>8}")
    return {k: dict(v) for k, v in counts.items()}


# ======================================================================
# 2. CLASS WEIGHTS (imbalance handling)
# ======================================================================
def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict:
    """Inverse-frequency weights, normalised so the mean weight is 1.

    Passed to model.fit(class_weight=...) so the loss is not dominated
    by whichever class happens to have the most images.
    """
    train_dir = Path(train_dir)
    counts = []
    for cls in class_names:
        d = train_dir / cls
        counts.append(len([f for f in d.iterdir()
                           if f.suffix.lower() in VALID_EXT]) if d.exists() else 0)
    counts = np.array(counts, dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError(f"No training images found in {train_dir}")
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return {i: float(w[i]) for i in range(len(counts))}


# ======================================================================
# 3. tf.data PIPELINES
# ======================================================================
def build_datasets(split_root: Path = C.SPLIT_DIR,
                   img_size: tuple = C.IMG_SIZE,
                   batch_size: int = C.BATCH_SIZE,
                   seed: int = C.RANDOM_SEED):
    """Return (train_ds, val_ds, test_ds, class_names).

    Labels are integer-encoded; the models use
    SparseCategoricalCrossentropy, so no one-hot conversion is needed.
    """
    import tensorflow as tf

    split_root = Path(split_root)
    common = dict(label_mode="int", image_size=img_size,
                  batch_size=batch_size, interpolation="bilinear")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "train", shuffle=True, seed=seed, **common)
    class_names = list(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "val", shuffle=False,
        class_names=class_names, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "test", shuffle=False,
        class_names=class_names, **common)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def augmentation_layer(seed: int = C.RANDOM_SEED):
    """Geometric + photometric augmentation, applied to training only.

    Vertical flip and full rotation are legitimate here: a diver's
    camera has no canonical "up" underwater, unlike, say, street photos.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.25, seed=seed),
        layers.RandomZoom(0.15, seed=seed),
        layers.RandomTranslation(0.10, 0.10, seed=seed),
        layers.RandomContrast(0.15, seed=seed),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
    ], name="augmentation")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="split data/raw into data/splits_raw (ablation arm)")
    if ap.parse_args().raw:
        stratified_split(src_root=C.RAW_DIR, dst_root=C.SPLIT_RAW_DIR)
    else:
        stratified_split()
