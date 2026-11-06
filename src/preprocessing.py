"""
preprocessing.py
Underwater image enhancement for coral reef photographs.

Water absorbs red light far faster than blue/green, so raw reef images
are colour-cast, low-contrast and hazy. Feeding them straight to a CNN
throws away signal. This module implements a four-stage restoration
pipeline and two no-reference quality metrics so the benefit can be
measured, not just asserted.

Pipeline:  Gray-World white balance -> CLAHE (L channel, LAB) ->
           bilateral denoise -> unsharp mask

All functions take and return BGR uint8 images (OpenCV convention).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

try:                                    # works both as package and as script
    from . import config as C
except ImportError:                     # pragma: no cover
    import config as C


# ======================================================================
# STAGE 1 - COLOUR CORRECTION
# ======================================================================
def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Gray-World assumption: the average of a natural scene is achromatic.

    Underwater the red channel is heavily attenuated, so its mean is far
    below the grey mean. Scaling each channel toward the global mean
    restores the red content and removes the blue-green cast.
    """
    img_f = img.astype(np.float32)
    means = img_f.reshape(-1, 3).mean(axis=0)          # B, G, R means
    grey = means.mean()
    # guard against a dead channel (mean == 0)
    scale = grey / np.maximum(means, 1e-6)
    balanced = img_f * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ======================================================================
# STAGE 2 - CONTRAST ENHANCEMENT
# ======================================================================
def clahe_lab(img: np.ndarray,
              clip_limit: float = C.CLAHE_CLIP_LIMIT,
              tile_grid: tuple = C.CLAHE_TILE_GRID) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on luminance only.

    Working in LAB and touching only L means contrast improves without
    the colour shifts that per-channel equalisation in BGR introduces.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ======================================================================
# STAGE 3 - NOISE REDUCTION
# ======================================================================
def bilateral_denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving smoothing.

    Suspended particles ("marine snow") and sensor noise are suppressed
    while coral branch and polyp boundaries stay sharp - which a plain
    Gaussian blur would destroy.
    """
    return cv2.bilateralFilter(img,
                               d=C.DENOISE_DIAMETER,
                               sigmaColor=C.DENOISE_SIGMA_COLOR,
                               sigmaSpace=C.DENOISE_SIGMA_SPACE)


# ======================================================================
# STAGE 4 - DETAIL RECOVERY
# ======================================================================
def unsharp_mask(img: np.ndarray, amount: float = 0.6,
                 sigma: float = 1.5) -> np.ndarray:
    """Re-sharpen texture softened by scattering and by stage 3."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ======================================================================
# FULL PIPELINE
# ======================================================================
def enhance_underwater_image(img: np.ndarray) -> np.ndarray:
    """Run all four stages in order."""
    img = gray_world_white_balance(img)
    img = clahe_lab(img)
    img = bilateral_denoise(img)
    img = unsharp_mask(img)
    return img


# ======================================================================
# NO-REFERENCE QUALITY METRICS (for the ablation table in the paper)
# ======================================================================
def colourfulness(img: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric. Higher is better."""
    b, g, r = cv2.split(img.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram (information content)."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grey], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def contrast_rms(img: np.ndarray) -> float:
    """Root-mean-square contrast of the luminance channel."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(grey.std())


def quality_report(img: np.ndarray) -> dict:
    return {
        "colourfulness": round(colourfulness(img), 3),
        "entropy": round(entropy(img), 3),
        "rms_contrast": round(contrast_rms(img), 4),
    }


# ======================================================================
# BATCH PROCESSING
# ======================================================================
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def enhance_dataset(src_root: Path = C.RAW_DIR,
                    dst_root: Path = C.ENHANCED_DIR,
                    verbose: bool = True) -> dict:
    """Walk src_root/<class>/*.jpg, enhance every image, mirror the tree
    into dst_root. Returns mean quality metrics before and after, which
    is exactly the ablation evidence Table 2 of the paper needs.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    before, after, n_fail = [], [], 0

    for class_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        out_dir = dst_root / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in sorted(class_dir.iterdir())
                 if f.suffix.lower() in VALID_EXT]

        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:                     # corrupt / unreadable file
                n_fail += 1
                continue
            out = enhance_underwater_image(img)
            before.append(quality_report(img))
            after.append(quality_report(out))
            cv2.imwrite(str(out_dir / f"{f.stem}.jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, C.ENHANCED_JPEG_QUALITY])

        if verbose:
            print(f"  {class_dir.name:12s} {len(files):5d} images -> {out_dir}")

    def _mean(dicts):
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in keys}

    summary = {"n_images": len(after),
               "n_unreadable": n_fail,
               "before": _mean(before),
               "after": _mean(after)}
    if verbose:
        print(f"\nEnhanced {summary['n_images']} images "
              f"({n_fail} unreadable skipped)")
        print(f"  before: {summary['before']}")
        print(f"  after : {summary['after']}")
    return summary


def side_by_side(image_path: str | Path, out_path: str | Path) -> None:
    """Save a raw|enhanced comparison strip - use this for Figure 2."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    pair = np.hstack([img, enhance_underwater_image(img)])
    cv2.imwrite(str(out_path), pair)


if __name__ == "__main__":
    enhance_dataset()
