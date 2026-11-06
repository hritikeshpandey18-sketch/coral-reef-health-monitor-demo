"""
inference.py
Prediction on NEW photographs, prepared exactly like the training data.

Training images were (1) about 300 px on the longest side, (2) enhanced,
(3) saved as JPEG q95, (4) decoded and bilinearly resized to 224x224 by
tf.data. `prepare()` reproduces those four steps for an arbitrary upload;
skipping them (the original app did) changes the prediction for ~7 % of
validation images, and more for large phone photos.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from . import config as C
    from .preprocessing import enhance_underwater_image
except ImportError:                     # pragma: no cover
    import config as C
    from preprocessing import enhance_underwater_image


def prepare(bgr: np.ndarray, enhance: bool = True):
    """Return (processed_bgr, model_input).

    processed_bgr : the image at training scale after (optional)
                    enhancement - what the model actually "sees", for display
    model_input   : float32 (1, H, W, 3) RGB in [0, 255]; each model applies
                    its own backbone preprocessing internally
    """
    import tensorflow as tf

    h, w = bgr.shape[:2]
    if max(h, w) > C.INFER_MAX_SIDE:
        s = C.INFER_MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    if enhance:
        bgr = enhance_underwater_image(bgr)
        ok, buf = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, C.ENHANCED_JPEG_QUALITY])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = tf.image.resize(rgb, C.IMG_SIZE, method="bilinear").numpy().astype("float32")
    return bgr, x[None]


def predict(models: dict, x: np.ndarray) -> dict:
    """Run one or more models on a prepared input.

    models : {name: keras model}. With several, the equal-weight mean of
             their softmax outputs is the prediction (the ensemble whose CV
             accuracy is in reports/table8_ensemble_crossval.json).
    Returns probs (mean), per-model probs, predicted index and agreement.
    """
    per = {n: m.predict(x, verbose=0)[0] for n, m in models.items()}
    probs = np.mean(list(per.values()), axis=0)
    idx = int(probs.argmax())
    votes = sum(int(p.argmax() == idx) for p in per.values())
    return {"probs": probs, "per_model": per, "index": idx,
            "agreement": votes / len(per)}
