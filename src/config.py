"""
config.py
Central configuration for the AI-Based Coral Reef Health Monitoring project.

Every other module imports from here, so you change a path or a
hyper-parameter in ONE place and the whole pipeline follows.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------------
# On Google Colab set PROJECT_ROOT = Path("/content/coral")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded, untouched
ENHANCED_DIR = DATA_DIR / "enhanced"  # after underwater image enhancement
SPLIT_DIR = DATA_DIR / "splits"       # train / val / test folders
# Same photographs, same seed, but un-enhanced: the raw arm of the
# preprocessing ablation. Its test set holds the same images as SPLIT_DIR's.
SPLIT_RAW_DIR = DATA_DIR / "splits_raw"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"   # saved models
REPORT_DIR = PROJECT_ROOT / "reports"       # figures, tables, logs
# Copies of the app's "Try a sample" test photos, committed so a hosted
# demo without data/ still has them (streamlit_cloud/). Read-only.
DEMO_SAMPLE_DIR = PROJECT_ROOT / "docs" / "samples"

for _d in (RAW_DIR, ENHANCED_DIR, SPLIT_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. DATASET
# ----------------------------------------------------------------------
# Class names are auto-discovered from folder names, but declaring the
# expected order here keeps confusion matrices consistent across runs.
# Use the 2-class list if you only have healthy/bleached data.
# This project uses the Kaggle BC-HC dataset:
#   vencerlanz09/healthy-and-bleached-corals-image-classification
#   923 images -> healthy (438), bleached (485)
# The download contains byte-identical duplicates: 9 images filed under BOTH
# classes and 9 same-class pairs. download_data.deduplicate() removes them
# (conflicts dropped entirely, one copy of each pair kept), leaving 896
# images -> bleached (473), healthy (423). See reports/dedup_log.csv.
CLASS_NAMES = ["bleached", "healthy"]   # alphabetical = the order Keras assigns
# CLASS_NAMES = ["bleached", "dead", "healthy"]   # if you add a third source

NUM_CLASSES = len(CLASS_NAMES)

SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train, validation, test
# 923 images is small: a 15% test split is only ~139 images, so a single
# split has a wide confidence interval. Use src/crossval.py for the
# headline numbers and keep the fixed split for figures and the demo.
CV_FOLDS = 5
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# 3. IMAGE / TRAINING HYPER-PARAMETERS
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)      # all three backbones expect 224x224
BATCH_SIZE = 16            # fixed for the 4 GB VRAM; identical for every model

EPOCHS_HEAD = 15           # stage 1: frozen backbone, train classifier head
EPOCHS_FINETUNE = 20       # stage 2: unfreeze top block, low LR
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_AT = 0.75        # unfreeze the top 25% of backbone layers
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 6

# ----------------------------------------------------------------------
# 4. UNDERWATER ENHANCEMENT
# ----------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
DENOISE_DIAMETER = 5       # bilateral filter neighbourhood
DENOISE_SIGMA_COLOR = 50
DENOISE_SIGMA_SPACE = 50

# ----------------------------------------------------------------------
# 5. BACKBONES COMPARED IN THE PAPER
# ----------------------------------------------------------------------
BACKBONES = ["mobilenetv2", "resnet50", "efficientnetb0", "densenet121"]
MODELS = ["baseline"] + BACKBONES      # every model in the comparison tables
# Demo-app default: equal-weight mean of the ImageNet backbones' softmax.
# Its CV accuracy is measured from saved out-of-fold predictions
# (report.ensemble_tables -> reports/table8_ensemble_crossval.json).
ENSEMBLE_MEMBERS = BACKBONES

# ----------------------------------------------------------------------
# 6. INFERENCE ON NEW PHOTOS
# ----------------------------------------------------------------------
# 95 % of training photos are exactly 300 px on the longest side, and the
# enhanced training images were stored as JPEG q95. New uploads are brought
# to the same state before enhancement; on the validation split this lifts
# agreement with the training pipeline from 93 % to 98 % (91 % -> 95.5 %
# for phone-sized photos).
INFER_MAX_SIDE = 300
ENHANCED_JPEG_QUALITY = 95
PRIMARY_MODEL = "mobilenetv2"          # fixed a priori; used for the ablation

# ----------------------------------------------------------------------
# 7. COMMUNITY CONTRIBUTIONS (src/contrib.py)
# ----------------------------------------------------------------------
# Opt-in uploads collected by the web app for a FUTURE dataset. They are
# never part of data/splits and never enter any number in the paper.
# Credentials come from .env (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN,
# CORAL_ADMIN_TOKEN); without them contributions are disabled.
ENV_FILE = PROJECT_ROOT / ".env"
CONTRIB_DIR = DATA_DIR / "contrib"            # local replica of the Turso DB
CONTRIB_EXPORT_DIR = DATA_DIR / "contributed"  # confirmed images, by label
CONTRIB_LABELS = CLASS_NAMES + ["unsure"]     # what an uploader may answer
CONTRIB_JPEG_QUALITY = 95                     # stored at INFER_MAX_SIDE, EXIF stripped
# dHash Hamming distance at or below which an upload counts as a copy of a
# dataset image (resized / re-saved). Stops held-out test photos leaking
# into any future training set via the "Test samples" page.
CONTRIB_NEAR_DUP_BITS = 6

# ----------------------------------------------------------------------
# 8. REEF ATLAS (web/src/components/Reefs.jsx and app.py)
# ----------------------------------------------------------------------
# Background page on world reefs; no model output involved. One data file
# feeds both interfaces so they cannot drift apart.
WEB_DIR = PROJECT_ROOT / "web"
REEF_ATLAS_JSON = WEB_DIR / "src" / "data" / "reefs.json"
WORLD_LAND_JSON = WEB_DIR / "src" / "data" / "land-110m.json"   # world-atlas 2.0.2, 1:110m
