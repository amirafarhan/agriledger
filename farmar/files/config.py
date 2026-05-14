"""
=============================================================================
config.py — Central Configuration for Fruit Freshness Classifier
=============================================================================
All hyperparameters, paths, and runtime flags live here.
Junior engineers: change values HERE rather than hunting through code files.
=============================================================================
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────────────────────────────────────
# After downloading from Kaggle, unzip into a folder called 'dataset'.
# Expected structure:
#   dataset/
#     train/
#       freshapples/   freshbananas/   freshoranges/
#       rottenapples/ rottenbananas/  rottenoranges/
#     test/
#       (same sub-folders)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR       = os.path.join(DATASET_DIR, "train")
TEST_DIR        = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR        = os.path.join(OUTPUT_DIR, "plots")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)      # EfficientNet / MobileNet native size
IMG_CHANNELS    = 3               # RGB
INPUT_SHAPE     = (*IMG_SIZE, IMG_CHANNELS)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────
# The Kaggle dataset has 6 fine-grained classes:
#   freshapples, freshbananas, freshoranges,
#   rottenapples, rottenbananas, rottenoranges
#
# We expose BOTH modes:
#   • BINARY_MODE = True  → 2 classes  (Fresh / Rotten)  ← default for supply chain
#   • BINARY_MODE = False → 6 classes  (fine-grained)
BINARY_MODE     = True
NUM_CLASSES     = 2 if BINARY_MODE else 6

# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE CHOICE
# ─────────────────────────────────────────────────────────────────────────────
# Options: "EfficientNetB0"  |  "EfficientNetB3"  |  "MobileNetV2"
# MobileNetV2  → smallest, fastest  (good for IoT / mobile edge)
# EfficientNetB0 → best accuracy/size trade-off (recommended default)
# EfficientNetB3 → highest accuracy, larger model
BACKBONE        = "EfficientNetB0"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — PHASE 1  (Frozen backbone, train classifier head only)
# ─────────────────────────────────────────────────────────────────────────────
PHASE1_EPOCHS       = 15
PHASE1_LR           = 1e-3
PHASE1_BATCH_SIZE   = 32

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — PHASE 2  (Fine-tune last N layers of backbone)
# ─────────────────────────────────────────────────────────────────────────────
PHASE2_EPOCHS       = 30
PHASE2_LR           = 1e-4          # 10× smaller than phase 1
PHASE2_BATCH_SIZE   = 16            # smaller batch → better generalisation
FINE_TUNE_AT        = -30           # unfreeze last 30 layers (negative index)

# ─────────────────────────────────────────────────────────────────────────────
# REGULARISATION
# ─────────────────────────────────────────────────────────────────────────────
DROPOUT_RATE        = 0.40
L2_LAMBDA           = 1e-4
LABEL_SMOOTHING     = 0.05          # prevents overconfidence → better calibration

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 8
REDUCE_LR_PATIENCE  = 4
REDUCE_LR_FACTOR    = 0.3
MIN_LR              = 1e-7

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION (training only — validation uses no augmentation)
# ─────────────────────────────────────────────────────────────────────────────
AUG_ROTATION        = 20            # degrees
AUG_ZOOM            = 0.20
AUG_BRIGHTNESS      = (0.75, 1.25)
AUG_CONTRAST        = 0.15
AUG_HFLIP           = True
AUG_VFLIP           = False         # fruit orientation is upright
AUG_SHEAR           = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SPLIT  (only used when there is no separate val set)
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_SPLIT    = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED         = 42

# ─────────────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────────────
MIXED_PRECISION     = True          # fp16 on GPU → 2× speed, same accuracy
VERBOSE             = 1             # Keras verbosity (0=silent, 1=bar, 2=line)
