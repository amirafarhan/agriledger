"""
=============================================================================
data_pipeline.py — Data Loading, Augmentation & Class Mapping
=============================================================================
Handles everything between raw image files and model-ready tensors.

Key design decisions:
  • Uses tf.data API for GPU-prefetched pipelines (no I/O bottleneck)
  • Augmentation is applied ONLY to training data
  • Binary mode collapses 6 labels → 2 (Fresh / Rotten) automatically
  • Class weights are computed to handle any imbalance
=============================================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import Counter

import config


# ─────────────────────────────────────────────────────────────────────────────
# CLASS MAPPING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Fine-grained label → binary label mapping
# Keywords "fresh" or "rotten" are detected in the folder name.
BINARY_LABEL_MAP = {
    "fresh": 0,   # class 0 = Fresh
    "rotten": 1,  # class 1 = Rotten
}

def get_binary_label(folder_name: str) -> int:
    """
    Maps a fine-grained folder name (e.g. 'freshapples') to a binary int.
    Raises ValueError if neither 'fresh' nor 'rotten' appears in the name.
    """
    name_lower = folder_name.lower()
    for keyword, label in BINARY_LABEL_MAP.items():
        if keyword in name_lower:
            return label
    raise ValueError(
        f"Cannot determine binary label for folder '{folder_name}'. "
        "Folder must contain 'fresh' or 'rotten' in its name."
    )


def build_class_names(data_dir: str) -> list[str]:
    """
    Returns sorted list of sub-folder names (= raw class names).
    In binary mode this will be ['Fresh', 'Rotten'].
    In fine-grained mode this will be the 6 original folder names.
    """
    if config.BINARY_MODE:
        return ["Fresh", "Rotten"]
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# FILE COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_files_and_labels(data_dir: str) -> tuple[list[str], list[int]]:
    """
    Walks `data_dir`, collects all image paths and their integer labels.

    Supports JPEG, JPG, PNG.  In BINARY_MODE the 6 fine-grained labels are
    collapsed to 0 (Fresh) / 1 (Rotten).

    Returns
    -------
    file_paths : list of absolute path strings
    labels     : list of integer class indices (same length as file_paths)
    """
    file_paths, labels = [], []
    subfolders = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # Build a fine-grained → integer mapping (needed for multi-class mode)
    fine_grained_map = {name: idx for idx, name in enumerate(subfolders)}

    for folder_name in subfolders:
        folder_path = os.path.join(data_dir, folder_name)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full_path = os.path.join(folder_path, fname)
            file_paths.append(full_path)
            if config.BINARY_MODE:
                labels.append(get_binary_label(folder_name))
            else:
                labels.append(fine_grained_map[folder_name])

    print(f"[DataPipeline] Collected {len(file_paths)} images from '{data_dir}'")
    dist = Counter(labels)
    label_names = build_class_names(data_dir)
    for idx, count in sorted(dist.items()):
        print(f"  Class {idx} ({label_names[idx]}): {count} images")

    return file_paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """
    Returns balanced class weights: weight[c] = total / (n_classes * count[c]).

    In a balanced dataset all weights ≈ 1.0.
    In an imbalanced dataset the minority class gets a higher weight so the
    model pays more attention to it.
    """
    counts = Counter(labels)
    n_total = len(labels)
    n_classes = len(counts)
    weights = {
        cls: n_total / (n_classes * count)
        for cls, count in counts.items()
    }
    print(f"[DataPipeline] Class weights: {weights}")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """
    Reads a JPEG/PNG from disk, decodes it, resizes, and normalises to [0, 1].
    This function is mapped over the tf.data pipeline on CPU.
    """
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=config.IMG_CHANNELS,
                                  expand_animations=False)
    image = tf.image.resize(image, config.IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0   # normalise to [0, 1]
    return image, label


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION  (training only)
# ─────────────────────────────────────────────────────────────────────────────

def augment(image: tf.Tensor, label: tf.Tensor):
    """
    Applies stochastic augmentations to a single (image, label) pair.

    All transforms are differentiable-friendly and run inside the tf.data
    pipeline on CPU (separate from GPU training) — so augmentation is
    effectively free in terms of training time.
    """
    # ── Horizontal flip ──────────────────────────────────────────────────────
    if config.AUG_HFLIP:
        image = tf.image.random_flip_left_right(image)

    # ── Vertical flip (disabled for fruits — kept for completeness) ───────────
    if config.AUG_VFLIP:
        image = tf.image.random_flip_up_down(image)

    # ── Brightness & contrast ─────────────────────────────────────────────────
    image = tf.image.random_brightness(image, max_delta=1 - config.AUG_BRIGHTNESS[0])
    image = tf.image.random_contrast(image,
                                     lower=1 - config.AUG_CONTRAST,
                                     upper=1 + config.AUG_CONTRAST)

    # ── Saturation & hue (helps generalise across different lighting) ─────────
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)

    # ── Rotation & zoom via random crop+resize ────────────────────────────────
    # Crop a random region (between 80% and 100% of original size), then
    # resize back to IMG_SIZE.  This simultaneously simulates zoom and slight
    # translation.
    crop_frac = tf.random.uniform([], 1.0 - config.AUG_ZOOM, 1.0)
    crop_size = tf.cast(
        tf.cast(config.IMG_SIZE, tf.float32) * crop_frac, tf.int32
    )
    image = tf.image.random_crop(image, size=[*crop_size, config.IMG_CHANNELS])
    image = tf.image.resize(image, config.IMG_SIZE)

    # ── Clip to valid range after all transforms ──────────────────────────────
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# ─────────────────────────────────────────────────────────────────────────────
# tf.data DATASET BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

AUTOTUNE = tf.data.AUTOTUNE


def build_dataset(
    file_paths: list[str],
    labels: list[int],
    batch_size: int,
    training: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Converts file_paths + labels into a high-performance tf.data.Dataset.

    Pipeline:
      shuffle → load image → (augment if training) → batch → prefetch

    Parameters
    ----------
    file_paths : list of image file path strings
    labels     : list of integer class indices
    batch_size : number of samples per batch
    training   : if True, apply augmentation
    shuffle    : if True, shuffle before each epoch
    """
    n_classes = config.NUM_CLASSES

    # One-hot encode labels for categorical cross-entropy
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=n_classes)

    ds = tf.data.Dataset.from_tensor_slices(
        (file_paths, one_hot_labels.tolist())
    )

    if shuffle:
        # Buffer = full dataset → perfect shuffle.  Reduce if RAM is limited.
        ds = ds.shuffle(buffer_size=len(file_paths), seed=config.RANDOM_SEED,
                        reshuffle_each_iteration=True)

    # Read & preprocess images in parallel
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    # Augmentation (training only)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    # Batch, drop the last incomplete batch during training
    ds = ds.batch(batch_size, drop_remainder=training)

    # Prefetch next batch to GPU while current batch is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATION SPLIT  (when no separate val folder exists)
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(
    file_paths: list[str],
    labels: list[int],
    val_fraction: float = config.VALIDATION_SPLIT,
    seed: int = config.RANDOM_SEED,
):
    """
    Splits (file_paths, labels) into train and validation sets while
    preserving class distribution (stratified split).

    Returns
    -------
    (train_paths, train_labels), (val_paths, val_labels)
    """
    rng = np.random.default_rng(seed)
    file_paths = np.array(file_paths)
    labels     = np.array(labels)

    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx   = idx[:n_val]
        train_idx = idx[n_val:]
        val_paths.extend(file_paths[val_idx])
        val_labels.extend(labels[val_idx])
        train_paths.extend(file_paths[train_idx])
        train_labels.extend(labels[train_idx])

    print(f"[DataPipeline] Train: {len(train_paths)} | Val: {len(val_paths)}")
    return (train_paths, train_labels), (val_paths, val_labels)


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXPORT  (for deployment — maps int index → class name)
# ─────────────────────────────────────────────────────────────────────────────

def save_class_metadata(class_names: list[str], output_path: str):
    """
    Saves class index → name mapping as JSON.
    The inference server loads this to convert model output back to labels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meta = {str(i): name for i, name in enumerate(class_names)}
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DataPipeline] Class metadata saved → {output_path}")
