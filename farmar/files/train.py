"""
=============================================================================
train.py — Two-Phase Training Pipeline
=============================================================================
Phase 1: Train only the classification head (backbone frozen)
         → Fast convergence, protects pretrained features
         → Typically 15 epochs, lr = 1e-3

Phase 2: Fine-tune the last 30 backbone layers + head
         → Adapts high-level features to fruit domain
         → Typically 30 epochs, lr = 1e-4 (10× smaller!)

Run this file directly:
    python train.py

Or import and call train_pipeline() from a notebook / orchestrator.
=============================================================================
"""

import os
import json
import random
import numpy as np
import tensorflow as tf

import config
import data_pipeline as dp
import model as model_module
from callbacks import build_callbacks


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# MIXED PRECISION  (fp16 on GPU → ~2× throughput)
# ─────────────────────────────────────────────────────────────────────────────

def configure_mixed_precision():
    if config.MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[Train] Mixed precision enabled (fp16 compute / fp32 weights)")
    else:
        print("[Train] Mixed precision disabled — using fp32")


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

def create_output_dirs():
    for path in [config.OUTPUT_DIR, config.MODEL_DIR,
                 config.LOG_DIR, config.PLOT_DIR]:
        os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads file paths and labels from the Kaggle dataset folder.
    Returns compiled tf.data datasets for train, validation, and test.
    """
    # ── Collect all file paths and labels ────────────────────────────────────
    train_paths, train_labels = dp.collect_files_and_labels(config.TRAIN_DIR)
    test_paths,  test_labels  = dp.collect_files_and_labels(config.TEST_DIR)

    # ── Stratified train/val split from the training set ─────────────────────
    (train_paths, train_labels), (val_paths, val_labels) = dp.train_val_split(
        train_paths, train_labels
    )

    # ── Compute class weights (handle imbalance) ──────────────────────────────
    class_weights = dp.compute_class_weights(train_labels)

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    train_ds = dp.build_dataset(
        train_paths, train_labels,
        batch_size=config.PHASE1_BATCH_SIZE,
        training=True, shuffle=True,
    )
    val_ds = dp.build_dataset(
        val_paths, val_labels,
        batch_size=config.PHASE1_BATCH_SIZE,
        training=False, shuffle=False,
    )
    test_ds = dp.build_dataset(
        test_paths, test_labels,
        batch_size=config.PHASE1_BATCH_SIZE,
        training=False, shuffle=False,
    )

    # ── Save class metadata for the inference server ──────────────────────────
    class_names = dp.build_class_names(config.TRAIN_DIR)
    dp.save_class_metadata(
        class_names,
        os.path.join(config.MODEL_DIR, "class_metadata.json"),
    )

    print(f"\n[Train] Class names: {class_names}")
    return train_ds, val_ds, test_ds, class_weights, train_paths, val_paths, test_paths


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — FROZEN BACKBONE
# ─────────────────────────────────────────────────────────────────────────────

def phase1_training(train_ds, val_ds, class_weights):
    """
    Trains only the custom classification head.
    The backbone is completely frozen — its ImageNet weights are preserved.

    Why freeze first?
    • The head starts with random weights; a large lr would corrupt the
      pretrained backbone features before the head has learned anything.
    • Freezing lets the head converge to a sensible initialisation,
      making Phase 2 fine-tuning stable and effective.
    """
    print("\n" + "="*60)
    print("  PHASE 1  —  Training classifier head (backbone frozen)")
    print("="*60)

    # Build model with frozen backbone
    fruit_model, _ = model_module.build_model(trainable=False)
    fruit_model    = model_module.compile_model(fruit_model, config.PHASE1_LR)
    fruit_model.summary(line_length=80)

    # Callbacks for phase 1
    cbs = build_callbacks(
        phase=1,
        checkpoint_path=os.path.join(config.MODEL_DIR, "phase1_best.keras"),
    )

    history1 = fruit_model.fit(
        train_ds,
        epochs=config.PHASE1_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=config.VERBOSE,
    )

    print(f"\n[Phase 1] Best val accuracy: "
          f"{max(history1.history['val_accuracy']):.4f}")

    return fruit_model, history1


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PARTIAL FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def phase2_training(fruit_model, train_ds, val_ds, class_weights):
    """
    Unfreezes the last 30 backbone layers and fine-tunes with a very low LR.

    Why a smaller LR?
    • The pretrained backbone layers have carefully tuned weights.
    • A large LR would destroy them.  1e-4 (10× smaller than Phase 1) makes
      tiny, controlled updates that adapt the feature extractor to fruit images
      without catastrophic forgetting.

    Why reduce batch size?
    • Smaller batches → noisier gradients → better generalisation (acts like
      implicit regularisation).
    """
    print("\n" + "="*60)
    print("  PHASE 2  —  Fine-tuning last 30 backbone layers")
    print("="*60)

    # Partially unfreeze backbone
    fruit_model = model_module.unfreeze_backbone(fruit_model, config.FINE_TUNE_AT)

    # Recompile AFTER changing trainable status (mandatory in Keras)
    fruit_model = model_module.compile_model(fruit_model, config.PHASE2_LR)

    # Rebuild dataset with smaller batch size for phase 2
    # (re-use file paths stored in the dataset metadata)
    cbs = build_callbacks(
        phase=2,
        checkpoint_path=os.path.join(config.MODEL_DIR, "phase2_best.keras"),
    )

    history2 = fruit_model.fit(
        train_ds,
        epochs=config.PHASE2_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=config.VERBOSE,
    )

    print(f"\n[Phase 2] Best val accuracy: "
          f"{max(history2.history['val_accuracy']):.4f}")

    return fruit_model, history2


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def save_model(fruit_model):
    """
    Saves in two formats:
    1. .keras (full model — preferred for resuming training or Python serving)
    2. SavedModel directory (TensorFlow Serving / TFLite conversion)
    """
    keras_path = os.path.join(config.MODEL_DIR, "fruit_classifier_final.keras")
    saved_path = os.path.join(config.MODEL_DIR, "fruit_classifier_savedmodel")

    fruit_model.save(keras_path)
    fruit_model.export(saved_path)

    print(f"\n[Train] Model saved:")
    print(f"  Keras format:       {keras_path}")
    print(f"  SavedModel format:  {saved_path}")

    # Also save training config for reproducibility / audit trail
    cfg_snapshot = {
        "backbone":       config.BACKBONE,
        "binary_mode":    config.BINARY_MODE,
        "num_classes":    config.NUM_CLASSES,
        "img_size":       config.IMG_SIZE,
        "phase1_lr":      config.PHASE1_LR,
        "phase2_lr":      config.PHASE2_LR,
        "dropout":        config.DROPOUT_RATE,
        "fine_tune_at":   config.FINE_TUNE_AT,
        "label_smoothing": config.LABEL_SMOOTHING,
    }
    cfg_path = os.path.join(config.MODEL_DIR, "training_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_snapshot, f, indent=2)
    print(f"  Training config:    {cfg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY PERSISTENCE  (for plotting after training)
# ─────────────────────────────────────────────────────────────────────────────

def save_histories(history1, history2):
    """Saves raw training history dicts as JSON for later plotting / analysis."""
    merged = {}
    for key in history1.history:
        merged[f"p1_{key}"] = [float(v) for v in history1.history[key]]
    for key in history2.history:
        merged[f"p2_{key}"] = [float(v) for v in history2.history[key]]

    path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[Train] Training history saved → {path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline():
    """
    End-to-end training pipeline.  Call this function to run full training.
    """
    set_seeds()
    configure_mixed_precision()
    create_output_dirs()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, class_weights, *paths = load_data()

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    fruit_model, history1 = phase1_training(train_ds, val_ds, class_weights)

    # ── Rebuild train_ds with smaller phase-2 batch size ─────────────────────
    train_paths, val_paths, test_paths = paths
    train_labels = [dp.get_binary_label(p.split(os.sep)[-2])
                    if config.BINARY_MODE
                    else None
                    for p in train_paths]

    # Simpler: rebuild full pipeline from scratch for phase 2 batch size
    train_ds_p2 = dp.build_dataset(
        train_paths,
        [dp.get_binary_label(os.path.basename(os.path.dirname(p)))
         if config.BINARY_MODE
         else int(os.path.basename(os.path.dirname(p)))
         for p in train_paths],
        batch_size=config.PHASE2_BATCH_SIZE,
        training=True, shuffle=True,
    )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    fruit_model, history2 = phase2_training(fruit_model, train_ds_p2, val_ds,
                                             class_weights)

    # ── Persist ──────────────────────────────────────────────────────────────
    save_model(fruit_model)
    merged_history = save_histories(history1, history2)

    # ── Final evaluation on held-out test set ─────────────────────────────────
    print("\n[Train] Evaluating on test set …")
    test_results = fruit_model.evaluate(test_ds, verbose=config.VERBOSE)
    metrics_names = fruit_model.metrics_names
    for name, val in zip(metrics_names, test_results):
        print(f"  test_{name}: {val:.4f}")

    return fruit_model, merged_history


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_pipeline()
