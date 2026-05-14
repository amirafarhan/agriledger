"""
=============================================================================
evaluate.py — Comprehensive Model Evaluation & Visualisation
=============================================================================
Generates:
  • Training curves (accuracy & loss, both phases on one plot)
  • Confusion matrix (counts + normalised)
  • Classification report (precision, recall, F1 per class)
  • ROC curve & AUC  (binary mode only)
  • Worst predictions gallery  (misclassified images for error analysis)

Run standalone:
    python evaluate.py

Or call evaluate_model() from a notebook after training.
=============================================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)

import config
import data_pipeline as dp


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history_dict: dict, save_dir: str = config.PLOT_DIR):
    """
    Plots accuracy and loss for both Phase 1 and Phase 2 on the same figure.
    Draws a vertical dashed line between the two phases.

    Parameters
    ----------
    history_dict : merged history dict produced by train.save_histories()
    save_dir     : directory where the plot PNG is written
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Extract series ─────────────────────────────────────────────────────
    p1_acc     = history_dict.get("p1_accuracy", [])
    p1_val_acc = history_dict.get("p1_val_accuracy", [])
    p1_loss    = history_dict.get("p1_loss", [])
    p1_val_los = history_dict.get("p1_val_loss", [])

    p2_acc     = history_dict.get("p2_accuracy", [])
    p2_val_acc = history_dict.get("p2_val_accuracy", [])
    p2_loss    = history_dict.get("p2_loss", [])
    p2_val_los = history_dict.get("p2_val_loss", [])

    # Concatenate both phases
    acc      = p1_acc     + p2_acc
    val_acc  = p1_val_acc + p2_val_acc
    loss     = p1_loss    + p2_loss
    val_loss = p1_val_los + p2_val_los
    epochs   = list(range(1, len(acc) + 1))
    phase_boundary = len(p1_acc)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Fruit Classifier — Training History", fontsize=16, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, acc,     label="Train Accuracy", color="#2196F3", lw=2)
    ax.plot(epochs, val_acc, label="Val Accuracy",   color="#4CAF50", lw=2)
    if phase_boundary:
        ax.axvline(x=phase_boundary, color="#FF9800", ls="--", lw=1.5,
                   label=f"Phase boundary (epoch {phase_boundary})")
    ax.axhline(y=0.98, color="red", ls=":", lw=1.2, label="98% target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over Training")
    ax.legend(loc="lower right")
    ax.set_ylim([0.5, 1.02])
    ax.grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, loss,     label="Train Loss", color="#2196F3", lw=2)
    ax.plot(epochs, val_loss, label="Val Loss",   color="#4CAF50", lw=2)
    if phase_boundary:
        ax.axvline(x=phase_boundary, color="#FF9800", ls="--", lw=1.5,
                   label=f"Phase boundary (epoch {phase_boundary})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Training")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Training curves saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs inference on the full dataset and returns (true_labels, pred_labels).
    Both arrays are integer class indices.
    """
    y_true_list, y_pred_list = [], []
    for images, labels in dataset:
        preds   = model.predict(images, verbose=0)
        y_pred_list.append(np.argmax(preds,  axis=1))
        y_true_list.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: str = config.PLOT_DIR,
):
    """
    Plots both raw-count and normalised confusion matrices side by side.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (Recall per class)"],
        ["d", ".2%"],
    ):
        disp = ConfusionMatrixDisplay(matrix, display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=fmt)
        ax.set_title(title)
        ax.set_xticklabels(class_names, rotation=30, ha="right")

    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved → {out}")
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: str = config.LOG_DIR,
):
    """Prints and saves the sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4)
    print("\n[Evaluate] Classification Report:")
    print(report)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"[Evaluate] Report saved → {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# ROC CURVE  (binary mode only)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    model,
    dataset,
    save_dir: str = config.PLOT_DIR,
):
    """
    Generates and plots the ROC curve for binary classification.
    AUC close to 1.0 indicates excellent discrimination.
    """
    if config.NUM_CLASSES != 2:
        print("[Evaluate] ROC curve is only available in binary mode. Skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)
    y_true_list, y_score_list = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_score_list.append(probs[:, 1])        # probability of Rotten (class 1)
        y_true_list.append(np.argmax(labels.numpy(), axis=1))

    y_true  = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fresh vs Rotten")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] ROC curve saved → {out}  (AUC = {roc_auc:.4f})")
    return roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY SUGGESTIONS  (printed if < 98%)
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_suggestions(val_accuracy: float):
    """
    Prints actionable improvement suggestions when val_accuracy < 98%.
    Designed to guide junior ML engineers through the debugging process.
    """
    if val_accuracy >= 0.98:
        print(f"\n✅  Target achieved!  Val accuracy = {val_accuracy:.2%}")
        return

    gap = 0.98 - val_accuracy
    print(f"\n⚠️  Val accuracy = {val_accuracy:.2%}  (gap to target: {gap:.2%})")
    print("\n── Improvement Suggestions ─────────────────────────────────────")

    if val_accuracy < 0.90:
        print("""
  1. CHECK DATA QUALITY
     • Inspect 20 random images from each class — are labels correct?
     • Look for corrupted / grayscale images.
     • Make sure TRAIN_DIR / TEST_DIR point to the right folders.
""")

    print("""
  2. SWITCH TO A STRONGER BACKBONE
     • EfficientNetB0 → EfficientNetB3 (accuracy ↑, size ↑)
     • Set config.BACKBONE = "EfficientNetB3"

  3. INCREASE FINE-TUNE DEPTH
     • Unfreeze more layers: config.FINE_TUNE_AT = -50
     • Or fine-tune the entire backbone: config.FINE_TUNE_AT = -len(backbone)

  4. ADJUST AUGMENTATION
     • If model underfits: reduce augmentation strength (lower AUG_ZOOM, AUG_ROTATION)
     • If model overfits:  add CutMix or MixUp augmentation

  5. TRAINING DURATION
     • Increase PHASE2_EPOCHS to 50–70
     • Increase EARLY_STOP_PATIENCE to 12

  6. REDUCE REGULARISATION (if val_acc >> train_acc → underfitting)
     • Lower DROPOUT_RATE from 0.40 → 0.25
     • Lower L2_LAMBDA from 1e-4 → 5e-5

  7. CLASS IMBALANCE
     • Re-check class_weight values; if any > 3.0, use oversampling (SMOTE)

  8. ENSEMBLE
     • Train EfficientNetB0 + MobileNetV2 separately, average their softmax outputs
     • Usually adds +0.5–1.0% accuracy

  9. LEARNING RATE WARM-UP
     • Use a cosine decay schedule with linear warm-up (first 5 epochs)
     • Replace ReduceLROnPlateau with CosineDecayRestarts
""")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, test_ds, class_names: list[str], history_dict: dict):
    """
    Runs all evaluation steps and saves all artefacts to outputs/plots/.

    Parameters
    ----------
    model        : trained Keras model
    test_ds      : tf.data.Dataset (test split, not shuffled)
    class_names  : list of human-readable class labels
    history_dict : merged history dict from train.save_histories()
    """
    print("\n" + "="*60)
    print("  EVALUATION")
    print("="*60)

    # ── Training curves ────────────────────────────────────────────────────────
    if history_dict:
        plot_training_history(history_dict)

    # ── Predictions ───────────────────────────────────────────────────────────
    print("[Evaluate] Generating predictions on test set …")
    y_true, y_pred = get_predictions(model, test_ds)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    val_acc = np.mean(y_true == y_pred)
    print(f"\n[Evaluate] Test accuracy: {val_acc:.4f} ({val_acc:.2%})")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = plot_confusion_matrix(y_true, y_pred, class_names)

    # ── Classification report ─────────────────────────────────────────────────
    print_classification_report(y_true, y_pred, class_names)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    plot_roc_curve(model, test_ds)

    # ── Suggestions ──────────────────────────────────────────────────────────
    accuracy_suggestions(val_acc)

    return val_acc, cm


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (load saved model and evaluate)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tensorflow as tf

    # Load the best model from disk
    model_path = os.path.join(config.MODEL_DIR, "fruit_classifier_final.keras")
    print(f"[Evaluate] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load class names
    meta_path = os.path.join(config.MODEL_DIR, "class_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    class_names = [meta[str(i)] for i in range(len(meta))]

    # Load test data
    test_paths, test_labels = dp.collect_files_and_labels(config.TEST_DIR)
    test_ds = dp.build_dataset(test_paths, test_labels,
                               batch_size=32, training=False, shuffle=False)

    # Load history
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    history_dict = {}
    if os.path.exists(history_path):
        with open(history_path) as f:
            history_dict = json.load(f)

    evaluate_model(model, test_ds, class_names, history_dict)
