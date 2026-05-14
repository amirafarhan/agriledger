"""
=============================================================================
callbacks.py — Keras Callbacks for Training Control
=============================================================================
Centralises all callback configuration so train.py stays clean.

Callbacks used:
  1. ModelCheckpoint  → save the best model (by val_accuracy)
  2. EarlyStopping    → halt training when val_accuracy stops improving
  3. ReduceLROnPlateau→ halve the LR when val_loss plateaus
  4. TensorBoard      → log metrics for visual inspection
  5. CSVLogger        → human-readable metric logs
  6. LearningRateLogger (custom) → records LR per epoch to history
=============================================================================
"""

import os
import tensorflow as tf
from tensorflow import keras

import config


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK: Log the current learning rate to history
# ─────────────────────────────────────────────────────────────────────────────

class LearningRateLogger(keras.callbacks.Callback):
    """Records the current learning rate at the end of every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        if logs is not None:
            logs["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(phase: int, checkpoint_path: str) -> list:
    """
    Returns a list of Keras callbacks configured for the given training phase.

    Parameters
    ----------
    phase           : 1 or 2  (used only for naming log sub-directories)
    checkpoint_path : where to save the best checkpoint .keras file
    """
    tb_log_dir = os.path.join(config.LOG_DIR, f"phase{phase}")
    csv_path   = os.path.join(config.LOG_DIR, f"phase{phase}_metrics.csv")

    callbacks = [

        # ── 1. Save best model ────────────────────────────────────────────────
        # Monitors val_accuracy and saves only when it improves.
        # This means the final checkpoint = the best epoch, not the last one.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,   # save full model, not just weights
            verbose=1,
        ),

        # ── 2. Early stopping ─────────────────────────────────────────────────
        # Stops training if val_accuracy hasn't improved for PATIENCE epochs.
        # restore_best_weights=True → model reverts to its best checkpoint.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),

        # ── 3. Reduce LR on plateau ───────────────────────────────────────────
        # If val_loss doesn't decrease for PATIENCE epochs, multiply LR by FACTOR.
        # This gives the optimizer a "nudge" to find a better local minimum.
        # min_lr prevents the LR from collapsing to zero.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),

        # ── 4. TensorBoard ────────────────────────────────────────────────────
        # Launch with:  tensorboard --logdir outputs/logs
        # Then open:    http://localhost:6006
        keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,          # weight histograms every epoch
            write_graph=True,
            update_freq="epoch",
        ),

        # ── 5. CSV Logger ─────────────────────────────────────────────────────
        # Plain-text backup of all metrics — readable without TensorBoard.
        keras.callbacks.CSVLogger(
            filename=csv_path,
            separator=",",
            append=False,
        ),

        # ── 6. Custom LR logger ───────────────────────────────────────────────
        LearningRateLogger(),
    ]

    return callbacks
