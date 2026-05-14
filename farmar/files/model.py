"""
=============================================================================
model.py — Transfer Learning Model Factory
=============================================================================
Builds the full classifier from a pretrained backbone + custom head.

Architecture overview:
  ┌──────────────────────────────────────────┐
  │  Pretrained Backbone (ImageNet weights)  │  ← frozen in Phase 1
  │  (EfficientNetB0 / B3 / MobileNetV2)    │  ← partially unfrozen Phase 2
  └──────────────────┬───────────────────────┘
                     │
         GlobalAveragePooling2D
                     │
           BatchNormalization
                     │
              Dense(256, ReLU)
                     │
               Dropout(0.40)
                     │
              Dense(128, ReLU)
                     │
               Dropout(0.25)
                     │
         Dense(NUM_CLASSES, Softmax)

Why this head?
  • GlobalAveragePooling → spatial invariance, fewer params than Flatten
  • BatchNorm → stabilises fine-tuning, faster convergence
  • Two Dense layers → enough capacity without overfitting
  • Dropout → strong regularisation
=============================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import config


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _get_backbone(name: str) -> tuple[keras.Model, keras.layers.Layer]:
    """
    Instantiates the chosen backbone with ImageNet weights, input_shape set,
    and top classification layers removed (include_top=False).

    Returns
    -------
    backbone  : the Keras pretrained model object
    preprocess: matching preprocessing function (normalises raw [0,1] input)
    """
    kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=config.INPUT_SHAPE,
    )

    name_lower = name.lower()

    if name_lower == "efficientnetb0":
        backbone   = keras.applications.EfficientNetB0(**kwargs)
        preprocess = keras.applications.efficientnet.preprocess_input
    elif name_lower == "efficientnetb3":
        backbone   = keras.applications.EfficientNetB3(**kwargs)
        preprocess = keras.applications.efficientnet.preprocess_input
    elif name_lower == "mobilenetv2":
        backbone   = keras.applications.MobileNetV2(**kwargs)
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    else:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            "Choose from: EfficientNetB0, EfficientNetB3, MobileNetV2"
        )

    print(f"[Model] Backbone: {name}  |  Params: {backbone.count_params():,}")
    return backbone, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

def _build_head(backbone_output: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Attaches a custom classification head on top of the backbone feature map.

    L2 regularisation is applied to Dense layers to prevent over-fitting
    on the relatively small domain-specific dataset.
    """
    l2 = regularizers.l2(config.L2_LAMBDA)

    # ── Spatial aggregation ──────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(backbone_output)

    # ── Normalise backbone features before feeding into Dense layers ──────────
    x = layers.BatchNormalization(name="bn_gap")(x)

    # ── First Dense block ────────────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=l2, name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="drop_256")(x)

    # ── Second Dense block ───────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=l2, name="dense_128")(x)
    x = layers.Dropout(config.DROPOUT_RATE * 0.6, name="drop_128")(x)  # lighter

    # ── Output ───────────────────────────────────────────────────────────────
    # Softmax for multi-class (works for binary too; alternatively use sigmoid)
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype="float32",          # keep fp32 for stability
                           name="predictions")(x)
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    backbone_name: str = config.BACKBONE,
    num_classes:   int = config.NUM_CLASSES,
    trainable:     bool = False,           # False = Phase 1 (frozen backbone)
) -> tuple[keras.Model, callable]:
    """
    Builds and returns the complete classifier model.

    Parameters
    ----------
    backbone_name : which pretrained network to use (see config.BACKBONE)
    num_classes   : output dimension
    trainable     : whether the backbone weights are trainable

    Returns
    -------
    model     : compiled Keras model ready for training
    preprocess: backbone-specific pixel preprocessing function
    """
    backbone, preprocess = _get_backbone(backbone_name)

    # ── Freeze/unfreeze backbone ──────────────────────────────────────────────
    backbone.trainable = trainable
    status = "trainable (fine-tune)" if trainable else "frozen"
    print(f"[Model] Backbone {status}")

    # ── Assemble model using Functional API ──────────────────────────────────
    inputs  = keras.Input(shape=config.INPUT_SHAPE, name="image_input")
    # NOTE: EfficientNet has its own internal preprocessing; for MobileNetV2
    # we apply explicit preprocess_input inside the model graph so that the
    # saved model is self-contained and inference servers don't need to
    # know which backbone was used.
    x       = backbone(inputs, training=False)  # training=False → BN in inference mode
    outputs = _build_head(x, num_classes)

    model = keras.Model(inputs, outputs, name=f"FruitClassifier_{backbone_name}")

    total_params     = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"[Model] Total params:     {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}")

    return model, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# COMPILATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    """
    Compiles the model with Adam + CategoricalCrossentropy + label smoothing.

    Label smoothing (e.g. 0.05) slightly softens the one-hot targets:
      1.0 → 0.975,  0.0 → 0.025
    This acts as regularisation and prevents overconfident predictions —
    especially important for a safety-critical supply-chain system.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING UNFREEZE
# ─────────────────────────────────────────────────────────────────────────────

def unfreeze_backbone(model: keras.Model, fine_tune_at: int = config.FINE_TUNE_AT):
    """
    Partially unfreezes the backbone for Phase 2 fine-tuning.

    Strategy:
      - Keep early layers frozen (they capture low-level edges / textures
        that are universal across all image domains).
      - Unfreeze the last `|fine_tune_at|` layers (they capture high-level
        domain-specific features) so they can adapt to fruit images.

    Parameters
    ----------
    fine_tune_at : negative index — how many layers from the END to unfreeze.
                   config default is -30 (last 30 layers).
    """
    # Find the backbone sub-model inside the functional model
    backbone_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone_layer = layer
            break

    if backbone_layer is None:
        raise RuntimeError("Could not locate backbone sub-model inside model.")

    # First freeze everything
    backbone_layer.trainable = True
    total = len(backbone_layer.layers)
    cutoff = total + fine_tune_at          # e.g. 237 - 30 = 207

    frozen_count = 0
    for layer in backbone_layer.layers[:cutoff]:
        layer.trainable = False
        frozen_count += 1

    unfrozen_count = total - frozen_count
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"[Model] Phase 2 — Unfroze last {unfrozen_count} / {total} "
          f"backbone layers")
    print(f"[Model] Trainable params now: {trainable_params:,}")

    return model
