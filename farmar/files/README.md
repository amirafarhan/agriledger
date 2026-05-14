# 🍎 Fruit Freshness Classifier
### Smart Agriculture Supply Chain — Fruit Quality Inspection

> **Target**: ≥ 98% validation accuracy  
> **Model**: EfficientNetB0 with two-phase transfer learning  
> **Deployment**: REST API · TFLite Mobile · IoT Edge

---

## 📁 Project Structure

```
fruit_classifier/
├── config.py            ← ALL hyperparameters live here
├── data_pipeline.py     ← Dataset loading, augmentation, tf.data
├── model.py             ← Backbone + classification head builder
├── callbacks.py         ← EarlyStopping, ReduceLR, Checkpoint, TensorBoard
├── train.py             ← Two-phase training pipeline  ← START HERE
├── evaluate.py          ← Confusion matrix, ROC curve, classification report
├── predict.py           ← Production inference engine + FastAPI example
├── export_tflite.py     ← TFLite conversion for mobile/edge deployment
├── requirements.txt
└── outputs/             ← auto-created during training
    ├── models/
    │   ├── fruit_classifier_final.keras
    │   ├── fruit_classifier_savedmodel/
    │   ├── fruit_classifier_dynamic_quant.tflite
    │   ├── class_metadata.json
    │   └── training_config.json
    ├── logs/
    │   ├── phase1/  ← TensorBoard logs
    │   ├── phase2/
    │   ├── training_history.json
    │   └── classification_report.txt
    └── plots/
        ├── training_curves.png
        ├── confusion_matrix.png
        └── roc_curve.png
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
```
Unzip so the structure is:
```
dataset/
  train/
    freshapples/   freshbananas/   freshoranges/
    rottenapples/  rottenbananas/  rottenoranges/
  test/
    (same sub-folders)
```

### 3. Train
```bash
python train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

### 5. Predict a single image
```bash
python predict.py --image path/to/apple.jpg
```

### 6. Export to TFLite
```bash
python export_tflite.py --mode dynamic_range --benchmark
```

---

## 🏗️ Architecture

```
Input (224×224×3)
       │
 EfficientNetB0 backbone (ImageNet pretrained)
   Phase 1: All layers FROZEN
   Phase 2: Last 30 layers unfrozen
       │
 GlobalAveragePooling2D
       │
 BatchNormalization
       │
 Dense(256, ReLU) + L2
       │
 BatchNormalization + Dropout(0.40)
       │
 Dense(128, ReLU) + L2
       │
 Dropout(0.24)
       │
 Dense(2, Softmax)   ← Fresh / Rotten
```

---

## ⚙️ Two-Phase Training Strategy

| | Phase 1 | Phase 2 |
|---|---|---|
| **Goal** | Train classification head | Fine-tune domain features |
| **Backbone** | Frozen | Last 30 layers unfrozen |
| **LR** | 1e-3 | 1e-4 (10× smaller) |
| **Epochs** | 15 | 30 |
| **Batch size** | 32 | 16 |
| **Why** | Protect ImageNet weights | Adapt to fruit images |

---

## 🎛️ Key Configuration (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `BACKBONE` | `EfficientNetB0` | Switch to B3 for +1% accuracy |
| `BINARY_MODE` | `True` | 2 classes vs 6 fine-grained |
| `FINE_TUNE_AT` | `-30` | Layers to unfreeze in Phase 2 |
| `DROPOUT_RATE` | `0.40` | Higher = more regularisation |
| `LABEL_SMOOTHING` | `0.05` | Prevents overconfident outputs |
| `PHASE2_LR` | `1e-4` | Must be ≤ 1/10 of Phase 1 LR |

---

## 📊 Expected Results

| Metric | Binary (Fresh/Rotten) | Fine-grained (6 classes) |
|---|---|---|
| Val Accuracy | **≥ 98%** | ≥ 95% |
| AUC | ≥ 0.999 | — |
| F1-score | ≥ 0.98 | ≥ 0.95 |
| Inference latency | ~15 ms (GPU) | ~15 ms (GPU) |
| TFLite latency | ~50 ms (mobile CPU) | ~50 ms |

---

## 🔧 If Accuracy < 98% — Troubleshooting Guide

### Gap > 5% (accuracy < 93%)
1. **Check data quality** — open 20 images from each folder, verify labels
2. **Check folder structure** — ensure train/ and test/ have 6 subfolders each
3. **GPU available?** — CPU training can produce worse results due to shorter epochs

### Gap 2–5% (93–96%)
1. **Stronger backbone** → `config.BACKBONE = "EfficientNetB3"`
2. **Longer fine-tuning** → `config.FINE_TUNE_AT = -50`
3. **More epochs** → `config.PHASE2_EPOCHS = 50`
4. **Reduce dropout** → `config.DROPOUT_RATE = 0.25`

### Gap < 2% (96–97.9%)
1. **Test-Time Augmentation (TTA)** — average predictions over 5 augmented views
2. **Ensemble** — train EfficientNetB0 + MobileNetV2, average softmax outputs
3. **Cosine LR schedule** — replace ReduceLROnPlateau with CosineDecayRestarts
4. **MixUp augmentation** — blends two images and their labels

---

## 🚀 Supply Chain Integration

### Prediction output format
```json
{
  "label": "Rotten",
  "class_index": 1,
  "confidence": 0.9987,
  "probabilities": { "Fresh": 0.0013, "Rotten": 0.9987 },
  "is_certain": true,
  "latency_ms": 14.3,
  "supply_chain_action": {
    "action": "REJECT",
    "storage": "segregate_for_disposal",
    "transport": "do_not_ship",
    "priority": "high",
    "reason": "Rotten fruit detected. Remove from supply chain immediately."
  }
}
```

### REST API
```bash
# Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# Predict via HTTP
curl -X POST http://localhost:8000/predict \
     -F "file=@apple.jpg"
```

### IoT / MQTT Integration
```python
from predict import FruitClassifier
import paho.mqtt.client as mqtt

clf = FruitClassifier()

def on_message(client, userdata, msg):
    result = clf.predict_one(msg.payload)   # msg.payload = raw image bytes
    client.publish("supply_chain/fruit_quality", json.dumps(result))
```

### TFLite on Raspberry Pi
```python
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter("fruit_classifier_dynamic_quant.tflite")
interpreter.allocate_tensors()
# Feed numpy image array, get probabilities
```

---

## 📈 TensorBoard Monitoring
```bash
tensorboard --logdir outputs/logs
# Open: http://localhost:6006
```

---

## 🧪 Reproducibility

All random seeds are set in `train.py:set_seeds()`:
- Python `random`
- NumPy
- TensorFlow
- `PYTHONHASHSEED`

Training config is saved to `outputs/models/training_config.json` for audit.

---

## 📦 Model Files

| File | Size | Use |
|---|---|---|
| `fruit_classifier_final.keras` | ~20 MB | Python serving, resuming training |
| `fruit_classifier_savedmodel/` | ~20 MB | TF Serving, Cloud Functions |
| `fruit_classifier_dynamic_quant.tflite` | ~5 MB | Mobile apps, Raspberry Pi |
| `fruit_classifier_int8.tflite` | ~5 MB | Microcontrollers, Coral Edge TPU |

---

*Built for the Smart Agriculture Supply Chain system.*  
*Model: EfficientNetB0 · Framework: TensorFlow/Keras · Python 3.10+*
