"""
=============================================================================
predict.py — Production Inference Engine
=============================================================================
Designed for deployment inside the Smart Agriculture Supply Chain system.

Usage modes:
  1. Python import (FastAPI / Flask server, IoT edge device)
  2. Command-line  (batch processing of a folder or single image)

The FruitClassifier class is fully self-contained:
  • Loads model + class metadata once at startup
  • Processes a single image or a list of images
  • Returns structured prediction dicts — ready for JSON serialisation
  • Implements confidence thresholding for supply-chain safety
=============================================================================
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Union
import tensorflow as tf

import config


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FruitClassifier:
    """
    Production-ready inference wrapper for the fruit freshness model.

    Designed for integration into:
      • FastAPI / Flask REST endpoint
      • AWS Lambda / GCP Cloud Function
      • MQTT consumer for IoT sensor payloads
      • Farmer dashboard backend
      • Logistics tracking microservice
    """

    # Confidence below this → "uncertain" flag raised in the response.
    # Supply-chain safety: uncertain items are flagged for human review.
    CONFIDENCE_THRESHOLD = 0.85

    def __init__(
        self,
        model_path: str = None,
        metadata_path: str = None,
    ):
        """
        Loads the saved model and class metadata.

        Parameters
        ----------
        model_path    : path to .keras file (defaults to config.MODEL_DIR)
        metadata_path : path to class_metadata.json
        """
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR,
                                      "fruit_classifier_final.keras")
        if metadata_path is None:
            metadata_path = os.path.join(config.MODEL_DIR,
                                         "class_metadata.json")

        print(f"[Predictor] Loading model from: {model_path}")
        t0 = time.time()
        self.model = tf.keras.models.load_model(model_path)
        print(f"[Predictor] Model loaded in {time.time()-t0:.2f}s")

        with open(metadata_path) as f:
            meta = json.load(f)
        # meta = {"0": "Fresh", "1": "Rotten"}
        self.class_names = [meta[str(i)] for i in range(len(meta))]
        self.num_classes  = len(self.class_names)
        print(f"[Predictor] Classes: {self.class_names}")

        # Warm up the model (first inference is slow due to JIT compilation)
        dummy = np.zeros((1, *config.IMG_SIZE, config.IMG_CHANNELS), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        print("[Predictor] Ready ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # IMAGE LOADING HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_from_path(self, image_path: str) -> np.ndarray:
        """Loads an image file → normalised float32 numpy array (H, W, C)."""
        raw   = tf.io.read_file(image_path)
        image = tf.image.decode_image(raw, channels=config.IMG_CHANNELS,
                                       expand_animations=False)
        image = tf.image.resize(image, config.IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image.numpy()

    def _load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Loads raw image bytes (from HTTP upload, MQTT, etc.) → numpy array."""
        image = tf.image.decode_image(
            tf.constant(image_bytes),
            channels=config.IMG_CHANNELS,
            expand_animations=False,
        )
        image = tf.image.resize(image, config.IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image.numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE-IMAGE PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict_one(
        self,
        source: Union[str, bytes, np.ndarray],
    ) -> dict:
        """
        Classifies a single fruit image.

        Parameters
        ----------
        source : one of:
                 • str          → local file path
                 • bytes        → raw image bytes (from API upload)
                 • np.ndarray   → pre-loaded array of shape (H, W, 3), [0,1]

        Returns
        -------
        dict with keys:
          label        : "Fresh" or "Rotten"
          class_index  : 0 or 1
          confidence   : float [0, 1]
          probabilities: dict {class_name: probability}
          is_certain   : bool (confidence > CONFIDENCE_THRESHOLD)
          latency_ms   : inference latency
          supply_chain_action: recommended downstream action
        """
        t0 = time.perf_counter()

        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(source, str):
            img = self._load_from_path(source)
        elif isinstance(source, bytes):
            img = self._load_from_bytes(source)
        elif isinstance(source, np.ndarray):
            img = source
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # ── Inference ─────────────────────────────────────────────────────────
        # Add batch dimension: (H, W, C) → (1, H, W, C)
        batch  = np.expand_dims(img, axis=0)
        probs  = self.model.predict(batch, verbose=0)[0]   # shape: (num_classes,)
        idx    = int(np.argmax(probs))
        conf   = float(probs[idx])
        label  = self.class_names[idx]
        latency = (time.perf_counter() - t0) * 1000        # ms

        # ── Supply-chain action ──────────────────────────────────────────────
        action = self._get_supply_chain_action(label, conf)

        return {
            "label":               label,
            "class_index":         idx,
            "confidence":          round(conf, 4),
            "probabilities":       {
                name: round(float(p), 4)
                for name, p in zip(self.class_names, probs)
            },
            "is_certain":          conf >= self.CONFIDENCE_THRESHOLD,
            "latency_ms":          round(latency, 2),
            "supply_chain_action": action,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH PREDICTION
    # ─────────────────────────────────────────────────────────────────────────

    def predict_batch(
        self,
        sources: list[Union[str, bytes, np.ndarray]],
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Classifies a list of images efficiently in batches.

        Suitable for bulk processing of a truckload / conveyor belt scan.
        """
        results = []
        for i in range(0, len(sources), batch_size):
            chunk = sources[i : i + batch_size]
            for src in chunk:
                results.append(self.predict_one(src))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # FOLDER BATCH PROCESSING  (for farmer dashboard bulk upload)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_folder(self, folder_path: str) -> list[dict]:
        """
        Runs inference on all images in a folder.  Returns a list of result
        dicts, each augmented with the filename.
        """
        folder  = Path(folder_path)
        images  = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        print(f"[Predictor] Processing {len(images)} images from '{folder_path}'")

        results = []
        for img_path in images:
            result = self.predict_one(str(img_path))
            result["filename"] = img_path.name
            results.append(result)

        # Summary
        fresh_count  = sum(1 for r in results if r["label"] == "Fresh")
        rotten_count = len(results) - fresh_count
        uncertain    = sum(1 for r in results if not r["is_certain"])
        avg_latency  = np.mean([r["latency_ms"] for r in results])

        print(f"\n── Batch Summary ────────────────────────────────────────")
        print(f"  Total:     {len(results)}")
        print(f"  Fresh:     {fresh_count}  ({fresh_count/len(results):.1%})")
        print(f"  Rotten:    {rotten_count} ({rotten_count/len(results):.1%})")
        print(f"  Uncertain: {uncertain}    (flagged for human review)")
        print(f"  Avg latency: {avg_latency:.1f} ms/image")

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # SUPPLY CHAIN ACTION MAP
    # ─────────────────────────────────────────────────────────────────────────

    def _get_supply_chain_action(self, label: str, confidence: float) -> dict:
        """
        Maps prediction to a downstream supply-chain recommendation.

        This is the bridge between the ML model and the business logic.
        The downstream logistics tracking system consumes these fields.
        """
        if not (confidence >= self.CONFIDENCE_THRESHOLD):
            return {
                "action":     "HOLD_FOR_REVIEW",
                "storage":    "quarantine",
                "transport":  "hold",
                "priority":   "high",
                "reason":     f"Low confidence ({confidence:.1%}). Human review required.",
            }

        if label == "Fresh":
            return {
                "action":     "APPROVE",
                "storage":    "standard_cold_storage",
                "transport":  "schedule_normal",
                "priority":   "normal",
                "reason":     "Fresh fruit confirmed. Proceed with standard flow.",
            }
        else:  # Rotten
            return {
                "action":     "REJECT",
                "storage":    "segregate_for_disposal",
                "transport":  "do_not_ship",
                "priority":   "high",
                "reason":     "Rotten fruit detected. Remove from supply chain immediately.",
            }


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI INTEGRATION EXAMPLE  (copy into main.py of your API service)
# ─────────────────────────────────────────────────────────────────────────────

FASTAPI_EXAMPLE = '''
"""
Example FastAPI server — copy this into your API service directory.

Install:  pip install fastapi uvicorn python-multipart
Run:      uvicorn main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from predict import FruitClassifier
import uvicorn

classifier = None   # loaded at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = FruitClassifier()
    yield
    classifier = None

app = FastAPI(
    title="Fruit Freshness API",
    description="Smart Agriculture Supply Chain — Fruit Quality Inspection",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await file.read()
    result = classifier.predict_one(image_bytes)
    return result

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        image_bytes = await f.read()
        result = classifier.predict_one(image_bytes)
        result["filename"] = f.filename
        results.append(result)
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Fruit Freshness Classifier — Inference CLI"
    )
    parser.add_argument("--image",  type=str, help="Path to a single image file")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    clf = FruitClassifier()

    if args.image:
        result = clf.predict_one(args.image)
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif args.folder:
        results = clf.predict_folder(args.folder)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved → {args.output}")
        else:
            print(json.dumps(results[:3], indent=2))  # preview first 3

    else:
        print("Please specify --image or --folder. Use --help for usage.")
