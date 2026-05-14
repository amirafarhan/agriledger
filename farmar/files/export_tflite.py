"""
=============================================================================
export_tflite.py — Convert to TensorFlow Lite for Edge / Mobile Deployment
=============================================================================
TFLite is the standard format for:
  • Android / iOS apps (farmer mobile app)
  • Raspberry Pi / Jetson Nano at conveyor belt (IoT edge)
  • On-device inference without cloud dependency

Quantisation options:
  ─────────────────────────────────────────────────────────────────────────
  Mode             │ Size     │ Accuracy loss │ Speed    │ Use when
  ─────────────────┼──────────┼───────────────┼──────────┼──────────────
  No quantisation  │ ~20 MB   │ none          │ baseline │ Cloud serving
  Dynamic range    │ ~5 MB    │ < 0.5%        │ 2-3×     │ Default mobile
  Full int8        │ ~5 MB    │ < 1%          │ 3-4×     │ Microcontrollers
  Float16          │ ~10 MB   │ < 0.1%        │ 1.5-2×   │ Mobile GPU
  ─────────────────────────────────────────────────────────────────────────

Run:
    python export_tflite.py --mode dynamic_range
=============================================================================
"""

import os
import argparse
import numpy as np
import tensorflow as tf

import config
import data_pipeline as dp


def representative_dataset_gen(n_samples: int = 200):
    """
    Generator that yields sample images for int8 full-integer quantisation.
    TFLite uses these to calibrate the quantisation scale factors.

    Must yield float32 batches of shape (1, H, W, C).
    """
    test_paths, test_labels = dp.collect_files_and_labels(config.TEST_DIR)
    # Randomly sample n_samples paths
    indices = np.random.choice(len(test_paths), size=min(n_samples, len(test_paths)),
                                replace=False)
    for idx in indices:
        raw   = tf.io.read_file(test_paths[idx])
        image = tf.image.decode_image(raw, channels=config.IMG_CHANNELS,
                                       expand_animations=False)
        image = tf.image.resize(image, config.IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        # Add batch dimension
        yield [image.numpy()[np.newaxis, ...].astype(np.float32)]


def convert_to_tflite(mode: str = "dynamic_range") -> str:
    """
    Converts the SavedModel to TFLite with the specified quantisation mode.

    Parameters
    ----------
    mode : "none" | "dynamic_range" | "float16" | "int8"

    Returns
    -------
    Path to the saved .tflite file
    """
    saved_model_dir = os.path.join(config.MODEL_DIR, "fruit_classifier_savedmodel")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    print(f"[Export] Loading SavedModel from: {saved_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # ── Quantisation mode ─────────────────────────────────────────────────────
    if mode == "none":
        filename = "fruit_classifier.tflite"
        print("[Export] No quantisation (float32)")

    elif mode == "dynamic_range":
        # DEFAULT RECOMMENDATION: 4× smaller, 2-3× faster, < 0.5% accuracy loss
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        filename = "fruit_classifier_dynamic_quant.tflite"
        print("[Export] Dynamic range quantisation (recommended for mobile)")

    elif mode == "float16":
        converter.optimizations          = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        filename = "fruit_classifier_float16.tflite"
        print("[Export] Float16 quantisation (good for GPU-accelerated mobile)")

    elif mode == "int8":
        # Full integer quantisation — smallest model, fastest on microcontrollers
        converter.optimizations              = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset     = representative_dataset_gen
        converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type       = tf.int8
        converter.inference_output_type      = tf.int8
        filename = "fruit_classifier_int8.tflite"
        print("[Export] Full int8 quantisation (for Raspberry Pi / microcontrollers)")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: none, dynamic_range, float16, int8")

    # ── Convert ───────────────────────────────────────────────────────────────
    print("[Export] Converting …")
    tflite_model = converter.convert()

    out_path = os.path.join(config.MODEL_DIR, filename)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Export] Saved → {out_path}  ({size_mb:.1f} MB)")
    return out_path


def benchmark_tflite(tflite_path: str, n_runs: int = 100):
    """
    Benchmarks the TFLite model inference speed using TFLite interpreter.
    Reports mean ± std latency in milliseconds.
    """
    import time
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape    = input_details[0]["shape"]       # (1, 224, 224, 3)
    dtype          = input_details[0]["dtype"]

    # Random test image
    dummy = np.random.rand(*input_shape).astype(dtype)
    if dtype == np.int8:
        dummy = (dummy * 127).astype(np.int8)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\n[Benchmark] Latency over {n_runs} runs:")
    print(f"  Mean:  {latencies.mean():.2f} ms")
    print(f"  Std:   {latencies.std():.2f} ms")
    print(f"  P95:   {np.percentile(latencies, 95):.2f} ms")
    print(f"  Min:   {latencies.min():.2f} ms")
    return latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFLite Export & Benchmark")
    parser.add_argument(
        "--mode", type=str, default="dynamic_range",
        choices=["none", "dynamic_range", "float16", "int8"],
        help="Quantisation mode",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark after conversion",
    )
    args = parser.parse_args()

    tflite_path = convert_to_tflite(args.mode)
    if args.benchmark:
        benchmark_tflite(tflite_path)
