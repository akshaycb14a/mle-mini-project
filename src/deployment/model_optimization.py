import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import accuracy_score, recall_score

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from src.training.normalization import (
    FEATURE_NAMES,
    load_training_stats,
    normalize_dataframe,
)
from src.training.utils import load_dataset_frame, make_holdout_split


DATASET = Path("data/processed/training_dataset.csv")
MODEL_PATH = Path("data/models/edge_classifier.keras")
STATS_PATH = Path("data/models/training_stats.npy")
ENCODER_PATH = Path("data/models/label_encoder.pkl")
TFLITE_DIR = Path("data/tflite")
MODEL_DIR = Path("data/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_DIR.mkdir(parents=True, exist_ok=True)


def load_training_artifacts():
    df = load_dataset_frame(DATASET)
    y = df["label"]

    import joblib

    encoder = joblib.load(ENCODER_PATH)
    y_encoded = encoder.transform(y)

    X_train, X_val, y_train, y_val = make_holdout_split(df, y_encoded)

    stats = load_training_stats(STATS_PATH)
    X_train_norm = normalize_dataframe(X_train, stats).astype(np.float32)
    X_val_norm = normalize_dataframe(X_val, stats).astype(np.float32)
    X_benchmark_norm = normalize_dataframe(X_val, stats).astype(np.float32)

    return {
        "encoder": encoder,
        "stats": stats,
        "X_train": X_train_norm,
        "y_train": np.asarray(y_train, dtype=np.int64),
        "X_val": X_val_norm,
        "y_val": np.asarray(y_val, dtype=np.int64),
        "X_benchmark": X_benchmark_norm,
        "y_benchmark": np.asarray(y_val, dtype=np.int64),
        "raw_frame": df,
    }


def build_base_architecture():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_NAMES),)),
        tf.keras.layers.Dense(32, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(16, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(3, activation="softmax", name="dense_output"),
    ])


def load_base_model():
    try:
        import keras

        source_model = keras.models.load_model(MODEL_PATH)
    except Exception:
        source_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    model = build_base_architecture()
    model.set_weights(source_model.get_weights())
    return model


def representative_dataset(samples, limit=200):
    shuffled = np.asarray(samples, dtype=np.float32).copy()
    np.random.default_rng(42).shuffle(shuffled)
    sample_count = min(limit, len(shuffled))
    if sample_count < 200:
        raise ValueError("Representative dataset requires at least 200 samples")

    for row in shuffled[:sample_count]:
        yield [np.asarray(row, dtype=np.float32).reshape(1, -1)]


def convert_fp32_model(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return Path(output_path)


def convert_full_int8_model(model, output_path, representative_samples):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        representative_samples,
        limit=200,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return Path(output_path)


def build_pruned_model(base_model, end_step, target_sparsity=0.35):
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=end_step,
        ),
        "block_size": (1, 4),
        "block_pooling_type": "AVG",
    }
    pruned_dense_1 = tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.layers.Dense(32, activation="relu", name="dense_1"),
        **pruning_params,
    )
    pruned_dense_2 = tfmot.sparsity.keras.prune_low_magnitude(
        tf.keras.layers.Dense(16, activation="relu", name="dense_2"),
        **pruning_params,
    )
    output_layer = tf.keras.layers.Dense(3, activation="softmax", name="dense_output")
    pruned_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_NAMES),)),
        pruned_dense_1,
        pruned_dense_2,
        output_layer,
    ])
    pruned_model(np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32))
    pruned_dense_1.layer.set_weights(base_model.get_layer("dense_1").get_weights())
    pruned_dense_2.layer.set_weights(base_model.get_layer("dense_2").get_weights())
    output_layer.set_weights(base_model.get_layer("dense_output").get_weights())
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return pruned_model


def fine_tune_pruned_model(pruned_model, X_train, y_train, X_val, y_val):
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]
    pruned_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=6,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )
    return pruned_model


def export_pruned_int8_model(base_model, output_path, representative_samples):
    steps_per_epoch = int(np.ceil(len(representative_samples["X_train"]) / 16.0))
    end_step = max(steps_per_epoch * 6, 1)
    pruned_model = build_pruned_model(base_model, end_step=end_step)
    pruned_model = fine_tune_pruned_model(
        pruned_model,
        representative_samples["X_train"],
        representative_samples["y_train"],
        representative_samples["X_val"],
        representative_samples["y_val"],
    )
    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    return convert_full_int8_model(
        stripped,
        output_path,
        representative_samples["X_train"],
    )


def create_compatibility_copy(source_path, target_path):
    data = Path(source_path).read_bytes()
    Path(target_path).write_bytes(data)


def get_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def quantize_input(sample, input_details):
    sample = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return sample.astype(np.float32)

    scale, zero_point = input_details["quantization"]
    if scale == 0:
        raise ValueError("Invalid quantization scale for int8 input")

    q = np.round(sample / scale + zero_point)
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return q.astype(dtype)


def dequantize_output(raw_output, output_details):
    dtype = output_details["dtype"]
    if dtype == np.float32:
        return raw_output

    scale, zero_point = output_details["quantization"]
    if scale == 0:
        return raw_output.astype(np.float32)
    return scale * (raw_output.astype(np.float32) - zero_point)


def run_tflite_predictions(model_path, samples):
    interpreter = get_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    predictions = []

    for sample in samples:
        q_sample = quantize_input(sample, input_details)
        interpreter.set_tensor(input_details["index"], q_sample)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])
        probs = dequantize_output(raw, output_details)[0]
        predictions.append(int(np.argmax(probs)))

    return np.asarray(predictions, dtype=np.int64)


def benchmark_latency_ms(model_path, samples, warmup=10, runs=200):
    interpreter = get_interpreter(model_path)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    total_needed = warmup + runs
    if len(samples) == 0:
        raise ValueError("Benchmark validation set cannot be empty")

    durations = []
    for idx in range(total_needed):
        sample = samples[idx % len(samples)]
        q_sample = quantize_input(sample, input_details)
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_details["index"], q_sample)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])
        _ = dequantize_output(raw, output_details)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if idx >= warmup:
            durations.append(elapsed_ms)

    durations = np.asarray(durations, dtype=np.float32)
    return float(durations.mean()), float(np.percentile(durations, 95)), durations


def estimate_power_from_cpu(cpu_utilization_fraction, laptop_tdp_watts):
    return float(laptop_tdp_watts) * float(cpu_utilization_fraction)


def compute_metrics_for_model(model_path, frame, labels, encoder, laptop_tdp_watts=28.0):
    samples = frame.to_numpy(dtype=np.float32)
    predictions = run_tflite_predictions(model_path, samples)
    accuracy = accuracy_score(labels, predictions) * 100.0
    critical_label = int(np.where(encoder.classes_ == "critical")[0][0])
    critical_recall = recall_score(
        labels,
        predictions,
        labels=[critical_label],
        average="macro",
        zero_division=0,
    ) * 100.0
    process = psutil.Process()
    cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()
    mean_latency, p95_latency, durations = benchmark_latency_ms(model_path, samples)
    wall_elapsed = max(time.perf_counter() - wall_start, 1e-9)
    cpu_end = process.cpu_times()
    cpu_time_used = (
        (cpu_end.user + cpu_end.system) -
        (cpu_start.user + cpu_start.system)
    )
    cpu_utilization_fraction = cpu_time_used / (wall_elapsed * cpu_count)
    cpu_utilization_fraction = float(np.clip(cpu_utilization_fraction, 0.0, 1.0))
    file_size_kb = Path(model_path).stat().st_size / 1024.0
    estimated_power_watts = estimate_power_from_cpu(
        cpu_utilization_fraction,
        laptop_tdp_watts,
    )
    mean_energy_mj = estimated_power_watts * mean_latency
    return {
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "model_size_kb": file_size_kb,
        "accuracy_percent": accuracy,
        "estimated_energy_mj": mean_energy_mj,
        "critical_recall_percent": critical_recall,
        "estimated_power_watts": estimated_power_watts,
        "cpu_utilization_fraction": cpu_utilization_fraction,
        "laptop_tdp_watts": float(laptop_tdp_watts),
        "latency_samples_ms": durations,
    }
