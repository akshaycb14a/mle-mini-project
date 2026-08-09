import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.deployment.model_optimization import (
    TFLITE_DIR,
    compute_metrics_for_model,
    load_training_artifacts,
    measure_power_watts,
)


RESULTS_CSV = Path("reports/benchmark_results.csv")
PARETO_PNG = Path("reports/pareto_chart.png")


def main():
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    artifacts = load_training_artifacts()
    validation_frame = pd.DataFrame(
        artifacts["X_benchmark"],
        columns=artifacts["stats"]["feature_names"],
    )
    validation_labels = artifacts["y_benchmark"]
    encoder = artifacts["encoder"]
    measured_power_watts, power_source = measure_power_watts()
    print(
        f"Using measured power: {measured_power_watts:.4f} W "
        f"({power_source})"
    )

    models = [
        ("M1_FP32", TFLITE_DIR / "model_fp32.tflite"),
        ("M2_INT8", TFLITE_DIR / "model_int8.tflite"),
        ("M3_Pruned_INT8", TFLITE_DIR / "model_pruned_int8.tflite"),
    ]

    rows = []
    for variant, model_path in models:
        print(f"Benchmarking {variant}: {model_path}")
        metrics = compute_metrics_for_model(
            model_path,
            validation_frame,
            validation_labels,
            encoder,
            power_watts=measured_power_watts,
        )

        row = {
            "variant": variant,
            "model_file": str(model_path),
            "mean_inference_latency_ms": metrics["mean_latency_ms"],
            "p95_inference_latency_ms": metrics["p95_latency_ms"],
            "model_size_kb": metrics["model_size_kb"],
            "classification_accuracy_percent": metrics["accuracy_percent"],
            "estimated_energy_per_inference_mj": metrics["estimated_energy_mj"],
            "critical_recall_percent": metrics["critical_recall_percent"],
            "estimated_power_watts": measured_power_watts,
            "power_source": power_source,
        }
        rows.append(row)
        print(
            f"{variant}: accuracy={row['classification_accuracy_percent']:.2f}% "
            f"latency={row['mean_inference_latency_ms']:.4f} ms "
            f"critical_recall={row['critical_recall_percent']:.2f}%"
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["mean_inference_latency_ms"],
        df["classification_accuracy_percent"],
        s=120,
    )
    for _, row in df.iterrows():
        plt.annotate(
            row["variant"],
            (row["mean_inference_latency_ms"], row["classification_accuracy_percent"]),
            textcoords="offset points",
            xytext=(8, 6),
        )
    plt.xlabel("Mean Inference Latency (ms)")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("Deployment Pareto Chart: Accuracy vs Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PARETO_PNG, dpi=180)
    plt.close()

    print(f"Saved benchmark results to {RESULTS_CSV}")
    print(f"Saved Pareto chart to {PARETO_PNG}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
