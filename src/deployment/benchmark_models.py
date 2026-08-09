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
    laptop_tdp_watts = float(os.getenv("LAPTOP_TDP_W", "28.0"))
    print(f"Using laptop TDP estimate: {laptop_tdp_watts:.2f} W")

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
            laptop_tdp_watts=laptop_tdp_watts,
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
            "estimated_power_watts": metrics["estimated_power_watts"],
            "cpu_utilization_fraction": metrics["cpu_utilization_fraction"],
            "laptop_tdp_watts": metrics["laptop_tdp_watts"],
            "energy_label": "estimated_from_cpu_utilization_and_laptop_tdp",
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
