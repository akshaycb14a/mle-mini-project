import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import joblib

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.normalization import (
    FEATURE_NAMES,
    load_training_stats,
    normalize_dataframe,
)


DATASET = Path("data/processed/training_dataset.csv")
MODEL = Path("data/models/best_model.keras")
STATS = Path("data/models/training_stats.npy")
ENCODER = Path("data/models/label_encoder.pkl")
REPORT_DIR = Path("reports")
RESULT_JSON = REPORT_DIR / "normalization_experiment.json"
RESULT_CSV = REPORT_DIR / "normalization_experiment.csv"


def evaluate_with_stats(model, frame, labels, stats):
    X = normalize_dataframe(frame, stats)
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    return accuracy_score(labels, preds)


def shift_stats_by_sigma(stats, sigma=3.0):
    shifted = {
        "feature_names": list(stats["feature_names"]),
        "mean": np.asarray(stats["mean"], dtype=np.float32)
        + sigma * np.asarray(stats["std"], dtype=np.float32),
        "std": np.asarray(stats["std"], dtype=np.float32),
    }
    return shifted


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    feature_cols = list(FEATURE_NAMES)
    X = df.loc[:, feature_cols]
    y = df["label"]

    model = tf.keras.models.load_model(MODEL)
    encoder = joblib.load(ENCODER)
    stats = load_training_stats(STATS)

    encoded_y = encoder.transform(y)

    correct_accuracy = evaluate_with_stats(model, X, encoded_y, stats)
    shifted_accuracy = evaluate_with_stats(
        model,
        X,
        encoded_y,
        shift_stats_by_sigma(stats, 3.0),
    )

    print(f"Accuracy with correct stats: {correct_accuracy:.4f}")
    print(f"Accuracy with stats shifted by +3 std: {shifted_accuracy:.4f}")

    payload = {
        "dataset": str(DATASET),
        "model": str(MODEL),
        "stats": str(STATS),
        "correct_stats_accuracy": float(correct_accuracy),
        "shifted_stats_accuracy": float(shifted_accuracy),
        "shift_sigma": 3.0,
    }

    RESULT_JSON.write_text(json.dumps(payload, indent=2))
    pd.DataFrame([payload]).to_csv(RESULT_CSV, index=False)
    print(f"Saved experiment results to {RESULT_JSON}")
    print(f"Saved experiment results to {RESULT_CSV}")


if __name__ == "__main__":
    main()
