"""
evaluate.py
Evaluate trained classifier.
"""

import sys
import os
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.training.normalization import FEATURE_NAMES, load_training_stats, normalize_dataframe

DATASET = Path("data/processed/training_dataset.csv")
MODEL = Path("data/models/best_model.keras")
STATS = Path("data/models/training_stats.npy")
ENCODER = Path("data/models/label_encoder.pkl")
REPORT_DIR = Path("reports")

df = pd.read_csv(DATASET)

X = df.loc[:, list(FEATURE_NAMES)]
y = df["label"]

import joblib

encoder = joblib.load(ENCODER)
y = encoder.transform(y)

stats = load_training_stats(STATS)
X = normalize_dataframe(X, stats)

model = tf.keras.models.load_model(MODEL)

pred = model.predict(X, verbose=0)
pred_labels = pred.argmax(axis=1)

report = classification_report(
    y,
    pred_labels,
    target_names=encoder.classes_
)

print(report)

(REPORT_DIR/"classification_report.txt").write_text(report)

cm = confusion_matrix(y, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot()
plt.tight_layout()
plt.savefig(REPORT_DIR/"confusion_matrix.png")
plt.close()
