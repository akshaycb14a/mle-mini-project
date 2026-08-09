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
import tensorflow as tf

from src.training.normalization import load_training_stats, normalize_dataframe
from src.training.utils import load_dataset_frame, make_holdout_split

DATASET = Path("data/processed/training_dataset.csv")
MODEL = Path("data/models/best_model.keras")
STATS = Path("data/models/training_stats.npy")
ENCODER = Path("data/models/label_encoder.pkl")
REPORT_DIR = Path("reports")

df = load_dataset_frame(DATASET)
y = df["label"]

import joblib

encoder = joblib.load(ENCODER)
y = encoder.transform(y)
_, X_val, _, y_val = make_holdout_split(df, y)

stats = load_training_stats(STATS)
X_val = normalize_dataframe(X_val, stats)

model = tf.keras.models.load_model(MODEL)

pred = model.predict(X_val, verbose=0)
pred_labels = pred.argmax(axis=1)

report = classification_report(
    y_val,
    pred_labels,
    target_names=encoder.classes_
)

print(report)

(REPORT_DIR/"classification_report.txt").write_text(report)

cm = confusion_matrix(y_val, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot()
plt.tight_layout()
plt.savefig(REPORT_DIR/"confusion_matrix.png")
plt.close()
