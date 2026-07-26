"""
evaluate.py
Evaluate trained classifier.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET = Path("data/processed/training_dataset.csv")
MODEL = Path("data/models/best_model.keras")
SCALER = Path("data/models/scaler.pkl")
ENCODER = Path("data/models/label_encoder.pkl")
REPORT_DIR = Path("reports")

df = pd.read_csv(DATASET)

X = df.drop(columns=["label"])
y = df["label"]

encoder = joblib.load(ENCODER)
y = encoder.transform(y)

scaler = joblib.load(SCALER)
X = scaler.transform(X)

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
