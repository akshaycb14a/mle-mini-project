"""
train_model.py
Training pipeline for the Edge AI cold-chain monitor.
"""

import sys
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.training.normalization import (
    FEATURE_NAMES,
    load_training_stats,
    normalize_dataframe,
    make_legacy_scaler,
)

DATASET = Path("data/processed/training_dataset.csv")
MODEL_DIR = Path("data/models")
REPORT_DIR = Path("reports")
STATS_PATH = MODEL_DIR / "training_stats.npy"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET)

X = df.loc[:, list(FEATURE_NAMES)]
y = df["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

joblib.dump(encoder, MODEL_DIR/"label_encoder.pkl")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

if STATS_PATH.exists():
    stats = load_training_stats(STATS_PATH)
else:
    raise FileNotFoundError(
        f"Missing {STATS_PATH}. Run src/training/generate_training_stats.py first."
    )

X_train = normalize_dataframe(X_train, stats)
X_val = normalize_dataframe(X_val, stats)
X_test = normalize_dataframe(X_test, stats)

legacy_scaler = make_legacy_scaler(stats)
joblib.dump(legacy_scaler, SCALER_PATH)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_DIR/"best_model.keras", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

model.save(MODEL_DIR/"edge_classifier.keras")

plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(REPORT_DIR/"training_accuracy.png")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(REPORT_DIR/"training_loss.png")
plt.close()

print("Training complete.")
