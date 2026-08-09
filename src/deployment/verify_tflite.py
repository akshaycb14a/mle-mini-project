"""
verify_tflite.py
Verify TensorFlow Lite model inference.
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.normalization import FEATURE_NAMES, load_training_stats, normalize_dataframe

df = pd.read_csv("data/processed/training_dataset.csv")
X = df.loc[:, list(FEATURE_NAMES)]

stats = load_training_stats("data/models/training_stats.npy")
X = normalize_dataframe(X, stats).astype(np.float32)

interpreter = tf.lite.Interpreter(
    model_path="data/tflite/edge_classifier_fp32.tflite"
)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

sample = X[:5]

for i,row in enumerate(sample):
    interpreter.set_tensor(inp["index"], row.reshape(1,-1))
    interpreter.invoke()
    pred = interpreter.get_tensor(out["index"])
    print(f"Sample {i}: class={np.argmax(pred)}, confidence={pred.max():.4f}")
