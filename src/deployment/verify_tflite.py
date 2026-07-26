"""
verify_tflite.py
Verify TensorFlow Lite model inference.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

df = pd.read_csv("data/processed/training_dataset.csv")
X = df.drop(columns=["label"])

scaler = joblib.load("data/models/scaler.pkl")
X = scaler.transform(X).astype(np.float32)

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
