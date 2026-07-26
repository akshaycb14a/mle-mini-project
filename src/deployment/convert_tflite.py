"""
convert_tflite.py
Convert trained Keras model to TensorFlow Lite.
"""
from pathlib import Path
import tensorflow as tf

MODEL_DIR = Path("data/models")
TFLITE_DIR = Path("data/tflite")
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

keras_model = tf.keras.models.load_model(MODEL_DIR/"edge_classifier.keras")

# FP32
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
(TFLITE_DIR/"edge_classifier_fp32.tflite").write_bytes(tflite_model)

# Dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
(TFLITE_DIR/"edge_classifier_int8.tflite").write_bytes(tflite_model)

print("TensorFlow Lite models generated successfully.")
