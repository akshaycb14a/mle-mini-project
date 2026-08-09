"""
predictor.py
Reusable TensorFlow Lite predictor.
"""
import os

import numpy as np
import tensorflow as tf

from src.training.normalization import load_training_stats, normalize_matrix

class EdgePredictor:
    def __init__(
        self,
        model_path=None,
        stats_path="data/models/training_stats.npy",
        encoder_path="data/models/label_encoder.pkl",
    ):
        model_path = model_path or os.getenv(
            "MODEL_PATH",
            "data/tflite/model_fp32.tflite",
        )
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

        import joblib

        self.stats = load_training_stats(stats_path)
        self.encoder = joblib.load(encoder_path)

    def _quantize_input(self, vector):
        dtype = self.input["dtype"]
        if dtype == np.float32:
            return vector.astype(np.float32)

        scale, zero_point = self.input["quantization"]
        if scale == 0:
            raise ValueError("Invalid quantization scale for model input")

        quantized = np.round(vector / scale + zero_point)
        if dtype == np.int8:
            quantized = np.clip(quantized, -128, 127)
        elif dtype == np.uint8:
            quantized = np.clip(quantized, 0, 255)
        return quantized.astype(dtype)

    def _dequantize_output(self, raw_output):
        dtype = self.output["dtype"]
        if dtype == np.float32:
            return raw_output.astype(np.float32)

        scale, zero_point = self.output["quantization"]
        if scale == 0:
            return raw_output.astype(np.float32)
        return scale * (raw_output.astype(np.float32) - zero_point)

    def predict(self, features: dict):
        vector = np.array([[
            features["temp_mean"],
            features["temp_std"],
            features["temp_rate"],
            features["vibration_rms"],
            features["vibration_peak"],
            features["vibration_kurtosis"],
        ]], dtype=np.float32)

        vector = normalize_matrix(vector, self.stats).astype(np.float32)
        input_tensor = self._quantize_input(vector)

        self.interpreter.set_tensor(self.input["index"], input_tensor)
        self.interpreter.invoke()

        raw_output = self.interpreter.get_tensor(self.output["index"])
        probs = self._dequantize_output(raw_output)[0]
        idx = int(np.argmax(probs))

        return {
            "prediction": self.encoder.inverse_transform([idx])[0],
            "confidence": float(probs[idx]),
            "probabilities": probs.tolist(),
        }
