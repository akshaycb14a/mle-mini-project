"""
predictor.py
Reusable TensorFlow Lite predictor.
"""
import numpy as np
import tensorflow as tf
import joblib

class EdgePredictor:
    def __init__(
        self,
        model_path="data/tflite/edge_classifier_fp32.tflite",
        scaler_path="data/models/scaler.pkl",
        encoder_path="data/models/label_encoder.pkl",
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

    def predict(self, features: dict):
        vector = np.array([[
            features["temp_mean"],
            features["temp_std"],
            features["temp_rate"],
            features["vibration_rms"],
            features["vibration_peak"],
            features["vibration_kurtosis"],
        ]], dtype=np.float32)

        vector = self.scaler.transform(vector).astype(np.float32)

        self.interpreter.set_tensor(self.input["index"], vector)
        self.interpreter.invoke()

        probs = self.interpreter.get_tensor(self.output["index"])[0]
        idx = int(np.argmax(probs))

        return {
            "prediction": self.encoder.inverse_transform([idx])[0],
            "confidence": float(probs[idx]),
            "probabilities": probs.tolist(),
        }
