"""
convert_tflite.py
Convert trained Keras model to TensorFlow Lite variants.
"""
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.deployment.build_deployment_models import main

if __name__ == "__main__":
    main()
