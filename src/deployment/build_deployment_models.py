import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from src.deployment.model_optimization import (
    MODEL_PATH,
    TFLITE_DIR,
    create_compatibility_copy,
    convert_fp32_model,
    convert_full_int8_model,
    export_pruned_int8_model,
    load_base_model,
    load_training_artifacts,
)


def main():
    artifacts = load_training_artifacts()
    base_model = load_base_model()

    fp32_path = TFLITE_DIR / "model_fp32.tflite"
    int8_path = TFLITE_DIR / "model_int8.tflite"
    pruned_int8_path = TFLITE_DIR / "model_pruned_int8.tflite"

    print(f"Exporting from base model: {MODEL_PATH}")
    convert_fp32_model(base_model, fp32_path)
    create_compatibility_copy(fp32_path, TFLITE_DIR / "edge_classifier_fp32.tflite")

    convert_full_int8_model(base_model, int8_path, artifacts["X_train"])
    create_compatibility_copy(int8_path, TFLITE_DIR / "edge_classifier_int8.tflite")

    export_pruned_int8_model(base_model, pruned_int8_path, artifacts)

    print(f"Saved {fp32_path}")
    print(f"Saved {int8_path}")
    print(f"Saved {pruned_int8_path}")


if __name__ == "__main__":
    main()
