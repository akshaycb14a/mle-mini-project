import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.simulator.sensor_simulator import SensorSimulator


DATASET_PATH = Path("data/processed/training_dataset.csv")
WINDOW_TARGETS = {
    "none": 120,
    "temp_drift": 90,
    "combined": 90,
}


def readings_needed(window_count, window_size=30, step_size=10):
    if window_count <= 0:
        return 0
    return window_size + step_size * (window_count - 1)


def generate_windows(anomaly, window_target, dataset_path):
    simulator = SensorSimulator(truck_id="TRUCK_001", anomaly=anomaly)
    pipeline = PreprocessingPipeline(save_dataset=True, dataset_path=dataset_path)

    saved_windows = 0
    total_readings = readings_needed(window_target)

    for _ in range(total_readings):
        reading = simulator.generate_reading()
        features = pipeline.process(reading)
        if features is not None:
            saved_windows += 1

    return saved_windows, total_readings


def main():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        DATASET_PATH.unlink()

    for anomaly, window_target in WINDOW_TARGETS.items():
        saved_windows, total_readings = generate_windows(
            anomaly,
            window_target,
            DATASET_PATH,
        )
        print(
            f"Generated anomaly={anomaly} windows={saved_windows} "
            f"from readings={total_readings}"
        )

    print(f"Saved dataset to {DATASET_PATH}")


if __name__ == "__main__":
    main()
