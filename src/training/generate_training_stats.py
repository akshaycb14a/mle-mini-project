import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing.dataset_builder import SlidingWindow
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.moving_average import MovingAverageFilter
from src.simulator.sensor_simulator import SensorSimulator
from src.training.normalization import (
    FEATURE_NAMES,
    build_training_stats,
    save_training_stats,
    make_legacy_scaler,
)


OUTPUT_PATH = Path("data/models/training_stats.npy")
LEGACY_SCALER_PATH = Path("data/models/scaler.pkl")


def collect_clean_feature_matrix(
    duration_seconds=600,
    sample_interval_seconds=1,
    window_size=30,
    step_size=10,
):
    simulator = SensorSimulator(truck_id="TRUCK_001", anomaly="none")
    temp_filter = MovingAverageFilter(window_size=5)
    vib_filter = MovingAverageFilter(window_size=5)
    window = SlidingWindow(size=window_size)

    feature_rows = []
    for sample_index in range(duration_seconds):
        reading = simulator.generate_reading()
        temp = temp_filter.update(reading["temperature"])
        vib = vib_filter.update(reading["vibration"])

        window.add(temp, vib, reading["door_open"], reading["status"])

        if window.ready() and (sample_index + 1) % step_size == 0:
            features = FeatureExtractor.extract(
                list(window.temperature),
                list(window.vibration),
            )
            feature_rows.append([features[name] for name in FEATURE_NAMES])

    if not feature_rows:
        raise RuntimeError("No feature windows were collected for stats generation")

    return np.asarray(feature_rows, dtype=np.float32)


def main():
    feature_matrix = collect_clean_feature_matrix()
    stats = build_training_stats(
        feature_matrix,
        source="10_minutes_clean_normal_simulation",
    )
    save_training_stats(stats, OUTPUT_PATH)

    legacy_scaler = make_legacy_scaler(stats)
    LEGACY_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(legacy_scaler, LEGACY_SCALER_PATH)

    print(f"Saved normalization stats to {OUTPUT_PATH}")
    print(f"Collected {stats['num_samples']} windows from clean normal data")


if __name__ == "__main__":
    main()
