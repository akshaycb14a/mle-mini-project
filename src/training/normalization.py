from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "temp_mean",
    "temp_std",
    "temp_rate",
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
)

DEFAULT_STATS_PATH = Path("data/models/training_stats.npy")


def build_training_stats(feature_matrix, feature_names=FEATURE_NAMES, source=None):
    matrix = np.asarray(feature_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("feature_matrix must be a 2D array")

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)

    stats = {
        "feature_names": list(feature_names),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "num_samples": int(matrix.shape[0]),
        "source": source or "clean_normal_class",
    }
    return stats


def save_training_stats(stats, path=DEFAULT_STATS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stats, allow_pickle=True)


def load_training_stats(path=DEFAULT_STATS_PATH):
    path = Path(path)
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        stats = payload.item()
    else:
        stats = payload.item()

    feature_names = stats.get("feature_names", list(FEATURE_NAMES))
    stats["feature_names"] = list(feature_names)
    stats["mean"] = np.asarray(stats["mean"], dtype=np.float32)
    stats["std"] = np.asarray(stats["std"], dtype=np.float32)
    stats["std"] = np.where(stats["std"] < 1e-6, 1e-6, stats["std"])
    return stats


def normalize_matrix(matrix, stats):
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (matrix - mean) / std


def normalize_dataframe(frame, stats):
    columns = list(stats.get("feature_names", FEATURE_NAMES))
    matrix = frame.loc[:, columns].to_numpy(dtype=np.float32)
    return normalize_matrix(matrix, stats)


def make_legacy_scaler(stats):
    scaler = StandardScaler()
    feature_count = len(stats["feature_names"])
    scaler.fit(np.zeros((1, feature_count), dtype=np.float32))
    scaler.mean_ = np.asarray(stats["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(stats["std"], dtype=np.float64)
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = feature_count
    scaler.n_samples_seen_ = 1
    return scaler
