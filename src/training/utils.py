"""
utils.py
Shared helper utilities.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.training.normalization import FEATURE_NAMES

def ensure_directories():
    for folder in [
        Path("reports"),
        Path("data/models"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset_frame(dataset_path="data/processed/training_dataset.csv"):
    return pd.read_csv(dataset_path)


def make_holdout_split(frame, labels, test_size=0.2, random_state=42):
    feature_frame = frame.loc[:, list(FEATURE_NAMES)]
    return train_test_split(
        feature_frame,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
