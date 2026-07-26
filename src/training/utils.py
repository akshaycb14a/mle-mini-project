"""
utils.py
Shared helper utilities.
"""

from pathlib import Path

def ensure_directories():
    for folder in [
        Path("reports"),
        Path("data/models"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
