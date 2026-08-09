#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR=/private/tmp/matplotlib

echo "Generating training statistics from clean normal data"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/training/generate_training_stats.py"

echo "Training the classifier"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/training/train_model.py"

echo "Evaluating the trained model"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/training/evaluate.py"
