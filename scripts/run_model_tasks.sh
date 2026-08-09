#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR=/private/tmp/matplotlib

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/training/normalization_experiment.py"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/deployment/build_deployment_models.py"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/deployment/benchmark_models.py"
