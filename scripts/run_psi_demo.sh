#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR=/private/tmp/matplotlib

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/monitoring/drift_monitor.py" --build-reference
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/src/monitoring/drift_monitor.py" --demo --demo-phase-samples 120,120,120 --demo-report-every 20
