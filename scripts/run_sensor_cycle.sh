#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/run_sensor.sh" --anomaly none --count 60 --interval 0.2
"$ROOT_DIR/scripts/run_sensor.sh" --anomaly temp_drift --count 300 --interval 0.05
"$ROOT_DIR/scripts/run_sensor.sh" --anomaly vibration --count 300 --interval 0.05
"$ROOT_DIR/scripts/run_sensor.sh" --anomaly combined --count 420 --interval 0.05
