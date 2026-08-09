#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
MODEL_PATH="$ROOT_DIR/data/tflite/model_fp32.tflite"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing $MODEL_PATH"
  echo "Run ./scripts/run_training_demo.sh and ./scripts/run_model_tasks.sh first"
  exit 1
fi

start_python_service() {
  local name="$1"
  shift
  local pidfile="$LOG_DIR/${name}.pid"
  local logfile="$LOG_DIR/${name}.log"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    kill "$(cat "$pidfile")" 2>/dev/null || true
    rm -f "$pidfile"
    sleep 1
  fi

  PYTHONUNBUFFERED=1 nohup "$ROOT_DIR/.venv/bin/python" -u "$@" >"$logfile" 2>&1 &
  echo $! >"$pidfile"
  echo "$name started"
}

docker compose -f "$ROOT_DIR/docker-compose.yml" up -d mosquitto
start_python_service inference "$ROOT_DIR/src/mqtt/subscriber.py"
start_python_service psi_monitor "$ROOT_DIR/src/monitoring/drift_monitor.py" --mqtt

echo "Live stack started"
echo "Inference log: $LOG_DIR/inference.log"
echo "PSI log: $LOG_DIR/psi_monitor.log"
