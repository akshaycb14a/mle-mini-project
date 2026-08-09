#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

start_python_service() {
  local name="$1"
  shift
  local pidfile="$LOG_DIR/${name}.pid"
  local logfile="$LOG_DIR/${name}.log"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name already running"
    return
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
