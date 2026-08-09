#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"

stop_pid_file() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

stop_pid_file "$LOG_DIR/inference.pid"
stop_pid_file "$LOG_DIR/psi_monitor.pid"

pkill -f "src/simulator/sensor_simulator.py" 2>/dev/null || true
docker compose -f "$ROOT_DIR/docker-compose.yml" down

echo "Live stack stopped"
