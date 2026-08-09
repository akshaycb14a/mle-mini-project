#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${WEBAPP_PORT:-5001}"

cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" -m src.webapp.app
