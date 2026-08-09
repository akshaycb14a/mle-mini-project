#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SLIDES_PORT:-8008}"

cd "$ROOT_DIR/docs"
echo "Open http://127.0.0.1:${PORT}/demo_slideshow.html"
exec "$ROOT_DIR/.venv/bin/python" -m http.server "$PORT" --bind 127.0.0.1
