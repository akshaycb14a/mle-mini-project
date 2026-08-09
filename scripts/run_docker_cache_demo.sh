#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cp "$ROOT_DIR/data/tflite/model_fp32.tflite" "$ROOT_DIR/data/tflite/model_demo.tflite"
docker build --build-arg MODEL_FILE=model_demo.tflite -t inference-service:ota "$ROOT_DIR"
cp "$ROOT_DIR/data/tflite/model_int8.tflite" "$ROOT_DIR/data/tflite/model_demo.tflite"
docker build --build-arg MODEL_FILE=model_demo.tflite -t inference-service:ota "$ROOT_DIR"
