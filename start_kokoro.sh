#!/usr/bin/env bash
set -euo pipefail

export KOKORO_PORT="${KOKORO_PORT:-8880}"
export KOKORO_MODEL_DIR="${KOKORO_MODEL_DIR:-/workspace/luma/tts/kokoro}"

echo "[kokoro] starting on :${KOKORO_PORT} at $(date)"
echo "[kokoro] python=/workspace/luma/tts/kokoro-env/bin/python"
echo "[kokoro] model_dir=${KOKORO_MODEL_DIR}"

cd /workspace/luma/tts

exec /workspace/luma/tts/kokoro-env/bin/uvicorn \
  kokoro_server:app \
  --host 0.0.0.0 \
  --port "${KOKORO_PORT}"
