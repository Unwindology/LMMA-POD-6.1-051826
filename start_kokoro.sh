#!/usr/bin/env bash
set -euo pipefail

KOKORO_PORT="${KOKORO_PORT:-8880}"
KOKORO_DIR="/workspace/luma/tts"
KOKORO_ENV="$KOKORO_DIR/kokoro-env"
KOKORO_MODEL_DIR="${KOKORO_MODEL_DIR:-$KOKORO_DIR/kokoro}"

echo "[kokoro] starting on :$KOKORO_PORT at $(date)"
echo "[kokoro] env: $KOKORO_ENV"
echo "[kokoro] model dir: $KOKORO_MODEL_DIR"

if [ ! -x "$KOKORO_ENV/bin/python" ]; then
  echo "[kokoro] FATAL: missing $KOKORO_ENV/bin/python"
  exit 1
fi

"$KOKORO_ENV/bin/python" -c "import kokoro_onnx; print('[kokoro] kokoro_onnx import OK')" || {
  echo "[kokoro] FATAL: kokoro_onnx not importable"
  exit 1
}

cd "$KOKORO_DIR"
export KOKORO_PORT KOKORO_MODEL_DIR

exec "$KOKORO_ENV/bin/uvicorn" \
  kokoro_server:app \
  --host 0.0.0.0 \
  --port "$KOKORO_PORT"
