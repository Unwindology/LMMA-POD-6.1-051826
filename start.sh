#!/usr/bin/env bash
# LUMA POD 6.1 CLEAN start.sh
# No Open WebUI. No actor-gemma4. No app-venv.
# Services:
#   vLLM       : 8000
#   LUMA RAG   : 8001
#   Gateway    : 8080
#   Tools      : 8010 if LUMA_TOOLS_API_KEY is set
#   Kokoro TTS : 8880 if KOKORO_PORT is set
#   JupyterLab : 8888
#   Tailscale  : optional via TAILSCALE_AUTHKEY

set -euo pipefail
START_T=$(date +%s)

export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR" /workspace/logs /workspace/luma /workspace/luma/data /workspace/luma/tts

if [ -z "${GIT_HUB_TOKEN:-}" ] && [ -n "${GET_HUB_TOKEN:-}" ]; then
  export GIT_HUB_TOKEN="$GET_HUB_TOKEN"
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

RUN_MODE="${RUN_MODE:-pod}"
PY="${PY:-python3}"

# Ports
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
RAW_VLLM_PORT="${RAW_VLLM_PORT:-${VLLM_PORT:-8000}}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${RAW_VLLM_PORT}}"
LUMA_PORT="${LUMA_PORT:-${RAG_PORT:-8001}}"
LUMA_RAG_URL="${LUMA_RAG_URL:-http://localhost:${LUMA_PORT}}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
LUMA_TOOLS_PORT="${LUMA_TOOLS_PORT:-8010}"
KOKORO_PORT="${KOKORO_PORT:-8880}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

# Model
MODEL_ID="${MODEL_ID:-QuantTrio/gemma-4-31B-it-AWQ}"
MODEL_NAME="${MODEL_NAME:-luma}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-$MODEL_NAME}"

# vLLM tuning
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL:-1}}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-${PIPELINE_PARALLEL:-1}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${GPU_MEM_UTIL:-0.85}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
DTYPE="${DTYPE:-auto}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-${EXTRA_VLLM_ARGS:-}}"

ENABLE_GEMMA4_REASONING="${ENABLE_GEMMA4_REASONING:-false}"
ENABLE_GEMMA4_TOOLS="${ENABLE_GEMMA4_TOOLS:-false}"
GEMMA4_REASONING_PARSER="${GEMMA4_REASONING_PARSER:-gemma4}"
GEMMA4_TOOL_PARSER="${GEMMA4_TOOL_PARSER:-gemma4}"

# RAG/Gateway
HF_REPO_ID="${HF_REPO_ID:-chap18700/luma-raw-data}"
FRACTAL_DB="${FRACTAL_DB:-/workspace/luma/data/fractal_index.db}"
LUMA_CURRENT_STATE_PATH="${LUMA_CURRENT_STATE_PATH:-/workspace/luma/LUMA_CURRENT_STATE.md}"
LUMA_README_PATH="${LUMA_README_PATH:-/workspace/luma/LUMA_README.md}"
FLUSH_INTERVAL_SECONDS="${FLUSH_INTERVAL_SECONDS:-180}"

LUMA_RETRIEVE_SAMPLE="${LUMA_RETRIEVE_SAMPLE:-5000}"
LUMA_TOP_K="${LUMA_TOP_K:-6}"
LUMA_MIN_SCORE="${LUMA_MIN_SCORE:-0.35}"
LUMA_KEYWORD_LIMIT="${LUMA_KEYWORD_LIMIT:-20}"
LUMA_KEYWORD_MIN_LEN="${LUMA_KEYWORD_MIN_LEN:-4}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-60000}"
MAX_RAG_CHARS="${MAX_RAG_CHARS:-16000}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS_DEFAULT:-2048}"

# Network/tools
TAILSCALE_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-luma-pod}"
ENABLE_CLOUDFLARE="${ENABLE_CLOUDFLARE:-false}"
CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"
OPEN_TERMINAL_API_KEY="${OPEN_TERMINAL_API_KEY:-}"  # reserved; not launched in this clean image
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"                  # empty = direct Lab when port is private/Tailscale

echo "============================================================"
echo " LUMA POD 6.1 CLEAN start.sh · RUN_MODE=$RUN_MODE · $(date)"
echo "============================================================"
echo "  vLLM:      $VLLM_BASE_URL  ($MODEL_ID -> $MODEL_NAME)"
echo "  RAG:       $LUMA_RAG_URL   (sample=$LUMA_RETRIEVE_SAMPLE kw_lim=$LUMA_KEYWORD_LIMIT)"
echo "  Gateway:   http://localhost:$GATEWAY_PORT"
echo "  Tools:     http://localhost:$LUMA_TOOLS_PORT  (only if LUMA_TOOLS_API_KEY is set)"
echo "  Kokoro:    http://localhost:$KOKORO_PORT"
echo "  Jupyter:   http://localhost:$JUPYTER_PORT/lab"
echo "  TP=$TENSOR_PARALLEL_SIZE  PP=$PIPELINE_PARALLEL_SIZE  mem=$GPU_MEMORY_UTILIZATION  ctx=$MAX_MODEL_LEN"
echo "  max_num_seqs=$MAX_NUM_SEQS  max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS  kv_cache_dtype=$KV_CACHE_DTYPE"
echo "  tailscale_hostname=$TAILSCALE_HOSTNAME  tailscale_key=$([ -n "$TAILSCALE_AUTHKEY" ] && echo "set" || echo "not set")"
echo "  OPEN_WEBUI: removed from pod image"

wait_for_health () {
  local url="$1"; local name="$2"; local tries="${3:-60}"
  echo -n "  waiting on $name ($url) "
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo " ✓"; return 0
    fi
    sleep 5; echo -n "."
  done
  echo " ✗ (timeout)"; return 1
}

is_listening () {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1\$"
}

guard_python () {
  echo "  Python: checking vLLM stack with $PY"
  "$PY" - <<'PY'
import importlib.util, sys, torch, vllm
print("  Python exe:", sys.executable)
print("  vLLM:", getattr(vllm, "__version__", "unknown"))
print("  torch:", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
if importlib.util.find_spec("open_webui") is not None:
    raise SystemExit("FATAL: open_webui is installed; refusing clean pod boot")
PY
}

ensure_state_files () {
  if [ ! -s "$LUMA_CURRENT_STATE_PATH" ] && [ -s "$LUMA_README_PATH" ]; then
    cp "$LUMA_README_PATH" "$LUMA_CURRENT_STATE_PATH"
  fi
  if [ -s "$LUMA_CURRENT_STATE_PATH" ]; then
    echo "  State: using baked current state ($(stat -c%s "$LUMA_CURRENT_STATE_PATH") bytes)"
  else
    echo "  State: WARNING no current-state file found at $LUMA_CURRENT_STATE_PATH"
  fi
}

ensure_db () {
  mkdir -p "$(dirname "$FRACTAL_DB")"
  local size=0
  [ -f "$FRACTAL_DB" ] && size=$(stat -c%s "$FRACTAL_DB")

  if [ -z "${HF_TOKEN:-}" ]; then
    if [ "$size" -gt 100000000 ]; then
      echo "  DB: no HF_TOKEN, using local ($((size/1024/1024)) MB)"
    else
      echo "  DB: no HF_TOKEN and no local DB — RAG may boot empty"
    fi
    return
  fi

  echo "  DB: fetching fractal_index.db from $HF_REPO_ID"
  if "$PY" - <<PY
import os, sys
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(
        repo_id="$HF_REPO_ID",
        repo_type="dataset",
        filename="fractal_index.db",
        local_dir="$(dirname "$FRACTAL_DB")",
        token=os.environ["HF_TOKEN"],
    )
    print("  DB ->", p, os.path.getsize(p) // (1024*1024), "MB")
except Exception as e:
    print("  DB fetch failed:", e, file=sys.stderr)
    sys.exit(1)
PY
  then :
  else
    if [ "$size" -gt 100000000 ]; then
      echo "  DB: fetch failed, falling back to local ($((size/1024/1024)) MB)"
    else
      echo "  DB: fetch failed and no local DB — RAG may boot empty"
    fi
  fi
}

start_tailscale () {
  [ -n "$TAILSCALE_AUTHKEY" ] || { echo "  Tailscale: TAILSCALE_AUTHKEY not set — skipping"; return 0; }
  if ! command -v tailscale >/dev/null 2>&1; then
    echo "  Tailscale: installing..."
    curl -fsSL https://tailscale.com/install.sh | sh -s - >/dev/null 2>&1 || true
  fi

  pkill -f tailscaled 2>/dev/null || true
  rm -f /tmp/tailscaled.sock
  sleep 2

  echo "  Tailscale: starting daemon..."
  /usr/sbin/tailscaled \
    --tun=userspace-networking \
    --state=/workspace/tailscaled.state \
    --socket=/tmp/tailscaled.sock \
    >> /workspace/logs/tailscale.log 2>&1 &
  disown
  sleep 5

  tailscale --socket=/tmp/tailscaled.sock up \
    --authkey="$TAILSCALE_AUTHKEY" \
    --hostname="$TAILSCALE_HOSTNAME" \
    --accept-routes \
    --accept-dns=false \
    --force-reauth \
    >> /workspace/logs/tailscale.log 2>&1 || {
      echo "  !! tailscale up failed — check /workspace/logs/tailscale.log"
      tail -20 /workspace/logs/tailscale.log || true
      return 1
    }

  local ts_ip
  ts_ip=$(tailscale --socket=/tmp/tailscaled.sock ip -4 2>/dev/null || echo "pending")
  echo "  Tailscale: connected — IP $ts_ip hostname=$TAILSCALE_HOSTNAME"
}

start_jupyter () {
  if is_listening "$JUPYTER_PORT"; then
    echo "  JupyterLab: already listening on $JUPYTER_PORT"
    return
  fi
  echo "  JupyterLab: launching on port $JUPYTER_PORT"
  nohup "$PY" -m jupyter lab \
    --ip=0.0.0.0 \
    --port="$JUPYTER_PORT" \
    --no-browser \
    --ServerApp.token="$JUPYTER_TOKEN" \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.root_dir=/workspace \
    --ServerApp.default_url=/lab \
    --allow-root \
    > /workspace/logs/jupyter.log 2>&1 &
  disown
  echo "  JupyterLab PID=$! — http://[tailscale-ip]:$JUPYTER_PORT/lab  log: /workspace/logs/jupyter.log"
}

start_vllm () {
  if is_listening "$RAW_VLLM_PORT"; then
    echo "  vLLM: already listening on $RAW_VLLM_PORT"
    return
  fi

  echo "  vLLM: launching $MODEL_ID"
  local args=( --host "$VLLM_HOST" --port "$RAW_VLLM_PORT"
               --served-model-name "$MODEL_NAME"
               --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
               --max-model-len "$MAX_MODEL_LEN"
               --max-num-seqs "$MAX_NUM_SEQS"
               --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
               --kv-cache-dtype "$KV_CACHE_DTYPE"
               --dtype "$DTYPE"
               --trust-remote-code )

  if [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    args+=( --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" )
  fi
  if [ "$PIPELINE_PARALLEL_SIZE" -gt 1 ]; then
    args+=( --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" )
  fi
  if [ "$ENABLE_GEMMA4_REASONING" = "true" ]; then
    args+=( --reasoning-parser "${GEMMA4_REASONING_PARSER:-gemma4}" )
  fi
  if [ "$ENABLE_GEMMA4_TOOLS" = "true" ]; then
    args+=( --enable-auto-tool-choice --tool-call-parser "${GEMMA4_TOOL_PARSER:-gemma4}" )
  fi

  # shellcheck disable=SC2206
  local extra=( $VLLM_EXTRA_ARGS )

  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" "${args[@]}" "${extra[@]}" \
    > /workspace/logs/vllm.log 2>&1 &
  disown
  echo "  vLLM PID=$! — log: /workspace/logs/vllm.log"
  wait_for_health "http://localhost:$RAW_VLLM_PORT/v1/models" "vLLM" 90 || true
}

start_rag () {
  if is_listening "$LUMA_PORT"; then
    echo "  RAG: already listening on $LUMA_PORT"
    return
  fi
  echo "  RAG: launching on $LUMA_PORT"
  export FRACTAL_DB HF_REPO_ID HF_TOKEN LUMA_PORT
  export LUMA_RETRIEVE_SAMPLE LUMA_TOP_K LUMA_MIN_SCORE LUMA_KEYWORD_LIMIT LUMA_KEYWORD_MIN_LEN
  nohup "$PY" /workspace/luma/luma_server.py > /workspace/logs/rag.log 2>&1 &
  disown
  echo "  RAG PID=$! — log: /workspace/logs/rag.log"
  wait_for_health "http://localhost:$LUMA_PORT/health" "RAG" 24 || true
}

start_gateway () {
  if is_listening "$GATEWAY_PORT"; then
    echo "  Gateway: already listening on $GATEWAY_PORT"
    return
  fi
  echo "  Gateway: launching on $GATEWAY_PORT"
  export VLLM_BASE_URL LUMA_RAG_URL GATEWAY_PORT MODEL_NAME VLLM_MODEL_NAME
  export LUMA_CURRENT_STATE_PATH LUMA_README_PATH HF_REPO_ID FLUSH_INTERVAL_SECONDS HF_TOKEN
  export MAX_CONTEXT_CHARS MAX_RAG_CHARS MAX_TOKENS_DEFAULT
  nohup "$PY" /workspace/luma/api_gateway.py > /workspace/logs/gateway.log 2>&1 &
  disown
  echo "  Gateway PID=$! — log: /workspace/logs/gateway.log"
  wait_for_health "http://localhost:$GATEWAY_PORT/health" "Gateway" 12 || true
}

start_kokoro () {
  [ -n "${KOKORO_PORT:-}" ] || { echo "  Kokoro: KOKORO_PORT unset — skipping"; return 0; }
  if is_listening "$KOKORO_PORT"; then
    echo "  Kokoro: already listening on $KOKORO_PORT"
    return
  fi
  echo "  Kokoro: launching on $KOKORO_PORT"
  nohup /workspace/luma/tts/start_kokoro.sh > /workspace/logs/kokoro.log 2>&1 &
  disown
  echo "  Kokoro PID=$! — log: /workspace/logs/kokoro.log"
  wait_for_health "http://localhost:$KOKORO_PORT/health" "Kokoro TTS" 18 || true
}

start_tools () {
  if [ -z "${LUMA_TOOLS_API_KEY:-}" ]; then
    echo "  Tools: LUMA_TOOLS_API_KEY not set — skipping"
    return 0
  fi
  if is_listening "$LUMA_TOOLS_PORT"; then
    echo "  Tools: already listening on $LUMA_TOOLS_PORT"
    return 0
  fi
  echo "  Tools: launching on $LUMA_TOOLS_PORT"
  export LUMA_TOOLS_PORT LUMA_LOG_DIR="${LUMA_LOG_DIR:-/workspace/logs}" LUMA_PROJECT_DIR="${LUMA_PROJECT_DIR:-/workspace/luma}" LUMA_WORKLOG_DIR="${LUMA_WORKLOG_DIR:-/workspace/luma/worklog}"
  mkdir -p "$LUMA_LOG_DIR" "$LUMA_WORKLOG_DIR"
  nohup "$PY" /workspace/luma/luma_tools_server.py > /workspace/logs/luma_tools_server.log 2>&1 &
  disown
  echo "  Tools PID=$! — log: /workspace/logs/luma_tools_server.log"
}

start_cloudflared () {
  [ "$ENABLE_CLOUDFLARE" = "true" ] || return 0
  [ -n "$CF_TUNNEL_TOKEN" ] || { echo "  cloudflared: ENABLE_CLOUDFLARE=true but CF_TUNNEL_TOKEN unset"; return 0; }
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "  cloudflared: installing"
    curl -sfL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
  fi
  nohup cloudflared tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" > /workspace/logs/cloudflared.log 2>&1 &
  disown
  echo "  cloudflared PID=$! — log: /workspace/logs/cloudflared.log"
}

guard_python
ensure_state_files
ensure_db

case "$RUN_MODE" in
  pod)
    start_tailscale
    start_jupyter
    start_vllm
    start_rag
    start_gateway
    start_kokoro
    start_tools
    start_cloudflared
    ;;
  serverless)
    start_rag
    start_gateway
    start_tools
    start_cloudflared
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE (expected pod|serverless)"
    exit 1
    ;;
esac

END_T=$(date +%s)
echo ""
echo "============================================================"
echo " start.sh finished in $((END_T - START_T))s ($RUN_MODE mode)"
echo "============================================================"
echo " Tail logs:"
echo "   tail -f /workspace/logs/tailscale.log"
echo "   tail -f /workspace/logs/vllm.log"
echo "   tail -f /workspace/logs/rag.log"
echo "   tail -f /workspace/logs/gateway.log"
echo "   tail -f /workspace/logs/kokoro.log"
echo "   tail -f /workspace/logs/jupyter.log"
echo "   tail -f /workspace/logs/luma_tools_server.log"
echo ""

if [ "${KEEP_FOREGROUND:-true}" = "true" ]; then
  case "$RUN_MODE" in
    pod)        exec tail -f /workspace/logs/vllm.log /workspace/logs/gateway.log /workspace/logs/kokoro.log ;;
    serverless) exec tail -f /workspace/logs/rag.log  /workspace/logs/gateway.log ;;
  esac
fi

echo "KEEP_FOREGROUND=false — start.sh returning."
