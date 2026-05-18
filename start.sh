#!/usr/bin/env bash
# start.sh — single env-driven launcher for the LUMA stack.
#
# PRIMARY env var names match the live working A100 pod EXACTLY:
#   RAW_VLLM_PORT, LUMA_PORT, GATEWAY_PORT, WEBUI_PORT,
#   MODEL_NAME (= "luma"), VLLM_MODEL_NAME,
#   GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN,
#   MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, KV_CACHE_DTYPE,
#   TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE,
#   ENABLE_REASONING,
#   MAX_CONTEXT_CHARS, MAX_RAG_CHARS, MAX_TOKENS_DEFAULT,
#   FRACTAL_DB, LUMA_CURRENT_STATE_PATH, LUMA_README_PATH,
#   HF_REPO_ID, FLUSH_INTERVAL_SECONDS, VLLM_BASE_URL, LUMA_RAG_URL.
#
# Older short names (VLLM_PORT, RAG_PORT, GPU_MEM_UTIL, TENSOR_PARALLEL,
# PIPELINE_PARALLEL) are accepted as backward-compat aliases.
#
# RUN_MODE:
#   pod         (default) — full stack: vLLM + RAG + gateway + Open WebUI
#   serverless           — RAG + gateway only (vLLM is on a separate endpoint)

set -euo pipefail
START_T=$(date +%s)

export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR" /workspace/logs /workspace/luma

if [ -z "${GIT_HUB_TOKEN:-}" ] && [ -n "${GET_HUB_TOKEN:-}" ]; then
  export GIT_HUB_TOKEN="$GET_HUB_TOKEN"
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

RUN_MODE="${RUN_MODE:-pod}"

# Inference engine endpoints — names from the live working pod.
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
RAW_VLLM_PORT="${RAW_VLLM_PORT:-${VLLM_PORT:-8000}}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${RAW_VLLM_PORT}}"

# RAG + gateway
LUMA_PORT="${LUMA_PORT:-${RAG_PORT:-8001}}"
LUMA_RAG_URL="${LUMA_RAG_URL:-http://localhost:${LUMA_PORT}}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
WEBUI_PORT="${WEBUI_PORT:-3000}"

# Model identity — alias is `luma`, NOT `luma-aethel-31b`.
MODEL_ID="${MODEL_ID:-RedHatAI/gemma-4-31B-it-NVFP4}"
MODEL_NAME="${MODEL_NAME:-luma}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-$MODEL_NAME}"

# vLLM tuning — defaults match the live working pod.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL:-1}}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-${PIPELINE_PARALLEL:-1}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${GPU_MEM_UTIL:-0.85}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
DTYPE="${DTYPE:-auto}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-$EXTRA_VLLM_ARGS}"

# Structured thinking output — set false to disable
ENABLE_REASONING="${ENABLE_REASONING:-false}"
ENABLE_GEMMA4_REASONING="${ENABLE_GEMMA4_REASONING:-false}"
ENABLE_GEMMA4_TOOLS="${ENABLE_GEMMA4_TOOLS:-false}"
GEMMA4_REASONING_PARSER="${GEMMA4_REASONING_PARSER:-gemma4}"
GEMMA4_TOOL_PARSER="${GEMMA4_TOOL_PARSER:-gemma4}"

# RAG retrieval tuning
LUMA_RETRIEVE_SAMPLE="${LUMA_RETRIEVE_SAMPLE:-5000}"
LUMA_TOP_K="${LUMA_TOP_K:-6}"
LUMA_MIN_SCORE="${LUMA_MIN_SCORE:-0.35}"
LUMA_KEYWORD_LIMIT="${LUMA_KEYWORD_LIMIT:-20}"
LUMA_KEYWORD_MIN_LEN="${LUMA_KEYWORD_MIN_LEN:-4}"

# Gateway context budget — protect against context overflow
# MAX_CONTEXT_CHARS: total budget for injected control blocks (~4 chars/token)
# MAX_RAG_CHARS: hard cap on RAG context block before truncation
# MAX_TOKENS_DEFAULT: fallback max_tokens if caller doesn't set one
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-60000}"
MAX_RAG_CHARS="${MAX_RAG_CHARS:-16000}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS_DEFAULT:-2048}"

# Repos / persistence — paths match the live working pod.
LUMA_REPO="${LUMA_REPO:-Unwindology/aethel-luma}"
HF_REPO_ID="${HF_REPO_ID:-chap18700/luma-raw-data}"
FRACTAL_DB="${FRACTAL_DB:-/workspace/luma/aethel-luma/fractal_index.db}"
LUMA_CURRENT_STATE_PATH="${LUMA_CURRENT_STATE_PATH:-/workspace/luma/LUMA_CURRENT_STATE.md}"
LUMA_README_PATH="${LUMA_README_PATH:-/workspace/luma/LUMA_README.md}"
FLUSH_INTERVAL_SECONDS="${FLUSH_INTERVAL_SECONDS:-180}"

export WEBUI_AUTH="${WEBUI_AUTH:-true}"
export ENABLE_SIGNUP="${ENABLE_SIGNUP:-true}"
export DEFAULT_USER_ROLE="${DEFAULT_USER_ROLE:-admin}"
export DEFAULT_MODELS="${DEFAULT_MODELS:-$MODEL_NAME}"
export ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API:-false}"

PY="${PY:-python3}"
if [ -x /workspace/app-venv/bin/python ]; then
  PY=/workspace/app-venv/bin/python
elif [ -x /opt/venv/bin/python ]; then
  PY=/opt/venv/bin/python
fi

ENABLE_CLOUDFLARE="${ENABLE_CLOUDFLARE:-false}"
CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"

# Tailscale — mesh VPN, preferred over Cloudflare tunnel
TAILSCALE_AUTHKEY="${TAILSCALE_AUTHKEY:-}"
TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-luma-pod}"

# Open Terminal — REST API shell for Open WebUI agent integration
OPEN_TERMINAL_PORT="${OPEN_TERMINAL_PORT:-8002}"
OPEN_TERMINAL_API_KEY="${OPEN_TERMINAL_API_KEY:-}"

echo "============================================================"
echo " LUMA start.sh  ·  RUN_MODE=$RUN_MODE  ·  $(date)"
echo "============================================================"
echo "  vLLM:    $VLLM_BASE_URL  ($MODEL_ID -> $MODEL_NAME)"
echo "  RAG:     $LUMA_RAG_URL   (sample=$LUMA_RETRIEVE_SAMPLE kw_lim=$LUMA_KEYWORD_LIMIT)"
echo "  Gateway: http://localhost:$GATEWAY_PORT"
echo "  WebUI:   http://localhost:$WEBUI_PORT"
echo "  TP=$TENSOR_PARALLEL_SIZE  PP=$PIPELINE_PARALLEL_SIZE  mem=$GPU_MEMORY_UTILIZATION  ctx=$MAX_MODEL_LEN"
echo "  max_num_seqs=$MAX_NUM_SEQS  max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS  kv_cache_dtype=$KV_CACHE_DTYPE"
echo "  enable_reasoning=$ENABLE_REASONING  max_tokens_default=$MAX_TOKENS_DEFAULT  max_rag_chars=$MAX_RAG_CHARS"
echo "  tailscale_hostname=$TAILSCALE_HOSTNAME  tailscale_key=$([ -n "$TAILSCALE_AUTHKEY" ] && echo "set" || echo "not set")"
echo "  open_terminal_port=$OPEN_TERMINAL_PORT  open_terminal_key=$([ -n "$OPEN_TERMINAL_API_KEY" ] && echo "set" || echo "not set")"

wait_for_health () {
  local url="$1"; local name="$2"; local tries="${3:-60}"
  echo -n "  waiting on $name ($url) "
  for i in $(seq 1 "$tries"); do
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

ensure_readme () {
  mkdir -p "$(dirname "$LUMA_README_PATH")" "$(dirname "$LUMA_CURRENT_STATE_PATH")"
  local has_baked=0
  if [ -f "$LUMA_CURRENT_STATE_PATH" ] && [ "$(stat -c%s "$LUMA_CURRENT_STATE_PATH")" -gt 5000 ]; then
    has_baked=1
  fi

  if [ -z "${GIT_HUB_TOKEN:-}" ]; then
    if [ "$has_baked" = 1 ]; then
      echo "  README: no GIT_HUB_TOKEN, using baked default ($(stat -c%s "$LUMA_CURRENT_STATE_PATH") bytes at $LUMA_CURRENT_STATE_PATH)"
    else
      echo "  README: no GIT_HUB_TOKEN and no baked default — gateway identity will be empty"
    fi
    return
  fi

  echo "  README: fetching latest from $LUMA_REPO main"
  local tmp
  tmp=$(mktemp)
  if curl -sfL \
       -H "Authorization: Bearer $GIT_HUB_TOKEN" \
       -H "Accept: application/vnd.github.raw" \
       "https://api.github.com/repos/$LUMA_REPO/contents/LUMA_README.md" \
       -o "$tmp" && [ "$(stat -c%s "$tmp")" -gt 5000 ]; then
    cp "$tmp" "$LUMA_README_PATH"
    cp "$tmp" "$LUMA_CURRENT_STATE_PATH"
    rm -f "$tmp"
    echo "  README: updated ($(stat -c%s "$LUMA_CURRENT_STATE_PATH") bytes) -> $LUMA_README_PATH + $LUMA_CURRENT_STATE_PATH"
  else
    rm -f "$tmp"
    if [ "$has_baked" = 1 ]; then
      echo "  README: fetch failed, falling back to baked default ($(stat -c%s "$LUMA_CURRENT_STATE_PATH") bytes)"
    else
      echo "  README: fetch failed and no baked default — gateway identity will be empty"
    fi
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
      echo "  DB: no HF_TOKEN and no local DB — RAG will boot empty"
    fi
    return
  fi

  echo "  DB: fetching latest fractal_index.db from $HF_REPO_ID"
  if $PY - <<PY
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
      echo "  DB: fetch failed and no local DB — RAG will boot empty"
    fi
  fi
}

start_vllm () {
  if is_listening "$RAW_VLLM_PORT"; then
    echo "  vLLM: already listening on $RAW_VLLM_PORT — leaving it alone"
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
    # --enable-reasoning removed — not a valid vLLM argument.
  if [ "${ENABLE_GEMMA4_REASONING}" = "true" ]; then
    args+=( --reasoning-parser "${GEMMA4_REASONING_PARSER:-gemma4}" )
    echo "  vLLM: --reasoning-parser ${GEMMA4_REASONING_PARSER:-gemma4} ON"
  fi
  if [ "${ENABLE_GEMMA4_TOOLS}" = "true" ]; then
    args+=( --enable-auto-tool-choice --tool-call-parser "${GEMMA4_TOOL_PARSER:-gemma4}" )
    echo "  vLLM: --enable-auto-tool-choice --tool-call-parser ${GEMMA4_TOOL_PARSER:-gemma4} ON"
  fi
  # shellcheck disable=SC2206
  local extra=( $VLLM_EXTRA_ARGS )

  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" "${args[@]}" "${extra[@]}" \
    > /workspace/logs/vllm.log 2>&1 &
  disown
  echo "  vLLM PID=$! — log: /workspace/logs/vllm.log"
  wait_for_health "http://localhost:$RAW_VLLM_PORT/v1/models" "vLLM" 60 || true
}

start_rag () {
  if is_listening "$LUMA_PORT"; then
    echo "  RAG: already listening on $LUMA_PORT — leaving it alone"
    return
  fi
  echo "  RAG: launching luma_server.py on $LUMA_PORT"
  export FRACTAL_DB HF_REPO_ID HF_TOKEN LUMA_PORT
  export LUMA_RETRIEVE_SAMPLE LUMA_TOP_K LUMA_MIN_SCORE
  export LUMA_KEYWORD_LIMIT LUMA_KEYWORD_MIN_LEN
  nohup $PY /workspace/luma/luma_server.py \
    > /workspace/logs/rag.log 2>&1 &
  disown
  echo "  RAG PID=$! — log: /workspace/logs/rag.log"
  wait_for_health "http://localhost:$LUMA_PORT/health" "RAG" 24 || true
}

start_gateway () {
  if is_listening "$GATEWAY_PORT"; then
    echo "  Gateway: already listening on $GATEWAY_PORT — leaving it alone"
    return
  fi
  echo "  Gateway: launching api_gateway.py on $GATEWAY_PORT"
  export VLLM_BASE_URL LUMA_RAG_URL GATEWAY_PORT MODEL_NAME VLLM_MODEL_NAME
  export LUMA_CURRENT_STATE_PATH LUMA_README_PATH HF_REPO_ID FLUSH_INTERVAL_SECONDS HF_TOKEN
  export MAX_CONTEXT_CHARS MAX_RAG_CHARS MAX_TOKENS_DEFAULT
  nohup $PY /workspace/luma/api_gateway.py \
    > /workspace/logs/gateway.log 2>&1 &
  disown
  echo "  Gateway PID=$! — log: /workspace/logs/gateway.log"
  wait_for_health "http://localhost:$GATEWAY_PORT/health" "Gateway" 12 || true
}

start_webui () {
  if is_listening "$WEBUI_PORT"; then
    echo "  WebUI: already listening on $WEBUI_PORT — leaving it alone"
    return
  fi
  echo "  WebUI: launching open-webui on $WEBUI_PORT"
  export OPENAI_API_BASE_URLS="http://localhost:${GATEWAY_PORT}/v1"
  export OPENAI_API_KEYS="${OPENAI_API_KEYS:-luma}"
  nohup /workspace/app-venv/bin/open-webui serve --port "$WEBUI_PORT" \
    > /workspace/logs/openwebui.log 2>&1 &
  disown
  echo "  WebUI PID=$! — log: /workspace/logs/openwebui.log"
}

start_cloudflared () {
  [ "$ENABLE_CLOUDFLARE" = "true" ] || return 0
  [ -n "$CF_TUNNEL_TOKEN" ] || { echo "  cloudflared: ENABLE_CLOUDFLARE=true but CF_TUNNEL_TOKEN unset"; return 0; }
  if pgrep -f "cloudflared.*tunnel.*run" >/dev/null; then
    echo "  cloudflared: already running"
    return
  fi
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "  cloudflared: installing"
    curl -sfL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
      -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
  fi
  nohup cloudflared tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" \
    > /workspace/logs/cloudflared.log 2>&1 &
  disown
  echo "  cloudflared PID=$! — log: /workspace/logs/cloudflared.log"
}

start_tailscale () {
  [ -n "$TAILSCALE_AUTHKEY" ] || { echo "  Tailscale: TAILSCALE_AUTHKEY not set — skipping"; return 0; }

  # Install if missing
  if ! command -v tailscale >/dev/null 2>&1; then
    echo "  Tailscale: installing..."
    curl -fsSL https://tailscale.com/install.sh | sh -s - >/dev/null 2>&1 || true
  fi

  # Kill any stale daemon before starting fresh — prevents ghost-pod where
  # tailscaled holds a stale state file from a previous pod registration.
  pkill -f tailscaled 2>/dev/null || true
  rm -f /tmp/tailscaled.sock
  sleep 2

  echo "  Tailscale: starting daemon (userspace-networking)..."
  /usr/sbin/tailscaled \
    --tun=userspace-networking \
    --state=/workspace/tailscaled.state \
    --socket=/tmp/tailscaled.sock \
    >> /workspace/logs/tailscale.log 2>&1 &
  disown
  sleep 5

  # --force-reauth ensures a fresh machine registration even if state file
  # was left over from a prior pod (one-time key consumed by previous pod).
  tailscale --socket=/tmp/tailscaled.sock up \
    --authkey="$TAILSCALE_AUTHKEY" \
    --hostname="$TAILSCALE_HOSTNAME" \
    --accept-routes \
    --accept-dns=false \
    --force-reauth \
    >> /workspace/logs/tailscale.log 2>&1 || {
      echo "  !! tailscale up failed — check /workspace/logs/tailscale.log"
      cat /workspace/logs/tailscale.log | tail -20
      return 1
    }

  local ts_ip
  ts_ip=$(tailscale --socket=/tmp/tailscaled.sock ip -4 2>/dev/null || echo "pending")
  echo "  Tailscale: connected — IP $ts_ip  hostname=$TAILSCALE_HOSTNAME"
  echo "  Tailscale log: /workspace/logs/tailscale.log"
}

start_jupyter () {
  if is_listening "8888"; then
    echo "  Jupyter: already listening on 8888 — leaving it alone"
    return 0
  fi
  echo "  Jupyter: installing..."
  # Use jupyter notebook — NOT jupyterlab (lab has a broken launcher on this image)
  $PY -m pip install jupyter --quiet --break-system-packages 2>/dev/null || true
  echo "  Jupyter: launching Jupyter Notebook on port 8888"
  nohup $PY -m jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --ServerApp.token='luma2026' \
    --ServerApp.allow_origin='*' \
    --ServerApp.root_dir=/workspace \
    --allow-root \
    > /workspace/logs/jupyter.log 2>&1 &
  disown
  echo "  Jupyter PID=$! — http://[tailscale-ip]:8888  token=luma2026  log: /workspace/logs/jupyter.log"
}

start_open_terminal () {
  if is_listening "$OPEN_TERMINAL_PORT"; then
    echo "  Open Terminal: already listening on $OPEN_TERMINAL_PORT — leaving it alone"
    return 0
  fi
  if [ -z "$OPEN_TERMINAL_API_KEY" ]; then
    echo "  Open Terminal: OPEN_TERMINAL_API_KEY not set — skipping"
    return 0
  fi
  echo "  Open Terminal: installing..."
  $PY -m pip install open-terminal --quiet --break-system-packages 2>/dev/null || \
    /workspace/app-venv/bin/pip install open-terminal --quiet 2>/dev/null || true
  local OT_BIN=""
  if command -v open-terminal >/dev/null 2>&1; then
    OT_BIN="open-terminal"
  elif [ -x /workspace/app-venv/bin/open-terminal ]; then
    OT_BIN="/workspace/app-venv/bin/open-terminal"
  else
    echo "  Open Terminal: binary not found after install — skipping"
    return 0
  fi
  echo "  Open Terminal: launching on port $OPEN_TERMINAL_PORT"
  nohup $OT_BIN run \
    --host 0.0.0.0 \
    --port "$OPEN_TERMINAL_PORT" \
    --api-key "$OPEN_TERMINAL_API_KEY" \
    > /workspace/logs/open_terminal.log 2>&1 &
  disown
  echo "  Open Terminal PID=$! — log: /workspace/logs/open_terminal.log"
  wait_for_health "http://localhost:$OPEN_TERMINAL_PORT/health" "OpenTerminal" 6 || true
}

ensure_readme
ensure_db

case "$RUN_MODE" in
  pod)
    start_tailscale
    start_jupyter
    start_vllm
    start_rag
    start_gateway
    start_webui
    start_open_terminal
    start_cloudflared
    ;;
  serverless)
    start_rag
    start_gateway
    start_cloudflared
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE (expected pod|serverless)"; exit 1 ;;
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
echo "   tail -f /workspace/logs/open_terminal.log"
echo "   tail -f /workspace/logs/openwebui.log"
echo ""

# --- Safe LUMA Tools Server (LUMA 6) ---
if [ -n "${LUMA_TOOLS_API_KEY:-}" ]; then
  LUMA_TOOLS_PORT="${LUMA_TOOLS_PORT:-8010}"
  LUMA_LOG_DIR="${LUMA_LOG_DIR:-/workspace/logs}"
  LUMA_PROJECT_DIR="${LUMA_PROJECT_DIR:-/workspace/luma}"
  LUMA_WORKLOG_DIR="${LUMA_WORKLOG_DIR:-/workspace/luma/worklog}"
  mkdir -p "$LUMA_LOG_DIR" "$LUMA_WORKLOG_DIR"
  echo "[start.sh] launching luma_tools_server on :$LUMA_TOOLS_PORT"
  nohup $PY /workspace/luma/luma_tools_server.py >>"$LUMA_LOG_DIR/luma_tools_server.log" 2>&1 &
else
  echo "[start.sh] LUMA_TOOLS_API_KEY not set - skipping luma_tools_server launch."
fi
# --- end LUMA Tools Server ---

# --- Kokoro TTS (LUMA 6) ---
if [ -n "${KOKORO_PORT:-}" ]; then
  echo "[start.sh] launching Kokoro TTS on :${KOKORO_PORT}"
  mkdir -p "${LUMA_LOG_DIR:-/workspace/logs}"
  nohup /workspace/luma/tts/start_kokoro.sh >>"${LUMA_LOG_DIR:-/workspace/logs}/kokoro.log" 2>&1 &
else
  echo "[start.sh] KOKORO_PORT not set - skipping Kokoro launch."
fi
# --- end Kokoro TTS ---

if [ "${KEEP_FOREGROUND:-true}" = "true" ]; then
  case "$RUN_MODE" in
    pod)        exec tail -f /workspace/logs/vllm.log /workspace/logs/gateway.log ;;
    serverless) exec tail -f /workspace/logs/rag.log  /workspace/logs/gateway.log ;;
  esac
fi
echo "KEEP_FOREGROUND=false — start.sh returning."
