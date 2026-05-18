# LUMA POD 6.1 CLEAN — CUDA 12.8 / vLLM 0.19.1 / no Open WebUI / isolated Kokoro
# Repository target: Unwindology/LMMA-POD-6.1-051826
# Docker Hub target: unwindology/luma-pod-61-051826
#
# Design:
#   - Clean base: NVIDIA CUDA 12.8.1, not actor-gemma4, not old LUMA image.
#   - vLLM stack is installed explicitly:
#       Python 3.12
#       torch 2.10.0 CUDA 12.8
#       vLLM 0.19.1
#   - Open WebUI is never installed.
#   - LUMA/RAG/Gateway run on system Python.
#   - Kokoro TTS runs in its own venv: /workspace/luma/tts/kokoro-env.
#   - JupyterLab opens directly on :8888.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG IMAGE_BUILT_AT
ARG GIT_SHA

ENV DEBIAN_FRONTEND=noninteractive \
    LUMA_DIR=/workspace/luma \
    TMPDIR=/tmp \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    RAW_VLLM_PORT=8000 \
    LUMA_PORT=8001 \
    GATEWAY_PORT=8080 \
    LUMA_TOOLS_PORT=8010 \
    KOKORO_PORT=8880 \
    JUPYTER_PORT=8888 \
    KOKORO_MODEL_DIR=/workspace/luma/tts/kokoro \
    FRACTAL_DB=/workspace/luma/data/fractal_index.db \
    LUMA_CURRENT_STATE_PATH=/workspace/luma/LUMA_CURRENT_STATE.md \
    LUMA_README_PATH=/workspace/luma/LUMA_README.md

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /workspace/luma /workspace/luma/data /workspace/luma/tts /workspace/logs /tmp

# OS tools only. No Open WebUI. No actor-gemma4.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl procps iproute2 iputils-ping net-tools jq git \
      build-essential python3 python3-dev python3-pip python3-venv python-is-python3 \
      libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Pin torch to CUDA 12.8 before installing vLLM.
# This prevents the image from silently resolving to cu129/cu130.
RUN python3 -m pip install --pre --no-cache-dir \
      torch==2.10.0 torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/test/cu128

# Install vLLM after torch is already pinned to cu128.
# The extra index keeps dependency resolution aligned with CUDA 12.8.
RUN python3 -m pip install --no-cache-dir \
      "vllm==0.19.1" \
      --extra-index-url https://download.pytorch.org/whl/test/cu128

# LUMA Python dependencies. Keep this narrow; do not reinstall torch/vLLM.
RUN python3 -m pip install --no-cache-dir \
      fastapi uvicorn httpx pydantic \
      "huggingface_hub==1.14.0" \
      "sentence-transformers==5.5.0" \
      jupyterlab

# LUMA application files
COPY LUMA_README.md          /workspace/luma/LUMA_README.md
COPY LUMA_CURRENT_STATE.md   /workspace/luma/LUMA_CURRENT_STATE.md
COPY luma_server.py          /workspace/luma/luma_server.py
COPY api_gateway.py          /workspace/luma/api_gateway.py
COPY luma_flusher.py         /workspace/luma/luma_flusher.py
COPY luma_tools_server.py    /workspace/luma/luma_tools_server.py
COPY start.sh                /workspace/luma/start.sh

# Kokoro TTS files
COPY kokoro_server.py        /workspace/luma/tts/kokoro_server.py
COPY start_kokoro.sh         /workspace/luma/tts/start_kokoro.sh

RUN chmod +x /workspace/luma/start.sh /workspace/luma/tts/start_kokoro.sh

# Isolated Kokoro environment. This must never touch system Python/vLLM.
RUN python3 -m venv /workspace/luma/tts/kokoro-env && \
    /workspace/luma/tts/kokoro-env/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /workspace/luma/tts/kokoro-env/bin/pip install --no-cache-dir \
      "kokoro-onnx==0.3.4" fastapi uvicorn soundfile huggingface_hub

# Build guards: fail before push if the image violates the contract.
RUN python3 - <<'PY'
import importlib.util
import sys
import torch
import vllm
import fastapi, uvicorn, httpx
import sentence_transformers, huggingface_hub

print("OK: system Python", sys.executable)
print("OK: vLLM", getattr(vllm, "__version__", "unknown"))
print("OK: torch", torch.__version__, "cuda", torch.version.cuda)

if not str(torch.version.cuda).startswith("12.8"):
    raise SystemExit(f"FATAL: torch CUDA must be 12.8.x, got {torch.version.cuda}")

if importlib.util.find_spec("open_webui") is not None:
    raise SystemExit("FATAL: open_webui is installed; this image must not contain it")

print("OK: CUDA contract is 12.8.x")
print("OK: open_webui not installed")
PY

RUN /workspace/luma/tts/kokoro-env/bin/python - <<'PY'
import sys
import kokoro_onnx
import fastapi, uvicorn, soundfile
print("OK: Kokoro isolated Python", sys.executable)
print("OK: kokoro_onnx import")
PY

LABEL org.opencontainers.image.title="luma-pod-6.1-clean-cu128" \
      org.opencontainers.image.description="Clean LUMA 6.1 pod: CUDA 12.8 + vLLM 0.19.1 + LUMA Gateway/RAG + JupyterLab + Tailscale + isolated Kokoro TTS. No Open WebUI." \
      org.opencontainers.image.source="https://github.com/Unwindology/LMMA-POD-6.1-051826" \
      io.unwindology.luma.no_open_webui="true" \
      io.unwindology.luma.kokoro_isolated="true" \
      io.unwindology.luma.cuda_contract="12.8.x" \
      io.unwindology.luma.built_at="${IMAGE_BUILT_AT}" \
      io.unwindology.luma.git_sha="${GIT_SHA}"

CMD ["bash", "/workspace/luma/start.sh"]
