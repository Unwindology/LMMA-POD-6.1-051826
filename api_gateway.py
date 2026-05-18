#!/usr/bin/env python3
"""
api_gateway.py — LUMA API Gateway v4.3
Handles OpenAI-compatible /v1/chat/completions for Open WebUI.

Changes from v4.2-current-state:
  - Streaming re-enabled: forwards body.stream as-is (was hardcoded False).
  - Context budget guard: RAG context capped at MAX_RAG_CHARS before injection.
    Prevents context overflow when long conversations + RAG + identity file
    exceed the model's max_model_len (was hitting 49153 > 49152 token limit).
  - max_tokens default: if caller doesn't set max_tokens, gateway injects
    MAX_TOKENS_DEFAULT so vLLM doesn't generate until context exhaustion.
  - Thinking capture: spool records reasoning_content alongside assistant reply
    so thinking tokens land in the fractal DB via luma_flusher.
  - All new tunables are env vars: MAX_CONTEXT_CHARS, MAX_RAG_CHARS,
    MAX_TOKENS_DEFAULT.
"""

import os
import json
import httpx
import threading
import time
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------
VLLM_BASE       = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
LUMA_RAG_URL    = os.environ.get("LUMA_RAG_URL",  "http://localhost:8001")
GATEWAY_PORT    = int(os.environ.get("GATEWAY_PORT", "8080"))
MODEL_NAME      = os.environ.get("MODEL_NAME", "luma")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", MODEL_NAME)
MODEL_ID             = os.environ.get("MODEL_ID", "")
MAX_MODEL_LEN        = os.environ.get("MAX_MODEL_LEN", "32768")
TENSOR_PARALLEL_SIZE = os.environ.get("TENSOR_PARALLEL_SIZE", "1")
PIPELINE_PARALLEL_SIZE = os.environ.get("PIPELINE_PARALLEL_SIZE", "1")
OPEN_TERMINAL_PORT   = os.environ.get("OPEN_TERMINAL_PORT", "8002")
CURRENT_STATE_PATH = os.environ.get("LUMA_CURRENT_STATE_PATH", "/workspace/luma/LUMA_CURRENT_STATE.md")

# Context budget — protect against overflow when conversation grows long.
# MAX_RAG_CHARS: hard cap on the RAG context block injected per request.
# MAX_CONTEXT_CHARS: informational total budget (not enforced here; RAG cap is the lever).
# MAX_TOKENS_DEFAULT: fallback max_tokens if the caller doesn't set one.
MAX_RAG_CHARS       = int(os.environ.get("MAX_RAG_CHARS", "16000"))
MAX_CONTEXT_CHARS   = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))
MAX_TOKENS_DEFAULT  = int(os.environ.get("MAX_TOKENS_DEFAULT", "2048"))


def load_current_state() -> str:
    try:
        if os.path.exists(CURRENT_STATE_PATH):
            with open(CURRENT_STATE_PATH, "r", encoding="utf-8") as f:
                text = f.read().strip()
            print(f"[GATEWAY] Loaded current-state file: {CURRENT_STATE_PATH} ({len(text)} chars)", flush=True)
            return text
        print(f"[GATEWAY] Current-state file not found: {CURRENT_STATE_PATH}", flush=True)
    except Exception as e:
        print(f"[GATEWAY] Failed to load current-state file: {e}", flush=True)
    return ""


LUMA_CURRENT_STATE = load_current_state()

# ---------------------------------------------------------------------------
# CONVERSATION SPOOL (for luma_flusher)
# ---------------------------------------------------------------------------
conversation_spool = []
spool_lock = threading.Lock()

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(title="LUMA Gateway", version="4.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GPU TELEMETRY HELPER
# ---------------------------------------------------------------------------
def _gpu_stats() -> dict:
    """
    Query nvidia-smi for per-GPU utilisation and VRAM.
    Returns aggregated totals across all GPUs.
    Falls back gracefully if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        ).decode().strip()
        rows = [line.strip().split(",") for line in out.splitlines() if line.strip()]
        if not rows:
            return {}
        util_vals, used_vals, total_vals = [], [], []
        for row in rows:
            if len(row) < 3:
                continue
            try:
                util_vals.append(float(row[0].strip()))
                used_vals.append(float(row[1].strip()))
                total_vals.append(float(row[2].strip()))
            except ValueError:
                pass
        if not util_vals:
            return {}
        return {
            "gpu_util":       round(sum(util_vals) / len(util_vals), 1),   # avg % across GPUs
            "gpu_count":      len(util_vals),
            "vram_used_gb":   round(sum(used_vals) / 1024, 2),             # MiB → GiB
            "vram_total_gb":  round(sum(total_vals) / 1024, 2),
            "vram_free_gb":   round((sum(total_vals) - sum(used_vals)) / 1024, 2),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CHUNK COUNT HELPER
# ---------------------------------------------------------------------------
def _rag_chunk_count() -> int | None:
    """Ask the RAG server how many chunks are indexed. Non-blocking best-effort."""
    try:
        resp = httpx.get(f"{LUMA_RAG_URL}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            # luma_server.py /health keys:
            return data.get("total_chunks") or data.get("chunk_count") or data.get("chunks")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# PING / HEALTH
# ---------------------------------------------------------------------------
@app.get("/ping")
@app.get("/health")
async def ping():
    gpu = _gpu_stats()
    total_chunks = _rag_chunk_count()
    payload = {
        "status": "ok",
        "service": "luma-gateway",
        "version": "4.5",
        "model_name": MODEL_NAME,
        "vllm_model_name": VLLM_MODEL_NAME,
        "vllm_base": VLLM_BASE,
        "current_state_loaded": bool(LUMA_CURRENT_STATE),
        "current_state_chars": len(LUMA_CURRENT_STATE),
        "max_rag_chars": MAX_RAG_CHARS,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "max_tokens_default": MAX_TOKENS_DEFAULT,
    }
    if gpu:
        payload.update(gpu)           # gpu_util, gpu_count, vram_used_gb, vram_total_gb, vram_free_gb
    if total_chunks is not None:
        payload["total_chunks"] = total_chunks
    return payload



# ---------------------------------------------------------------------------
# RUNTIME STATE HELPERS — authoritative runtime facts for LUMA
# ---------------------------------------------------------------------------
_RUNTIME_KEYWORDS = {
    "runtime", "model", "gpu", "vram", "port", "context", "max_model_len",
    "pod", "telemetry", "gateway", "vllm", "rag", "open webui", "terminal",
    "8000", "8001", "8080", "3000", "8002", "8888",
}


def _runtime_keywords_match(text: str) -> bool:
    """Return True if user text asks about live runtime/system facts."""
    lower = (text or "").lower()
    return any(kw in lower for kw in _RUNTIME_KEYWORDS)


def _runtime_state_data() -> dict:
    """Build authoritative runtime state from env + live GPU telemetry."""
    gpu = _gpu_stats()
    payload = {
        "service": "luma-runtime-state",
        "gateway_version": "4.5",
        "model": {
            "served_name": MODEL_NAME,
            "vllm_model_name": VLLM_MODEL_NAME,
            "model_id": MODEL_ID,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
        },
        "ports": {
            "8000": "raw vLLM — bypasses RAG, identity, capture, flusher, and gateway logic",
            "8001": "LUMA RAG retrieval",
            "8080": "LUMA gateway — use this for all normal LUMA/OpenAI API calls",
            "3000": "Open WebUI",
            "8002": "Open Terminal",
            "8888": "Jupyter/debug",
        },
        "urls": {
            "vllm_base_url": VLLM_BASE,
            "rag_url": LUMA_RAG_URL,
        },
        "runtime_locks": {
            "runtime_truth_rule": "Do not infer runtime facts from memory. Use VERIFIED_RUNTIME_STATE or /runtime_state; if absent, say unknown.",
            "openwebui_rule": "Use port 8080 for normal LUMA/OpenAI API calls. Do not use raw vLLM port 8000 except debugging.",
            "chgt_term_lock": "Canonical term: Clockwise Hair Growth Theory (CHGT). Forbidden: Counterclockwise Hair Growth Theory. Correct before answering. Do not rename CHGT based on release direction.",
        },
    }
    if gpu:
        payload["gpu"] = {
            "gpu_count": gpu.get("gpu_count"),
            "gpu_util": gpu.get("gpu_util"),
            "vram_used_gb": gpu.get("vram_used_gb"),
            "vram_total_gb": gpu.get("vram_total_gb"),
            "vram_free_gb": gpu.get("vram_free_gb"),
        }
    return payload


def _runtime_state_block() -> str:
    """Render runtime state as an authoritative system-prompt block."""
    return (
        "VERIFIED_RUNTIME_STATE:\n"
        + json.dumps(_runtime_state_data(), indent=2)
        + "\n\nRuntime rule:\n"
        "Use VERIFIED_RUNTIME_STATE for runtime facts. "
        "Do not infer runtime facts from memory. "
        "If VERIFIED_RUNTIME_STATE is absent, say unknown."
    )


@app.get("/runtime_state")
async def runtime_state():
    return _runtime_state_data()

# ---------------------------------------------------------------------------
# MODELS LIST — Open WebUI checks this on connect
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "luma",
            }
        ],
    }

# ---------------------------------------------------------------------------
# CHAT COMPLETIONS — main endpoint Open WebUI calls
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Honor the caller's stream preference — no longer forced off.
    stream = body.get("stream", False)

    # Pull user query for RAG
    user_query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_query = m.get("content", "")
            break

    print(f"[GATEWAY] chat request model={body.get('model')} stream={stream} -> forwarding as {VLLM_MODEL_NAME}; user_query={user_query[:160]!r}", flush=True)

    # --- RAG retrieval from luma_server ---
    rag_context = ""
    if user_query:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                rag_resp = await client.post(
                    f"{LUMA_RAG_URL}/retrieve",
                    json={"query": user_query, "top_k": 6},
                )
                print(f"[GATEWAY] RAG status={rag_resp.status_code} body={rag_resp.text[:300]!r}", flush=True)
                if rag_resp.status_code == 200:
                    chunks = rag_resp.json().get("chunks", [])
                    if chunks:
                        rag_context = "\n\n---\n\n".join(chunks)
        except Exception as e:
            print(f"[GATEWAY] RAG retrieval failed (non-fatal): {e}", flush=True)

    # --- Context budget guard ---
    # Cap RAG block so a long conversation thread + identity file can't overflow
    # the model's max_model_len. Identity file (~11KB) is not trimmed.
    if rag_context and len(rag_context) > MAX_RAG_CHARS:
        rag_context = rag_context[:MAX_RAG_CHARS] + "\n[RAG context truncated for token budget]"
        print(f"[GATEWAY] RAG context trimmed to {MAX_RAG_CHARS} chars", flush=True)

    # Inject current-state first, then RAG context.
    # --- Conditional runtime_state injection ---
    # Runtime facts must come from VERIFIED_RUNTIME_STATE, not model memory.
    runtime_block = ""
    if user_query and _runtime_keywords_match(user_query):
        runtime_block = _runtime_state_block()
        print("[GATEWAY] runtime_state injected", flush=True)

    control_blocks = []
    if LUMA_CURRENT_STATE:
        control_blocks.append("[LUMA CURRENT STATE — highest priority orientation]\n" + LUMA_CURRENT_STATE)
    if runtime_block:
        control_blocks.append(
            "[VERIFIED RUNTIME STATE — authoritative, overrides memory]\n" + runtime_block
        )
    if rag_context:
        control_blocks.append("[LUMA RETRIEVED CONTEXT]\n" + rag_context)

    if control_blocks:
        control_context = "\n\n---\n\n".join(control_blocks)
        system_injected = False
        for m in messages:
            if m.get("role") == "system":
                m["content"] = control_context + "\n\n---\n\n" + m.get("content", "")
                system_injected = True
                break
        if not system_injected:
            messages.insert(0, {
                "role": "system",
                "content": control_context,
            })

    # Inject max_tokens default if caller didn't set one — prevents vLLM from
    # running all the way to the context limit on every turn.
    if "max_tokens" not in body and MAX_TOKENS_DEFAULT > 0:
        body["max_tokens"] = MAX_TOKENS_DEFAULT

    # Forward to vLLM.
    payload = {**body, "messages": messages, "stream": stream, "model": VLLM_MODEL_NAME}

    if stream:
        # Streaming: pass through as SSE
        async def stream_generator():
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{VLLM_BASE}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                except Exception as e:
                    print(f"[GATEWAY] vLLM stream failed: {e}", flush=True)
                    yield b"data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming path
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            print(f"[GATEWAY] vLLM request failed: {e}", flush=True)
            return JSONResponse({"error": {"message": f"vLLM request failed: {e}", "type": "gateway_error"}}, status_code=502)

    print(f"[GATEWAY] vLLM status={resp.status_code}", flush=True)
    print(f"[GATEWAY] vLLM raw={resp.text[:1000]!r}", flush=True)

    try:
        result = resp.json()
    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"vLLM returned non-JSON: {e}", "raw": resp.text[:1000], "type": "gateway_error"}},
            status_code=502,
        )

    # If vLLM returned an error, pass it through with its status code.
    if resp.status_code >= 400:
        return JSONResponse(result, status_code=resp.status_code)

    # Spool the exchange — capture thinking tokens too if present
    try:
        choice = result["choices"][0]["message"]
        reply_text = choice.get("content", "")
        # vLLM returns thinking in reasoning_content when --enable-reasoning is active
        thinking_text = choice.get("reasoning_content", "")
        if user_query and reply_text:
            entry = {
                "timestamp": time.time(),
                "user": user_query,
                "assistant": reply_text,
            }
            if thinking_text:
                entry["thinking"] = thinking_text
            with spool_lock:
                conversation_spool.append(entry)
    except Exception as e:
        print(f"[GATEWAY] spool append skipped: {e}", flush=True)

    return JSONResponse(result)

# ---------------------------------------------------------------------------
# FLUSH endpoint (called by luma_flusher or manually)
# ---------------------------------------------------------------------------
@app.post("/flush")
async def flush():
    with spool_lock:
        count = len(conversation_spool)
    return {"status": "ok", "spooled": count}

# ---------------------------------------------------------------------------
# START luma_flusher background thread
# ---------------------------------------------------------------------------
try:
    from luma_flusher import start_flusher
    start_flusher(conversation_spool, spool_lock)
    print("[GATEWAY] luma_flusher started.", flush=True)
except Exception as e:
    print(f"[GATEWAY] luma_flusher not loaded (non-fatal): {e}", flush=True)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[GATEWAY] Starting on port {GATEWAY_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
