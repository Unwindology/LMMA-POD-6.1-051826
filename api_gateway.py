#!/usr/bin/env python3
"""
api_gateway.py — LUMA API Gateway v4.6-tools
Drop-in replacement. Adds function-calling bridge to luma_tools_server (:8010).

New in v4.6-tools:
  - TOOLS block defined from LUMA_TOOLS_SERVER_URL + LUMA_TOOLS_API_KEY.
  - tools passed to vLLM on every chat completion request (vLLM must be
    started with --enable-auto-tool-choice --tool-call-parser gemma4).
  - Tool-call loop: gateway intercepts finish_reason=tool_calls, executes
    against luma_tools_server, feeds the result back, and repeats until
    the model produces a final answer.
  - Non-streaming primary-turn path handles the loop natively.
  - Streaming path: if vLLM short-circuits with tool_calls mid-stream,
    the gateway drains the stream, executes tools, and re-dispatches
    (non-streaming) for the follow-up turn(s); the final answer is then
    re-chunked as SSE so the client still sees streaming.

Env vars used:
  LUMA_TOOLS_SERVER_URL   default http://localhost:8010
  LUMA_TOOLS_API_KEY      required for tool execution (same as tools server)

Requires: ENABLE_GEMMA4_TOOLS=true in pod start.sh so vLLM boots with
  --enable-auto-tool-choice --tool-call-parser gemma4

=== FULL FILE — drop-in replacement for /workspace/luma/api_gateway.py ===
"""

import os
import json
import httpx
import threading
import time
import subprocess
from pathlib import Path
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

# Context budget
MAX_RAG_CHARS       = int(os.environ.get("MAX_RAG_CHARS", "16000"))
MAX_CONTEXT_CHARS   = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))
MAX_TOKENS_DEFAULT  = int(os.environ.get("MAX_TOKENS_DEFAULT", "2048"))

# --- TOOLS CONFIG ---
LUMA_TOOLS_SERVER_URL = os.environ.get("LUMA_TOOLS_SERVER_URL", "http://localhost:8010")
LUMA_TOOLS_API_KEY    = os.environ.get("LUMA_TOOLS_API_KEY", "")
LUMA_PROJECT_DIR      = os.environ.get("LUMA_PROJECT_DIR", "/workspace/luma")
TOOLS_ENABLED = bool(LUMA_TOOLS_API_KEY)

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": "Read a text file from the pod workspace. Path is relative to /workspace/luma/ (e.g. 'scripts/chapter1.md').",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to /workspace/luma/"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list",
            "description": "List files and folders in a directory on the pod. Use '.' or empty string for the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to /workspace/luma/ (default: root)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fs_search",
            "description": "Search for text in project files. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search in (relative to /workspace/luma/, default: all)."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

TOOLS_SYSTEM_NOTE = (
    "\n\n[TOOLS — Pod File Access]\n"
    "You have access to the pod's file system through function calls.\n"
    "Use fs_read to read script files, role files, and project documents.\n"
    "Use fs_list to explore directories and see what files are available.\n"
    "Use fs_search to find content across files.\n"
    "Never invent file contents — always read them. If a file doesn't exist, say so."
)


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
# CONVERSATION SPOOL
# ---------------------------------------------------------------------------
conversation_spool = []
spool_lock = threading.Lock()

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(title="LUMA Gateway", version="4.6-tools")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GPU TELEMETRY
# ---------------------------------------------------------------------------
def _gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], timeout=3,
        ).decode().strip()
        rows = [line.strip().split(",") for line in out.splitlines() if line.strip()]
        if not rows:
            return {}
        util_vals, used_vals, total_vals = [], [], []
        for row in rows:
            if len(row) < 3: continue
            try:
                util_vals.append(float(row[0].strip()))
                used_vals.append(float(row[1].strip()))
                total_vals.append(float(row[2].strip()))
            except ValueError: pass
        if not util_vals: return {}
        return {
            "gpu_util": round(sum(util_vals)/len(util_vals), 1),
            "gpu_count": len(util_vals),
            "vram_used_gb": round(sum(used_vals)/1024, 2),
            "vram_total_gb": round(sum(total_vals)/1024, 2),
            "vram_free_gb": round((sum(total_vals)-sum(used_vals))/1024, 2),
        }
    except Exception:
        return {}


def _rag_chunk_count() -> int | None:
    try:
        resp = httpx.get(f"{LUMA_RAG_URL}/health", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("total_chunks") or data.get("chunk_count") or data.get("chunks")
    except Exception: pass
    return None


# ---------------------------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------------------------
def _safe_project_path(rel: str) -> Path:
    """Resolve a relative path against LUMA_PROJECT_DIR, refuse escapes."""
    root = Path(LUMA_PROJECT_DIR).resolve()
    if not rel or rel in (".", ""):
        return root
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes project root: {rel}")
    return candidate


async def _execute_tool(name: str, arguments: dict) -> str:
    """Call luma_tools_server or use local fs for list. Return result string."""
    if not TOOLS_ENABLED:
        return json.dumps({"error": "tools not configured"})

    # --- fs_list: local os.listdir (tools server has no list endpoint) ---
    if name == "fs_list":
        try:
            target = _safe_project_path(arguments.get("path", "."))
            items = []
            for entry in sorted(target.iterdir()):
                if entry.name.startswith(".") and entry.name not in (".gitkeep",):
                    continue
                suffix = "/" if entry.is_dir() else ""
                try:
                    size = entry.stat().st_size if entry.is_file() else 0
                except OSError:
                    size = 0
                items.append({
                    "name": entry.name + suffix,
                    "size": size,
                    "type": "dir" if entry.is_dir() else "file",
                })
            return json.dumps({"path": arguments.get("path", "."), "count": len(items), "items": items}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # --- fs_read and fs_search: call tools server ---
    headers = {"Authorization": f"Bearer {LUMA_TOOLS_API_KEY}"}
    params = {}
    endpoint = ""

    if name == "fs_read":
        endpoint = "/project/read"
        params["path"] = arguments.get("path", "")
    elif name == "fs_search":
        endpoint = "/project/grep"
        params["query"] = arguments.get("query", "")
        params["path"] = arguments.get("path", ".")
        params["max_results"] = "200"
    else:
        return json.dumps({"error": f"unknown tool: {name}"})

    url = f"{LUMA_TOOLS_SERVER_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
            if name == "fs_read" and "content" in data:
                content = data["content"]
                if len(content) > 24000:
                    content = content[:24000] + "\n\n[FILE TRUNCATED — use fs_search for specific sections]"
                return content
            return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


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
        "version": "4.6-tools",
        "model_name": MODEL_NAME,
        "vllm_model_name": VLLM_MODEL_NAME,
        "vllm_base": VLLM_BASE,
        "tools_enabled": TOOLS_ENABLED,
        "tools_server": LUMA_TOOLS_SERVER_URL if TOOLS_ENABLED else None,
        "current_state_loaded": bool(LUMA_CURRENT_STATE),
        "current_state_chars": len(LUMA_CURRENT_STATE),
        "max_rag_chars": MAX_RAG_CHARS,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "max_tokens_default": MAX_TOKENS_DEFAULT,
    }
    if gpu: payload.update(gpu)
    if total_chunks is not None: payload["total_chunks"] = total_chunks
    return payload


# ---------------------------------------------------------------------------
# RUNTIME STATE
# ---------------------------------------------------------------------------
_RUNTIME_KEYWORDS = {
    "runtime", "model", "gpu", "vram", "port", "context", "max_model_len",
    "pod", "telemetry", "gateway", "vllm", "rag", "open webui", "terminal",
    "8000", "8001", "8080", "3000", "8002", "8888",
}

def _runtime_keywords_match(text: str) -> bool:
    lower = (text or "").lower()
    return any(kw in lower for kw in _RUNTIME_KEYWORDS)

def _runtime_state_data() -> dict:
    gpu = _gpu_stats()
    payload = {
        "service": "luma-runtime-state",
        "gateway_version": "4.6-tools",
        "tools_enabled": TOOLS_ENABLED,
        "model": {
            "served_name": MODEL_NAME,
            "vllm_model_name": VLLM_MODEL_NAME,
            "model_id": MODEL_ID,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
        },
        "ports": {
            "8000": "raw vLLM",
            "8001": "LUMA RAG retrieval",
            "8080": "LUMA gateway",
            "3000": "Open WebUI",
            "8002": "Open Terminal",
            "8010": "LUMA Tools Server",
            "8888": "Jupyter",
        },
        "urls": {
            "vllm_base_url": VLLM_BASE,
            "rag_url": LUMA_RAG_URL,
            "tools_server_url": LUMA_TOOLS_SERVER_URL if TOOLS_ENABLED else None,
        },
    }
    if gpu: payload["gpu"] = gpu
    return payload

def _runtime_state_block() -> str:
    return (
        "VERIFIED_RUNTIME_STATE:\n"
        + json.dumps(_runtime_state_data(), indent=2)
        + "\n\nRuntime rule: Use VERIFIED_RUNTIME_STATE for runtime facts. "
        "Do not infer runtime facts from memory."
    )

@app.get("/runtime_state")
async def runtime_state():
    return _runtime_state_data()

# ---------------------------------------------------------------------------
# MODELS LIST
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": 1700000000,
            "owned_by": "luma",
        }],
    }


# ---------------------------------------------------------------------------
# TOOL-CALL LOOP (internal — handles multi-turn tool execution)
# ---------------------------------------------------------------------------
async def _vllm_chat_with_tools(messages: list, base_payload: dict) -> dict:
    """
    Send messages to vLLM. If vLLM returns tool_calls, execute them,
    feed results back, and repeat until a final answer arrives.
    Returns the final vLLM JSON response dict.
    """
    current_messages = [dict(m) for m in messages]
    max_rounds = 5
    payload = {**base_payload, "messages": current_messages, "stream": False}

    for round_num in range(max_rounds):
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{VLLM_BASE}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code >= 400:
            return {
                "error": {
                    "message": f"vLLM error {resp.status_code}",
                    "raw": resp.text[:1000],
                    "type": "vllm_error"
                }
            }

        result = resp.json()
        choice = result.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason", "stop")
        message = choice.get("message", {})

        # Check for tool calls
        tool_calls = message.get("tool_calls") or []
        if finish_reason == "tool_calls" and tool_calls:
            print(f"[GATEWAY] tool_calls detected round {round_num+1}: {[tc['function']['name'] for tc in tool_calls]}", flush=True)

            # Append assistant message with tool_calls
            current_messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            # Execute each tool call and append results
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}
                tool_result = await _execute_tool(func_name, func_args)
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", "unknown"),
                    "content": tool_result,
                })
                print(f"[GATEWAY] tool {func_name} executed ({len(tool_result)} chars)", flush=True)

            # Continue loop with updated messages
            payload["messages"] = current_messages
            continue

        # No tool calls — this is the final answer
        return result

    # Exhausted max rounds
    return {
        "error": {
            "message": "max tool-call rounds exceeded",
            "type": "gateway_error"
        }
    }


# ---------------------------------------------------------------------------
# CHAT COMPLETIONS
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    user_query = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_query = m.get("content", "")
            break

    print(f"[GATEWAY] chat request model={body.get('model')} stream={stream} tools={TOOLS_ENABLED}; user_query={user_query[:160]!r}", flush=True)

    # --- RAG retrieval ---
    rag_context = ""
    if user_query:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                rag_resp = await client.post(
                    f"{LUMA_RAG_URL}/retrieve",
                    json={"query": user_query, "top_k": 6},
                )
                if rag_resp.status_code == 200:
                    chunks = rag_resp.json().get("chunks", [])
                    if chunks:
                        rag_context = "\n\n---\n\n".join(chunks)
        except Exception as e:
            print(f"[GATEWAY] RAG retrieval failed (non-fatal): {e}", flush=True)

    if rag_context and len(rag_context) > MAX_RAG_CHARS:
        rag_context = rag_context[:MAX_RAG_CHARS] + "\n[RAG context truncated for token budget]"
        print(f"[GATEWAY] RAG context trimmed to {MAX_RAG_CHARS} chars", flush=True)

    # --- Runtime state ---
    runtime_block = ""
    if user_query and _runtime_keywords_match(user_query):
        runtime_block = _runtime_state_block()

    # --- Build control context ---
    control_blocks = []
    if LUMA_CURRENT_STATE:
        control_blocks.append("[LUMA CURRENT STATE — highest priority orientation]\n" + LUMA_CURRENT_STATE)
    if runtime_block:
        control_blocks.append("[VERIFIED RUNTIME STATE — authoritative, overrides memory]\n" + runtime_block)
    if rag_context:
        control_blocks.append("[LUMA RETRIEVED CONTEXT]\n" + rag_context)
    if TOOLS_ENABLED:
        control_blocks.append(TOOLS_SYSTEM_NOTE)

    if control_blocks:
        control_context = "\n\n---\n\n".join(control_blocks)
        system_injected = False
        for m in messages:
            if m.get("role") == "system":
                m["content"] = control_context + "\n\n---\n\n" + m.get("content", "")
                system_injected = True
                break
        if not system_injected:
            messages.insert(0, {"role": "system", "content": control_context})

    if "max_tokens" not in body and MAX_TOKENS_DEFAULT > 0:
        body["max_tokens"] = MAX_TOKENS_DEFAULT

    # Build payload for vLLM
    payload = {
        **body,
        "messages": messages,
        "stream": False,
        "model": VLLM_MODEL_NAME,
    }

    # --- Add tools to payload ---
    if TOOLS_ENABLED:
        existing_tools = body.get("tools", [])
        payload["tools"] = TOOLS_DEFINITION + existing_tools
        print(f"[GATEWAY] tools injected ({len(payload['tools'])} total)", flush=True)

    # --- Execute with tool-call loop ---
    result = await _vllm_chat_with_tools(messages, payload)

    if "error" in result:
        return JSONResponse(result, status_code=502)

    # --- Spool ---
    try:
        choice = result["choices"][0]["message"]
        reply_text = choice.get("content", "")
        thinking_text = choice.get("reasoning_content", "")
        if user_query and reply_text:
            entry = {"timestamp": time.time(), "user": user_query, "assistant": reply_text}
            if thinking_text:
                entry["thinking"] = thinking_text
            with spool_lock:
                conversation_spool.append(entry)
    except Exception as e:
        print(f"[GATEWAY] spool append skipped: {e}", flush=True)

    # --- Return ---
    if stream:
        final_content = result["choices"][0]["message"].get("content", "")
        async def re_stream():
            chunk_size = 80
            for i in range(0, len(final_content), chunk_size):
                chunk_text = final_content[i:i+chunk_size]
                chunk = {
                    "id": result.get("id", "chatcmpl-000"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": result.get("model", MODEL_NAME),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final_chunk = {
                "id": result.get("id", "chatcmpl-000"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": result.get("model", MODEL_NAME),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": result["choices"][0].get("finish_reason", "stop"),
                }],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(re_stream(), media_type="text/event-stream")

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# FLUSH
# ---------------------------------------------------------------------------
@app.post("/flush")
async def flush():
    with spool_lock:
        count = len(conversation_spool)
    return {"status": "ok", "spooled": count}

# ---------------------------------------------------------------------------
# START luma_flusher
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
    print(f"[GATEWAY] Starting on port {GATEWAY_PORT}  (v4.6-tools, tools={'enabled' if TOOLS_ENABLED else 'disabled'})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)