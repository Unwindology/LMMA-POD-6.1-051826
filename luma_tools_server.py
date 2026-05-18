#!/usr/bin/env python3
"""
luma_tools_server.py — Safe coworker tool server for LUMA 6.

Read-only access to logs and project files, plus a single append-only
worklog endpoint. Bearer auth required on every endpoint. No arbitrary
shell, no write paths outside the worklog dir, no access to secrets.

Env vars:
  LUMA_TOOLS_PORT      default 8010
  LUMA_TOOLS_API_KEY   required; if missing the server refuses to start
  LUMA_LOG_DIR         default /workspace/logs
  LUMA_PROJECT_DIR     default /workspace/luma
  LUMA_WORKLOG_DIR     default /workspace/luma/worklog
  GATEWAY_PORT         default 8080  (only used to build status URLs)
  VLLM_RAW_PORT        default 8000  (only used to build status URLs)
  LUMA_RAG_PORT        default 8001  (only used to build status URLs)

Endpoints:
  GET  /health
  GET  /runtime_status
  GET  /logs/list
  GET  /logs/read   ?name=&tail=
  GET  /logs/grep   ?query=&tail=&name=
  GET  /project/read ?path=
  GET  /project/grep ?query=&path=&max_results=
  POST /worklog/append   body: {"text": "...", "tag": "..."}
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERSION = "luma-tools 0.1.0"

LUMA_TOOLS_PORT    = int(os.environ.get("LUMA_TOOLS_PORT", "8010"))
LUMA_TOOLS_API_KEY = os.environ.get("LUMA_TOOLS_API_KEY", "").strip()
LUMA_LOG_DIR       = Path(os.environ.get("LUMA_LOG_DIR", "/workspace/logs")).resolve()
LUMA_PROJECT_DIR   = Path(os.environ.get("LUMA_PROJECT_DIR", "/workspace/luma")).resolve()
LUMA_WORKLOG_DIR   = Path(os.environ.get("LUMA_WORKLOG_DIR", "/workspace/luma/worklog")).resolve()

GATEWAY_PORT  = int(os.environ.get("GATEWAY_PORT", "8080"))
VLLM_RAW_PORT = int(os.environ.get("VLLM_RAW_PORT", "8000"))
LUMA_RAG_PORT = int(os.environ.get("LUMA_RAG_PORT", "8001"))

# Hard caps.
MAX_TAIL_LINES      = 2000
MAX_READ_BYTES      = 1 * 1024 * 1024     # 1 MiB
MAX_GREP_MATCHES    = 500
MAX_WORKLOG_CHARS   = 16 * 1024
MAX_QUERY_LEN       = 512
HTTP_TIMEOUT        = 3.0

# Sensitive filename patterns. If the resolved path matches any of these,
# the endpoint returns 403 — regardless of whether the path is otherwise
# inside an allowed root.
SECRET_NAME_PATTERNS = [
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"(^|/)\.git(/|$)"),
    re.compile(r"(^|/)\.ssh(/|$)"),
    re.compile(r"(^|/)id_rsa($|\.)"),
    re.compile(r"(^|/)id_ed25519($|\.)"),
    re.compile(r"(^|/)\.netrc$"),
    re.compile(r"(^|/)\.npmrc$"),
    re.compile(r"(^|/)\.pypirc$"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
    re.compile(r"\.p12$"),
    re.compile(r"\.pfx$"),
    re.compile(r"(token|secret|password|api[_-]?key|credentials)", re.IGNORECASE),
]

# Binary-ish extensions we refuse to read or grep.
BINARY_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".bin", ".so", ".dylib", ".dll",
    ".pyc", ".pyo", ".whl", ".tar", ".gz", ".zip", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ico",
    ".mp3", ".mp4", ".wav", ".ogg", ".mov", ".avi",
    ".pt", ".bin",
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def require_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    if not LUMA_TOOLS_API_KEY:
        # Server should not be running without a key, but guard anyway.
        raise HTTPException(status_code=503, detail="server not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != LUMA_TOOLS_API_KEY:
        raise HTTPException(status_code=401, detail="invalid token")


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def _is_sensitive(path: Path) -> bool:
    s = str(path)
    return any(p.search(s) for p in SECRET_NAME_PATTERNS)


def _safe_join(root: Path, rel: str) -> Path:
    """
    Resolve `rel` against `root`, then verify the resolved path is still
    inside `root`. Raises HTTPException(400/403) on failure.
    """
    if not rel or not isinstance(rel, str):
        raise HTTPException(status_code=400, detail="path is required")
    if "\x00" in rel:
        raise HTTPException(status_code=400, detail="invalid path")
    # Reject absolute paths and obvious traversal up front.
    if rel.startswith("/") or rel.startswith("\\"):
        raise HTTPException(status_code=400, detail="absolute paths not allowed")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="path escapes root")
    if _is_sensitive(candidate):
        raise HTTPException(status_code=403, detail="path is in secrets blocklist")
    return candidate


def _safe_log_name(name: str) -> Path:
    """
    Logs are referenced by filename only. No subdirectories. The file must
    exist directly inside LUMA_LOG_DIR.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid log name")
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="log name too long")
    candidate = (LUMA_LOG_DIR / name).resolve()
    try:
        candidate.relative_to(LUMA_LOG_DIR)
    except ValueError:
        raise HTTPException(status_code=403, detail="log path escapes root")
    if _is_sensitive(candidate):
        raise HTTPException(status_code=403, detail="log name in blocklist")
    return candidate


def _tail_lines(path: Path, n: int) -> List[str]:
    """Read the last `n` lines without slurping the whole file when possible."""
    n = max(1, min(int(n), MAX_TAIL_LINES))
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file not found")
    # Heuristic: average 200 bytes/line, read 1.5x to be safe, cap at file size.
    approx = min(size, max(8192, n * 300))
    with path.open("rb") as f:
        if size > approx:
            f.seek(-approx, os.SEEK_END)
            f.readline()  # drop partial first line
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-n:]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="LUMA Tools Server", version=VERSION)


@app.get("/health")
def health(_: None = Depends(require_bearer)) -> Dict[str, Any]:
    return {
        "ok": True,
        "version": VERSION,
        "port": LUMA_TOOLS_PORT,
        "ts": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "logs": str(LUMA_LOG_DIR),
            "project": str(LUMA_PROJECT_DIR),
            "worklog": str(LUMA_WORKLOG_DIR),
        },
    }


def _http_get_json(url: str) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as c:
            r = c.get(url)
        out: Dict[str, Any] = {"status": r.status_code}
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                out["json"] = r.json()
            except Exception as e:
                out["error"] = f"json decode: {e}"
                out["body"] = r.text[:512]
        else:
            out["body"] = r.text[:512]
        return out
    except Exception as e:
        return {"status": 0, "error": str(e)}


def _nvidia_smi_summary() -> Dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [exe,
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=False, shell=False,
        )
    except Exception as e:
        return {"available": False, "reason": f"exec error: {e}"}
    if proc.returncode != 0:
        return {"available": False, "reason": f"exit {proc.returncode}", "stderr": proc.stderr[:400]}
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "mem_used_mib": int(parts[2]),
                "mem_total_mib": int(parts[3]),
                "util_pct": int(parts[4]),
                "temp_c": int(parts[5]),
            })
        except ValueError:
            continue
    return {"available": True, "count": len(rows), "gpus": rows}


@app.get("/runtime_status")
def runtime_status(_: None = Depends(require_bearer)) -> Dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gateway_runtime_state": _http_get_json(f"http://localhost:{GATEWAY_PORT}/runtime_state"),
        "gateway_health":        _http_get_json(f"http://localhost:{GATEWAY_PORT}/health"),
        "vllm_models":           _http_get_json(f"http://localhost:{VLLM_RAW_PORT}/v1/models"),
        "rag_health":            _http_get_json(f"http://localhost:{LUMA_RAG_PORT}/health"),
        "nvidia_smi":            _nvidia_smi_summary(),
    }


@app.get("/logs/list")
def logs_list(_: None = Depends(require_bearer)) -> Dict[str, Any]:
    if not LUMA_LOG_DIR.exists():
        return {"root": str(LUMA_LOG_DIR), "exists": False, "files": []}
    files: List[Dict[str, Any]] = []
    for entry in sorted(LUMA_LOG_DIR.iterdir()):
        if not entry.is_file():
            continue
        if _is_sensitive(entry):
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        files.append({
            "name": entry.name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"root": str(LUMA_LOG_DIR), "exists": True, "files": files}


@app.get("/logs/read")
def logs_read(
    name: str = Query(..., description="log filename inside LUMA_LOG_DIR"),
    tail: int = Query(100, ge=1, le=MAX_TAIL_LINES),
    _: None = Depends(require_bearer),
) -> Dict[str, Any]:
    p = _safe_log_name(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="log not found")
    lines = _tail_lines(p, tail)
    return {"name": name, "tail": len(lines), "lines": lines}


@app.get("/logs/grep")
def logs_grep(
    query: str = Query(..., min_length=1, max_length=MAX_QUERY_LEN),
    tail: int = Query(200, ge=1, le=MAX_TAIL_LINES),
    name: Optional[str] = Query(None, description="restrict to one log file"),
    _: None = Depends(require_bearer),
) -> Dict[str, Any]:
    try:
        pat = re.compile(query)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"bad regex: {e}")
    targets: List[Path] = []
    if name:
        targets.append(_safe_log_name(name))
    else:
        if LUMA_LOG_DIR.exists():
            for entry in sorted(LUMA_LOG_DIR.iterdir()):
                if entry.is_file() and not _is_sensitive(entry):
                    targets.append(entry)
    results: List[Dict[str, Any]] = []
    total = 0
    for t in targets:
        if total >= MAX_GREP_MATCHES:
            break
        try:
            lines = _tail_lines(t, tail)
        except HTTPException:
            continue
        for i, line in enumerate(lines):
            if pat.search(line):
                results.append({"file": t.name, "lineno_in_tail": i + 1, "line": line})
                total += 1
                if total >= MAX_GREP_MATCHES:
                    break
    return {"query": query, "files_scanned": [t.name for t in targets], "matches": results, "truncated": total >= MAX_GREP_MATCHES}


@app.get("/project/read")
def project_read(
    path: str = Query(..., description="relative path inside LUMA_PROJECT_DIR"),
    _: None = Depends(require_bearer),
) -> Dict[str, Any]:
    p = _safe_join(LUMA_PROJECT_DIR, path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="file not found")
    if not p.is_file():
        raise HTTPException(status_code=400, detail="not a regular file")
    if p.suffix.lower() in BINARY_SUFFIXES:
        raise HTTPException(status_code=415, detail="binary file not readable")
    st = p.stat()
    if st.st_size > MAX_READ_BYTES:
        raise HTTPException(status_code=413, detail=f"file too large (>{MAX_READ_BYTES} bytes)")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read error: {e}")
    return {
        "path": path,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "content": content,
    }


@app.get("/project/grep")
def project_grep(
    query: str = Query(..., min_length=1, max_length=MAX_QUERY_LEN),
    path: str = Query(".", description="subdir of LUMA_PROJECT_DIR to search"),
    max_results: int = Query(200, ge=1, le=MAX_GREP_MATCHES),
    _: None = Depends(require_bearer),
) -> Dict[str, Any]:
    try:
        pat = re.compile(query)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"bad regex: {e}")
    root = _safe_join(LUMA_PROJECT_DIR, path) if path not in ("", ".", "./") else LUMA_PROJECT_DIR
    if not root.exists() or not root.is_dir():
        raise HTTPException(status_code=404, detail="path not a directory")
    matches: List[Dict[str, Any]] = []
    for f in root.rglob("*"):
        if len(matches) >= max_results:
            break
        if not f.is_file():
            continue
        if f.suffix.lower() in BINARY_SUFFIXES:
            continue
        if _is_sensitive(f):
            continue
        try:
            if f.stat().st_size > MAX_READ_BYTES:
                continue
        except OSError:
            continue
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if pat.search(line):
                        rel = str(f.relative_to(LUMA_PROJECT_DIR))
                        matches.append({"path": rel, "lineno": lineno, "line": line.rstrip("\n")[:400]})
                        if len(matches) >= max_results:
                            break
        except OSError:
            continue
    return {"query": query, "root": str(root.relative_to(LUMA_PROJECT_DIR)) or ".", "matches": matches, "truncated": len(matches) >= max_results}


@app.post("/worklog/append")
async def worklog_append(request: Request, _: None = Depends(require_bearer)) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    text = body.get("text")
    tag  = body.get("tag", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="`text` is required")
    if len(text) > MAX_WORKLOG_CHARS:
        raise HTTPException(status_code=413, detail=f"text exceeds {MAX_WORKLOG_CHARS} chars")
    if not isinstance(tag, str) or len(tag) > 64:
        raise HTTPException(status_code=400, detail="`tag` must be a string <= 64 chars")
    tag_clean = re.sub(r"[^A-Za-z0-9_.-]", "_", tag).strip("_")[:64]

    LUMA_WORKLOG_DIR.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir in case of symlink racing.
    wl_root = LUMA_WORKLOG_DIR.resolve()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = (wl_root / f"worklog-{today}.md").resolve()
    try:
        target.relative_to(wl_root)
    except ValueError:
        raise HTTPException(status_code=500, detail="worklog path escapes root")

    ts = datetime.now(timezone.utc).isoformat()
    header = f"\n### {ts}" + (f" [{tag_clean}]" if tag_clean else "") + "\n\n"
    entry  = header + text.rstrip() + "\n"
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"write error: {e}")
    return {"ok": True, "file": str(target.relative_to(wl_root)), "bytes_written": len(entry.encode("utf-8")), "ts": ts}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    if not LUMA_TOOLS_API_KEY:
        raise SystemExit("LUMA_TOOLS_API_KEY is not set — refusing to start.")
    print(f"[luma_tools] starting on :{LUMA_TOOLS_PORT}")
    print(f"[luma_tools]   logs root    = {LUMA_LOG_DIR}")
    print(f"[luma_tools]   project root = {LUMA_PROJECT_DIR}")
    print(f"[luma_tools]   worklog root = {LUMA_WORKLOG_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=LUMA_TOOLS_PORT, log_level="info")


if __name__ == "__main__":
    main()
