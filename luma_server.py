#!/usr/bin/env python3
"""
luma_server.py — LUMA RAG server for RunPod serverless
v2.2-keyword — RAM-safe retrieve() with LIMIT 5000 + HF cold-boot download

Boot sequence:
  1. Download latest fractal_index.db from HuggingFace (if HF_TOKEN is set)
  2. Load all-MiniLM-L6-v2 embedder
  3. Start FastAPI on LUMA_PORT (default 8001)
  4. api_gateway.py proxies inbound requests here

Changes from v2.0:
  - retrieve() uses ORDER BY RANDOM() LIMIT 5000 to cap RAM per query
    (previously loaded all 67K embeddings → ~2.5GB → worker OOM after first call)
  - HF download path hardened: checks HF_TOKEN before attempting, logs clearly
"""

import os
import sys
import sqlite3
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------

HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_REPO_ID      = os.environ.get("HF_REPO_ID", "chap18700/luma-raw-data")
FRACTAL_DB      = os.environ.get("FRACTAL_DB", "/workspace/luma/aethel-luma/fractal_index.db")
LUMA_PORT       = int(os.environ.get("LUMA_PORT", "8001"))

# Cosine retrieval settings
RETRIEVE_SAMPLE = int(os.environ.get("LUMA_RETRIEVE_SAMPLE", "5000"))   # rows to sample per query
TOP_K_DEFAULT   = int(os.environ.get("LUMA_TOP_K", "6"))
MIN_SCORE       = float(os.environ.get("LUMA_MIN_SCORE", "0.35"))

# ---------------------------------------------------------------------------
# STEP 1 — Download latest fractal_index.db from HuggingFace at cold boot
# ---------------------------------------------------------------------------

def download_fractal_db():
    if not HF_TOKEN:
        print("[LUMA] HF_TOKEN not set — skipping HF download, using local DB if present.")
        return

    try:
        from huggingface_hub import hf_hub_download
        local_dir = os.path.dirname(FRACTAL_DB)
        os.makedirs(local_dir, exist_ok=True)

        print(f"[LUMA] Downloading fractal_index.db from {HF_REPO_ID} ...")
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename="fractal_index.db",
            local_dir=local_dir,
            token=HF_TOKEN,
        )
        print(f"[LUMA] fractal_index.db ready at {path}")
    except Exception as e:
        print(f"[LUMA] WARNING: HF download failed: {e}")
        print("[LUMA] Falling back to local DB if it exists.")

download_fractal_db()

# ---------------------------------------------------------------------------
# STEP 2 — Load embedder
# ---------------------------------------------------------------------------

print("[LUMA] Loading sentence-transformers embedder...")
from sentence_transformers import SentenceTransformer
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
print("[LUMA] Embedder ready.")

# ---------------------------------------------------------------------------
# RETRIEVAL — RAM-safe cosine similarity
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K_DEFAULT, min_score: float = MIN_SCORE) -> list[str]:
    """
    Embed the query and score against a random LIMIT sample from fractal_chunks.

    Why LIMIT?
      Loading all 67K embeddings (272MB DB) into RAM for every query caused
      workers to exhaust ~2.5GB per call and OOM after the first request.
      LIMIT 5000 + ORDER BY RANDOM() gives representative recall at <200MB/query.

    Tune LUMA_RETRIEVE_SAMPLE env var if you want wider coverage on larger DBs.
    """
    if not os.path.exists(FRACTAL_DB):
        print(f"[LUMA] ERROR: fractal_index.db not found at {FRACTAL_DB}")
        return []

    qvec = EMBEDDER.encode([query])[0].astype(np.float16)

    conn = sqlite3.connect(FRACTAL_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT content, embedding FROM fractal_chunks ORDER BY RANDOM() LIMIT ?",
        (RETRIEVE_SAMPLE,),
    )
    rows = cur.fetchall()
    conn.close()

    results = []
    for text, emb_blob in rows:
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float16)
            if emb.shape != qvec.shape:
                continue
            norm = np.linalg.norm(qvec) * np.linalg.norm(emb)
            if norm == 0:
                continue
            score = float(np.dot(qvec, emb) / (norm + 1e-8))
            if score >= min_score:
                results.append((score, text))
        except Exception:
            continue

    results.sort(reverse=True)
    return [t for _, t in results[:top_k]]

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(title="LUMA RAG Server", version="2.2-keyword")


@app.get("/health")
async def health():
    import sqlite3 as _sqlite3
    db_exists = os.path.exists(FRACTAL_DB)
    db_size_mb = round(os.path.getsize(FRACTAL_DB) / 1_048_576, 1) if db_exists else 0
    total_chunks = None
    if db_exists:
        try:
            conn = _sqlite3.connect(FRACTAL_DB, check_same_thread=False)
            row = conn.execute("SELECT COUNT(*) FROM fractal_chunks").fetchone()
            conn.close()
            total_chunks = row[0] if row else 0
        except Exception:
            pass
    payload = {
        "status": "ok",
        "fractal_db": FRACTAL_DB,
        "db_exists": db_exists,
        "db_size_mb": db_size_mb,
        "total_chunks": total_chunks,
        "retrieve_sample": RETRIEVE_SAMPLE,
        "top_k": TOP_K_DEFAULT,
        "min_score": MIN_SCORE,
    }
    return payload


@app.post("/retrieve")
async def retrieve_endpoint(request: Request):
    """
    Direct retrieval endpoint.
    Body: {"query": "...", "top_k": 6}
    Returns: {"chunks": [...]}
    """
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", TOP_K_DEFAULT))

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    chunks = retrieve(query, top_k=top_k)
    return {"chunks": chunks, "count": len(chunks)}


@app.post("/runsync")
async def runsync(request: Request):
    """
    RunPod serverless handler entry point.
    Expects RunPod's standard {"input": {"query": "..."}} envelope.
    Returns context chunks for the gateway to splice into the prompt.
    """
    body = await request.json()
    inp = body.get("input", {})
    query = inp.get("query", "")
    top_k = int(inp.get("top_k", TOP_K_DEFAULT))

    chunks = retrieve(query, top_k=top_k)

    context = "\n\n---\n\n".join(chunks) if chunks else ""
    return {"output": {"context": context, "chunks": chunks, "count": len(chunks)}}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[LUMA] Starting on port {LUMA_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=LUMA_PORT)
