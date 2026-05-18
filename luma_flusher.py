#!/usr/bin/env python3
"""
luma_flusher.py — Conversation delta uploader for LUMA serverless worker

Uploads in-memory conversation spools to HuggingFace dataset staging.
Called periodically (every 3 min) from api_gateway.py and on end-of-thread markers.
Never blocks the chat path — always runs in a background thread.

Staging layout on HF dataset (chap18700/luma-raw-data):
conversations/YYYY-MM-DD/session_YYYYMMDD_HHMMSS_<uuid4short>.jsonl
"""

import os
import json
import hashlib
import threading
import datetime
import time
import uuid
from typing import List, Dict, Any

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "chap18700/luma-raw-data")

# Lock so two flush threads never collide
_upload_lock = threading.Lock()


def _build_jsonl(turns: List[Dict[str, Any]]) -> bytes:
    """Serialise a list of conversation turns to JSONL bytes."""
    lines = [json.dumps(t, ensure_ascii=False) for t in turns]
    return "\n".join(lines).encode("utf-8")


def _content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes — used for dedup in the ingest job."""
    return hashlib.sha256(data).hexdigest()


def flush_to_hf(turns: List[Dict[str, Any]], session_id: str = "") -> bool:
    """
    Upload a batch of conversation turns to HF dataset staging.

    Args:
        turns: List of dicts with keys: timestamp, user, assistant, session_id
        session_id: Optional session identifier string

    Returns:
        True on success, False on any error (errors are logged, never raised)
    """
    if not HF_TOKEN:
        print("[FLUSHER] HF_TOKEN not set — skipping upload")
        return False
    if not turns:
        print("[FLUSHER] No turns to flush")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[FLUSHER] huggingface_hub not installed")
        return False

    with _upload_lock:
        try:
            now = datetime.datetime.utcnow()
            date_str = now.strftime("%Y-%m-%d")
            ts_str = now.strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            filename = f"session_{ts_str}_{uid}.jsonl"
            hf_path = f"conversations/{date_str}/{filename}"

            payload = _build_jsonl(turns)
            chash = _content_hash(payload)

            # Attach metadata header as first line
            meta = json.dumps({
                "_meta": True,
                "session_id": session_id or uid,
                "content_hash": chash,
                "turn_count": len(turns),
                "uploaded_utc": now.isoformat()
            }, ensure_ascii=False).encode("utf-8")
            full_payload = meta + b"\n" + payload

            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=full_payload,
                path_in_repo=hf_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"[auto] conversation delta {ts_str} ({len(turns)} turns)"
            )
            print(f"[FLUSHER] Uploaded {len(turns)} turns -> {hf_path} (hash={chash[:12]})")
            return True

        except Exception as e:
            print(f"[FLUSHER] Upload failed: {e}")
            return False


def flush_async(turns: List[Dict[str, Any]], session_id: str = "") -> None:
    """
    Non-blocking wrapper — spawns a daemon thread so the chat response
    is never delayed waiting for the HF upload.
    """
    if not turns:
        return
    t = threading.Thread(
        target=flush_to_hf,
        args=(list(turns), session_id),  # copy so caller can clear spool safely
        daemon=True
    )
    t.start()


def start_flusher(spool: list, lock: threading.Lock, interval_seconds: int = 180) -> None:
    """
    Start a background daemon thread that periodically flushes the conversation
    spool to HF staging. Call once from api_gateway.py at startup.

    Args:
        spool:            The shared conversation_spool list from api_gateway.py
        lock:             The spool_lock threading.Lock from api_gateway.py
        interval_seconds: How often to flush (default 180s = 3 minutes)
    """
    session_id = uuid.uuid4().hex[:8]

    def _loop():
        print(f"[FLUSHER] Periodic flush thread started — interval={interval_seconds}s session={session_id}")
        while True:
            time.sleep(interval_seconds)
            with lock:
                if not spool:
                    continue
                batch = list(spool)
                spool.clear()
            print(f"[FLUSHER] Periodic flush: {len(batch)} turns -> HF")
            flush_async(batch, session_id=session_id)

    t = threading.Thread(target=_loop, daemon=True, name="luma-flusher")
    t.start()
