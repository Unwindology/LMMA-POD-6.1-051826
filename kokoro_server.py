#!/usr/bin/env python3
"""
kokoro_server.py — OpenAI-compatible TTS API using kokoro-onnx.

Exposes:
  GET  /health
  GET  /models
  GET  /v1/models
  GET  /audio/voices
  GET  /v1/audio/voices
  POST /v1/audio/speech
  POST /audio/speech

Open WebUI settings:
  TTS Engine: OpenAI
  API Base URL: http://<tailscale-ip>:8880
  API Key: kokoro  (any string)
  Voice: af_sky
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

PORT = int(os.environ.get("KOKORO_PORT", "8880"))
MODEL_DIR = Path(os.environ.get("KOKORO_MODEL_DIR", "/workspace/luma/tts/kokoro"))

VOICES = [
    {"id": "af_sky", "name": "Sky (US Female)"},
    {"id": "af_bella", "name": "Bella (US Female)"},
    {"id": "af_nicole", "name": "Nicole (US Female)"},
    {"id": "af_sarah", "name": "Sarah (US Female)"},
    {"id": "am_adam", "name": "Adam (US Male)"},
    {"id": "am_michael", "name": "Michael (US Male)"},
    {"id": "bf_emma", "name": "Emma (British Female)"},
    {"id": "bf_isabella", "name": "Isabella (British Female)"},
    {"id": "bm_george", "name": "George (British Male)"},
    {"id": "bm_lewis", "name": "Lewis (British Male)"},
]

app = FastAPI(title="Kokoro TTS", version="1.2-luma6-kokoro-onnx")
_kokoro = None


def _download(url: str, dest: Path) -> None:
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[KOKORO] Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"[KOKORO] Saved -> {dest} ({dest.stat().st_size} bytes)", flush=True)


def ensure_models():
    """Use matched v0.19 model + voices from kokoro-onnx release files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_DIR / "kokoro-v0_19.onnx"
    voices_path = MODEL_DIR / "voices.bin"

    sources = [
        (
            onnx_path,
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
            10_000_000,
        ),
        (
            voices_path,
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin",
            1_000_000,
        ),
    ]

    for dest, url, min_size in sources:
        if dest.exists() and dest.stat().st_size >= min_size:
            continue
        if dest.exists():
            print(
                f"[KOKORO] {dest} is {dest.stat().st_size} bytes; re-downloading",
                flush=True,
            )
            try:
                dest.unlink()
            except Exception:
                pass
        _download(url, dest)

    return onnx_path, voices_path


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        onnx_path, voices_path = ensure_models()
        print("[KOKORO] Loading kokoro-onnx model...", flush=True)
        _kokoro = Kokoro(str(onnx_path), str(voices_path))
        print("[KOKORO] Model ready.", flush=True)
    return _kokoro


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_sky"
    response_format: str = "wav"
    speed: float = 1.0


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kokoro-tts", "port": PORT}


@app.get("/models")
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "kokoro", "object": "model", "owned_by": "kokoro"}],
    }


@app.get("/audio/models")
async def audio_models():
    return {"models": [{"id": "kokoro", "name": "Kokoro 82M"}]}


@app.get("/audio/voices")
@app.get("/v1/audio/voices")
async def list_voices():
    return {"voices": VOICES}


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)

    try:
        kokoro = get_kokoro()
        samples, sample_rate = kokoro.create(
            text,
            voice=req.voice,
            speed=req.speed,
            lang="en-us",
        )

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"},
        )
    except Exception as e:
        print(f"[KOKORO] TTS error: {e}", flush=True)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    print(f"[KOKORO] Starting on port {PORT}", flush=True)
    try:
        get_kokoro()
    except Exception as e:
        print(f"[KOKORO] Pre-load failed; will retry on first request: {e}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
