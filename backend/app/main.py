"""FastAPI entrypoint for the Lao tutor backend."""
from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import get_settings
from .models.schemas import HealthResponse, SegmentFeedback, UtteranceRequest, UtteranceResponse
from .services.tutor import TutorEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Lao Tutor API")
settings = get_settings()
tutor_engine = TutorEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    """Landing endpoint that serves HTML by default with a JSON fallback."""
    accept_header = (request.headers.get("accept") or "").lower()
    payload = {
        "service": "lao-tutor",
        "message": "Use /api/v1/utterance for audio interactions or /health for status",
        "docs": "/docs",
    }

    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(payload)

    html = """
    <main style="font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto; line-height: 1.6;">
        <h1 style="margin-bottom: 0.5rem;">Lao Tutor API</h1>
        <p style="margin-top: 0; color: #555;">Voice-first Lao language teaching backend.</p>
        <section>
            <h2>Try it out</h2>
            <ul>
                <li><strong>Health check:</strong> <code>/health</code></li>
                <li><strong>Submit audio:</strong> <code>/api/v1/utterance</code> (POST base64-encoded 16&nbsp;kHz mono PCM)</li>
                <li><strong>Interactive docs:</strong> <a href="/docs">Swagger UI</a></li>
            </ul>
        </section>
        <section>
            <h2>Quickstart</h2>
            <ol>
                <li>Base64-encode your Lao utterance audio (16&nbsp;kHz mono) and POST it to <code>/api/v1/utterance</code>.</li>
                <li>Receive transcription feedback, teaching prompts, and synthesized teacher audio.</li>
                <li>Poll <code>/health</code> to verify ASR/TTS availability.</li>
            </ol>
        </section>
        <footer style="margin-top: 3rem; font-size: 0.9rem; color: #777;">
            Prefer JSON? Send <code>Accept: application/json</code> with your request.
        </footer>
    </main>
    """
    return HTMLResponse(content=html)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        whisper_loaded=tutor_engine.asr.is_ready,
        vad_backend=tutor_engine.vad.backend_name,
        tts_available=tutor_engine.tts.is_ready,
    )


def _decode_audio(audio_base64: str, expected_sample_rate: int) -> np.ndarray:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc
    if len(audio_bytes) % 2 == 0:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return audio
