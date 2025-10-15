"""FastAPI entrypoint for the Lao tutor backend."""
from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
def index() -> dict[str, str]:
    """Simple landing endpoint for service discovery."""
    return {
        "service": "lao-tutor",
        "message": "Send audio to /api/v1/utterance or probe /health",
        "docs": "/docs",
    }


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


@app.post("/api/v1/utterance", response_model=UtteranceResponse)
def handle_utterance(payload: UtteranceRequest) -> UtteranceResponse:
    sample_rate = payload.sample_rate or settings.sample_rate
    audio = _decode_audio(payload.audio_base64, sample_rate)
    feedback: SegmentFeedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
    teacher_audio_base64 = tutor_engine.prepare_teacher_audio(feedback)
    debug_info: dict[str, Any] = {
        "task": tutor_engine.state.current_task,
        "vad_backend": tutor_engine.vad.backend_name,
        "asr_ready": tutor_engine.asr.is_ready,
        "tts_ready": tutor_engine.tts.is_ready,
    }
    return UtteranceResponse(
        feedback=feedback,
        teacher_audio_base64=teacher_audio_base64,
        debug=debug_info,
    )


__all__ = ["app"]
