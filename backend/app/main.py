"""FastAPI entrypoint for the Lao tutor backend."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .logging_utils import configure_logging, get_logger
from .models.schemas import (
    ChatMessage,
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
    SegmentFeedback,
    UtteranceRequest,
    UtteranceResponse,
)
from .services.tutor import TutorEngine
from .services.llm import ConversationService

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Lao Tutor API")
settings = get_settings()
configure_logging(settings.log_dir)
logger = get_logger(__name__)
logger.info("Application initialised", extra={"log_dir": str(settings.log_dir)})

try:
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
except AssertionError:
    templates = None
    logger.warning("Jinja2 unavailable; falling back to plain HTML rendering")
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

tutor_engine = TutorEngine()
conversation_service = ConversationService(tutor_engine.export_phrase_banks())
logger.info(
    "Tutor services ready",
    extra={
        "vad": tutor_engine.vad.backend_name,
        "tts_ready": tutor_engine.tts.is_ready,
        "llm_ready": conversation_service.is_ready,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index(request: Request) -> Response:
    """Landing endpoint that serves HTML by default with a JSON fallback."""
    accept_header = (request.headers.get("accept") or "").lower()
    logger.debug("Serving index", extra={"accept": accept_header})
    payload = {
        "service": "lao-tutor",
        "message": "Use /api/v1/utterance for audio interactions or /health for status",
        "docs": "/docs",
    }

    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(payload)

    if templates is None:
        html_path = BASE_DIR / "templates" / "index.html"
        logger.debug("Rendering index via static file fallback", extra={"path": str(html_path)})
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content=content)

    return templates.TemplateResponse("index.html", {"request": request, **payload})


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    logger.info(
        "Health check invoked",
        extra={
            "vad_backend": tutor_engine.vad.backend_name,
            "asr_ready": tutor_engine.asr.is_ready,
            "tts_ready": tutor_engine.tts.is_ready,
            "llm_ready": conversation_service.is_ready,
            "translation_ready": tutor_engine.translator.is_ready,
        },
    )
    return HealthResponse(
        status="ok",
        whisper_loaded=tutor_engine.asr.is_ready,
        vad_backend=tutor_engine.vad.backend_name,
        tts_available=tutor_engine.tts.is_ready,
        llm_available=conversation_service.is_ready,
        translation_available=tutor_engine.translator.is_ready,
    )


def _decode_audio(audio_base64: str, expected_sample_rate: int) -> np.ndarray:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        logger.warning("Invalid audio payload", exc_info=exc)
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc
    if len(audio_bytes) % 2 == 0:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio.size == 0:
        logger.warning("Empty audio payload received")
        raise HTTPException(status_code=400, detail="Empty audio payload")
    logger.debug(
        "Decoded audio payload",
        extra={"frames": int(audio.size), "expected_sample_rate": expected_sample_rate},
    )
    return audio


@app.post("/api/v1/utterance", response_model=UtteranceResponse)
def handle_utterance(payload: UtteranceRequest) -> UtteranceResponse:
    sample_rate = payload.sample_rate or settings.sample_rate
    logger.info(
        "Processing utterance",
        extra={
            "sample_rate": sample_rate,
            "task_id": payload.task_id,
            "has_audio": bool(payload.audio_base64),
        },
    )
    audio = _decode_audio(payload.audio_base64, sample_rate)
    feedback: SegmentFeedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
    logger.debug(
        "Utterance processed",
        extra={
            "heard_text": feedback.lao_text,
            "romanised": feedback.romanised,
            "corrections": len(feedback.corrections or []),
        },
    )
    tts_result = tutor_engine.prepare_teacher_audio(feedback)
    teacher_audio_base64 = tts_result.audio_base64 if tts_result else None
    teacher_audio_sample_rate = tts_result.sample_rate if tts_result else None
    debug_info: dict[str, Any] = {
        "task": tutor_engine.state.current_task,
        "vad_backend": tutor_engine.vad.backend_name,
        **tutor_engine.dependency_status(),
    }
    return UtteranceResponse(
        feedback=feedback,
        teacher_audio_base64=teacher_audio_base64,
        teacher_audio_sample_rate=teacher_audio_sample_rate,
        debug=debug_info,
    )


@app.post("/api/v1/conversation", response_model=ConversationResponse)
def handle_conversation(payload: ConversationRequest) -> ConversationResponse:
    history_payload = [message.dict() for message in payload.history]

    sample_rate = payload.sample_rate or settings.sample_rate
    utterance_feedback: Optional[SegmentFeedback] = None
    heard_text: Optional[str] = None

    message_text = payload.message.strip() if payload.message else ""

    logger.info(
        "Conversation turn received",
        extra={
            "history_count": len(history_payload),
            "has_audio": bool(payload.audio_base64),
            "task_id": payload.task_id,
        },
    )

    if payload.audio_base64:
        audio = _decode_audio(payload.audio_base64, sample_rate)
        utterance_feedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
        heard_text = utterance_feedback.lao_text or None
        if not message_text:
            message_text = utterance_feedback.lao_text or utterance_feedback.romanised or ""
        if not message_text and utterance_feedback.corrections:
            message_text = utterance_feedback.corrections[0]

    if not message_text.strip():
        message_text = "I could not speak clearly."

    dependency_status = tutor_engine.dependency_status()

    try:
        result = conversation_service.generate(
            history_payload,
            message_text,
            payload.task_id,
            observation=utterance_feedback,
        )
    except ValueError as exc:
        logger.error("Conversation generation failed", exc_info=exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tts_result = None
    spoken_text = result.spoken_text
    if spoken_text:
        tts_result = tutor_engine.prepare_teacher_audio(text_override=spoken_text)
    elif result.focus_phrase:
        tts_result = tutor_engine.prepare_teacher_audio(text_override=result.focus_phrase)

    response_history = [ChatMessage(**entry) for entry in result.history]
    reply_message = ChatMessage(role="assistant", content=result.reply_text)

    logger.debug(
        "Conversation turn processed",
        extra={
            "reply_chars": len(result.reply_text),
            "heard_text": heard_text,
            "spoken_text": spoken_text,
            "teacher_audio": bool(tts_result),
        },
    )

    debug_payload: dict[str, Any] = {**result.debug, **dependency_status}
    if payload.audio_base64:
        debug_payload.update(
            {
                "audio_processed": True,
                "sample_rate": sample_rate,
                "vad_backend": tutor_engine.vad.backend_name,
            }
        )
    else:
        debug_payload.setdefault("audio_processed", False)

    return ConversationResponse(
        reply=reply_message,
        history=response_history,
        heard_text=heard_text,
        focus_phrase=result.focus_phrase,
        focus_translation=result.focus_translation,
        spoken_text=spoken_text,
        utterance_feedback=utterance_feedback,
        teacher_audio_base64=tts_result.audio_base64 if tts_result else None,
        teacher_audio_sample_rate=tts_result.sample_rate if tts_result else None,
        debug=debug_payload,
    )


__all__ = ["app"]
