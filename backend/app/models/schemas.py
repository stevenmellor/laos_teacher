"""Pydantic schemas for API payloads."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class UtteranceRequest(BaseModel):
    """Incoming audio buffer encoded as base64."""

    audio_base64: str = Field(..., description="Base64 encoded 16-bit PCM mono audio")
    sample_rate: Optional[int] = Field(None, description="Original sample rate of the recording")
    task_id: Optional[str] = Field(None, description="Current curriculum task identifier")


class SegmentFeedback(BaseModel):
    """Feedback about a learner's utterance."""

    lao_text: str = Field(..., description="Recognised Lao text")
    romanised: str = Field(..., description="BGN/PCGN romanised text")
    translation: Optional[str] = Field(None, description="English gloss")
    corrections: List[str] = Field(default_factory=list, description="List of correction hints")
    praise: Optional[str] = Field(None, description="Positive reinforcement message")
    review_card_ids: List[str] = Field(default_factory=list, description="Related spaced-repetition card IDs")


class UtteranceResponse(BaseModel):
    """Response payload for processed utterances."""

    feedback: SegmentFeedback
    teacher_audio_base64: Optional[str] = Field(
        None, description="Base64 encoded PCM audio of the teacher response"
    )
    debug: Optional[dict] = Field(default=None, description="Optional debug information")


class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    vad_backend: str
    tts_available: bool
