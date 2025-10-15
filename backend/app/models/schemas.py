"""Pydantic schemas for API payloads."""
from __future__ import annotations

from typing import List, Literal, Optional

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
    teacher_audio_sample_rate: Optional[int] = Field(
        None, description="Sample rate in Hz for the teacher audio clip"
    )
    debug: Optional[dict] = Field(default=None, description="Optional debug information")


class HealthResponse(BaseModel):
    status: str
    whisper_loaded: bool
    vad_backend: str
    tts_available: bool
    llm_available: bool


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationRequest(BaseModel):
    message: str = Field(..., description="Latest learner utterance in text form")
    history: List[ChatMessage] = Field(
        default_factory=list, description="Prior conversation turns for additional context"
    )
    task_id: Optional[str] = Field(
        default=None, description="Optional curriculum task identifier to steer the conversation"
    )


class ConversationResponse(BaseModel):
    reply: ChatMessage
    history: List[ChatMessage]
    focus_phrase: Optional[str] = Field(
        default=None, description="Lao phrase the tutor suggests practising"
    )
    focus_translation: Optional[str] = Field(
        default=None, description="English gloss for the focus phrase"
    )
    teacher_audio_base64: Optional[str] = Field(
        default=None, description="Optional teacher audio clip encoded as base64 float32 PCM"
    )
    teacher_audio_sample_rate: Optional[int] = Field(
        default=None, description="Sample rate corresponding to the teacher audio clip"
    )
    debug: Optional[dict] = Field(default=None, description="Backend metadata for diagnostics")