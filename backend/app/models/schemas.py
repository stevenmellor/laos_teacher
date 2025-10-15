"""Pydantic schemas for API payloads."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator

from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Schemas module loaded")


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
    focus_phrase: Optional[str] = Field(
        default=None, description="Preferred Lao phrase to practise in the next turn"
    )
    focus_translation: Optional[str] = Field(
        default=None, description="English gloss for the focus phrase"
    )
    focus_romanised: Optional[str] = Field(
        default=None, description="Romanised form of the focus phrase"
    )


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
    translation_available: bool


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationRequest(BaseModel):
    message: Optional[str] = Field(
        None, description="Latest learner utterance in text form (optional when audio is provided)"
    )
    audio_base64: Optional[str] = Field(
        None, description="Optional base64 encoded learner audio clip (PCM)"
    )
    sample_rate: Optional[int] = Field(
        None, description="Sample rate of the provided learner audio clip"
    )
    history: List[ChatMessage] = Field(
        default_factory=list, description="Prior conversation turns for additional context"
    )
    task_id: Optional[str] = Field(
        default=None, description="Optional curriculum task identifier to steer the conversation"
    )

    @root_validator(pre=True)
    def check_message_or_audio(cls, values: dict) -> dict:
        message = values.get("message")
        audio = values.get("audio_base64")
        if (not message or not str(message).strip()) and not audio:
            logger.warning("ConversationRequest missing message and audio")
            raise ValueError("Either 'message' or 'audio_base64' must be provided")
        logger.debug(
            "ConversationRequest payload accepted",
            extra={
                "has_message": bool(message and str(message).strip()),
                "has_audio": bool(audio),
            },
        )
        return values


class ConversationResponse(BaseModel):
    reply: ChatMessage
    history: List[ChatMessage]
    heard_text: Optional[str] = Field(
        default=None, description="Learner utterance decoded from audio, if supplied"
    )
    focus_phrase: Optional[str] = Field(
        default=None, description="Lao phrase the tutor suggests practising"
    )
    focus_translation: Optional[str] = Field(
        default=None, description="English gloss for the focus phrase"
    )
    spoken_text: Optional[str] = Field(
        default=None,
        description="Portion of the tutor reply that was synthesised into speech",
    )
    utterance_feedback: Optional[SegmentFeedback] = Field(
        default=None, description="Detailed feedback derived from the learner's spoken audio"
    )
    teacher_audio_base64: Optional[str] = Field(
        default=None, description="Optional teacher audio clip encoded as base64 float32 PCM"
    )
    teacher_audio_sample_rate: Optional[int] = Field(
        default=None, description="Sample rate corresponding to the teacher audio clip"
    )
    debug: Optional[dict] = Field(default=None, description="Backend metadata for diagnostics")
