"""Automatic speech recognition service built around Whisper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import get_settings
from ..logging_utils import get_logger
from .nlp import contains_lao_characters

logger = get_logger(__name__)
logger.debug("ASR service module loaded")

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel  # type: ignore

    _WHISPER_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore
    _WHISPER_AVAILABLE = False


def _whisper_supported() -> bool:
    return _WHISPER_AVAILABLE


@dataclass
class AsrResult:
    text: str
    language: Optional[str]
    confidence: float


class AsrService:
    """Load Whisper on-demand and expose a transcription helper."""

    def __init__(self, model_size: Optional[str] = None, device: str = "auto") -> None:
        settings = get_settings()
        self.model_size = model_size or settings.whisper_model_size
        self.model_identifier = settings.whisper_model_id.strip() or self.model_size
        self.device = device
        self._model: Optional[WhisperModel] = None  # type: ignore[assignment]
        self._language_hint = settings.whisper_language.strip() or None
        self._auto_detect_fallback = settings.whisper_auto_detect_fallback
        self._compute_type = settings.whisper_compute_type

        if _whisper_supported():
            try:
                self._model = WhisperModel(
                    self.model_identifier,
                    device=self.device,
                    compute_type=self._compute_type,
                    download_root=str(settings.model_dir),
                )
                logger.info(
                    "Loaded Whisper model",
                    extra={
                        "model_size": self.model_size,
                        "model_identifier": self.model_identifier,
                        "device": self.device,
                        "language_hint": self._language_hint or "auto",
                        "auto_detect_fallback": self._auto_detect_fallback,
                        "compute_type": self._compute_type,
                    },
                )
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Could not load Whisper model", exc_info=exc)
        else:
            logger.warning("faster-whisper not available; ASR will return placeholders")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _run_transcription(self, audio: np.ndarray, language: Optional[str]) -> tuple[str, Optional[str], float]:
        if self._model is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Whisper model is not initialised")
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            audio=audio,
            beam_size=5,
            language=language,
            temperature=0.0,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        probability = getattr(info, "language_probability", 0.0) or 0.0
        detected_language = getattr(info, "language", None)
        return text, detected_language, probability

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:
        """Transcribe the audio buffer into Lao text."""

        if not self.is_ready:
            logger.debug("Returning placeholder transcription")
            return AsrResult(text="", language=None, confidence=0.0)

        language_hint = self._language_hint
        text, detected_language, probability = self._run_transcription(audio, language_hint)

        if (
            self._auto_detect_fallback
            and (not text or not contains_lao_characters(text))
            and (language_hint is not None)
        ):
            logger.info(
                "Falling back to auto language detection",
                extra={"initial_text": text, "hint": language_hint},
            )
            text, detected_language, probability = self._run_transcription(audio, None)

        logger.debug(
            "ASR transcription complete",
            extra={"text": text, "language": detected_language, "confidence": probability},
        )
        return AsrResult(text=text, language=detected_language, confidence=probability)


__all__ = ["AsrService", "AsrResult"]
