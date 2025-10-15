"""Automatic speech recognition service built around Whisper."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

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
        self.device = device
        self._model: Optional[WhisperModel] = None  # type: ignore[assignment]

        if _whisper_supported():
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    download_root=str(settings.model_dir),
                )
                logger.info("Loaded Whisper model %s", self.model_size)
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Could not load Whisper model: %s", exc)
        else:
            logger.warning("faster-whisper not available; ASR will return placeholders")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:
        """Transcribe the audio buffer into Lao text."""

        if not self.is_ready:
            logger.debug("Returning placeholder transcription")
            return AsrResult(text="", language=None, confidence=0.0)

        segments, info = self._model.transcribe(
            audio=audio,
            beam_size=5,
            language="lo",
            temperature=0.0,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return AsrResult(text=text.strip(), language=info.language, confidence=info.language_probability)


__all__ = ["AsrService", "AsrResult"]
