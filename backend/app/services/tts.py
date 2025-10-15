"""Text-to-speech interface for Lao tutor."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSpeechSeq2Seq = None  # type: ignore
    AutoProcessor = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


@dataclass
class TtsResult:
    audio_base64: str
    sample_rate: int


class TtsService:
    """Wrap Meta's MMS-TTS model with a graceful fallback."""

    def __init__(self, model_name: str = "facebook/mms-tts-lo") -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        if _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE:
            try:
                settings = get_settings()
                self._processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=settings.model_dir)
                self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    cache_dir=settings.model_dir,
                )
                logger.info("Loaded TTS model %s", self.model_name)
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Could not load TTS model: %s", exc)
        else:
            logger.warning("Transformers or torch unavailable; TTS will emit placeholders")

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._processor is not None

    def synthesize(self, text: str) -> Optional[TtsResult]:
        if not self.is_ready:
            logger.debug("Returning placeholder TTS for text: %s", text)
            return None
        inputs = self._processor(text=text, return_tensors="pt")  # type: ignore[operator]
        with torch.no_grad():  # type: ignore[operator]
            waveform = self._model.generate(**inputs)  # type: ignore[operator]
        audio = waveform.cpu().numpy().squeeze().astype(np.float32)  # type: ignore[operator]
        audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")
        return TtsResult(audio_base64=audio_base64, sample_rate=self._model.config.sampling_rate)  # type: ignore[union-attr]


__all__ = ["TtsService", "TtsResult"]
