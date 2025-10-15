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
    from transformers import AutoTokenizer, VitsModel  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore
    VitsModel = None  # type: ignore
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

    def __init__(self, model_name: Optional[str] = None, device_override: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.tts_model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        if device_override:
            self._device = device_override
        else:
            self._device = settings.tts_device
        self._torch_device = None

        if _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=settings.model_dir)
                self._model = VitsModel.from_pretrained(
                    self.model_name,
                    cache_dir=settings.model_dir,
                )
                self._torch_device = self._resolve_device()
                if self._torch_device:
                    self._model = self._model.to(self._torch_device)
                logger.info("Loaded TTS model %s on %s", self.model_name, self._torch_device or "cpu")
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Could not load TTS model: %s", exc)
        else:
            logger.warning("Transformers or torch unavailable; TTS will emit placeholders")

    def _resolve_device(self):  # pragma: no cover - trivial helper
        target = (self._device or "cpu").lower()
        if target.startswith("cuda") and torch.cuda.is_available():  # type: ignore[operator]
            return torch.device("cuda")  # type: ignore[call-arg]
        if target == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")  # type: ignore[call-arg]
        return torch.device("cpu")  # type: ignore[call-arg]

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def synthesize(self, text: str) -> Optional[TtsResult]:
        if not self.is_ready:
            logger.debug("Returning placeholder TTS for text: %s", text)
            return None
        inputs = self._tokenizer(text, return_tensors="pt")  # type: ignore[operator]
        if self._torch_device:
            inputs = {key: value.to(self._torch_device) for key, value in inputs.items()}
        with torch.no_grad():  # type: ignore[operator]
            waveform = self._model(**inputs).waveform  # type: ignore[operator]
        audio = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
        audio_base64 = base64.b64encode(audio.tobytes()).decode("utf-8")
        sample_rate = int(getattr(self._model.config, "sampling_rate", 16000))  # type: ignore[union-attr]
        return TtsResult(audio_base64=audio_base64, sample_rate=sample_rate)


__all__ = ["TtsService", "TtsResult"]