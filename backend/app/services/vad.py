"""Voice activity detection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..logging_utils import get_logger

try:  # pragma: no cover - optional dependency
    import webrtcvad  # type: ignore

    _WEBRTCVAD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore
    _WEBRTCVAD_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from silero_vad import load_silero_vad  # type: ignore

    _SILERO_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    load_silero_vad = None  # type: ignore
    _SILERO_AVAILABLE = False

logger = get_logger(__name__)
logger.debug("VAD service module loaded")


@dataclass
class VadResult:
    """Result from a VAD check."""

    has_speech: bool
    probability: float
    backend: str


class VoiceActivityDetector:
    """Wrapper around WebRTC / Silero VAD with an energy fallback."""

    def __init__(self, sample_rate: int, threshold: float = 0.35) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._webrtc_vad: Optional["webrtcvad.Vad"] = None  # type: ignore[name-defined]
        self._silero_model: Optional[torch.nn.Module] = None  # type: ignore[attr-defined]

        if _WEBRTCVAD_AVAILABLE:
            try:
                self._webrtc_vad = webrtcvad.Vad(2)  # type: ignore[attr-defined]
                logger.info("Loaded WebRTC VAD")
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Failed to load WebRTC VAD", exc_info=exc)
        if self._webrtc_vad is None and _SILERO_AVAILABLE and _TORCH_AVAILABLE:
            try:
                self._silero_model = load_silero_vad()
                logger.info("Loaded Silero VAD model")
            except Exception as exc:  # pragma: no cover - best-effort load
                logger.warning("Failed to load Silero VAD, falling back to energy gate", exc_info=exc)
        if self._webrtc_vad is None and self._silero_model is None:
            logger.info("Using energy-based VAD fallback")

    @property
    def backend_name(self) -> str:
        if self._webrtc_vad is not None:
            return "webrtc"
        if self._silero_model is not None:
            return "silero"
        return "energy"

    def detect(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> VadResult:
        """Determine whether the buffer contains speech."""

        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)

        sr = sample_rate or self.sample_rate
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)
            sr = self.sample_rate

        if self._webrtc_vad is not None:
            return self._detect_with_webrtc(audio, sr)
        if self._silero_model is not None:
            return self._detect_with_silero(audio, sr)
        return self._detect_with_energy(audio)

    def _detect_with_webrtc(self, audio: np.ndarray, sample_rate: int) -> VadResult:
        frame_size = int(0.03 * sample_rate)
        if frame_size <= 0:
            return VadResult(has_speech=False, probability=0.0, backend=self.backend_name)
        int_audio = self._to_int16(audio)
        usable = (int_audio.size // frame_size) * frame_size
        if usable == 0:
            return VadResult(has_speech=False, probability=0.0, backend=self.backend_name)
        frames = int_audio[:usable].reshape(-1, frame_size)
        speech_frames = 0
        for frame in frames:
            try:
                if self._webrtc_vad.is_speech(frame.tobytes(), sample_rate):  # type: ignore[union-attr]
                    speech_frames += 1
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("WebRTC VAD frame error: %s", exc)
        probability = speech_frames / max(len(frames), 1)
        has_speech = probability >= self.threshold
        logger.debug(
            "WebRTC VAD completed",
            extra={"frames": len(frames), "probability": probability, "threshold": self.threshold},
        )
        return VadResult(has_speech=has_speech, probability=probability, backend=self.backend_name)

    def _detect_with_silero(self, audio: np.ndarray, sample_rate: int) -> VadResult:
        tensor = torch.from_numpy(audio).float()  # type: ignore[attr-defined]
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():  # type: ignore[attr-defined]
            prob = float(self._silero_model(tensor, sample_rate).item())  # type: ignore[operator]
        has_speech = prob >= self.threshold
        logger.debug(
            "Silero VAD completed",
            extra={"probability": prob, "threshold": self.threshold},
        )
        return VadResult(has_speech=has_speech, probability=prob, backend=self.backend_name)

    def _detect_with_energy(self, audio: np.ndarray) -> VadResult:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        reference = 0.03
        probability = float(np.clip(rms / reference, 0.0, 1.0))
        has_speech = probability >= self.threshold
        logger.debug(
            "Energy VAD completed",
            extra={"probability": probability, "threshold": self.threshold},
        )
        return VadResult(has_speech=has_speech, probability=probability, backend=self.backend_name)

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        return (audio_clipped * 32767).astype(np.int16)

    @staticmethod
    def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        if from_rate == to_rate or audio.size == 0:
            return audio.astype(np.float32)
        duration = audio.size / float(from_rate)
        target_length = int(duration * to_rate)
        if target_length <= 0:
            return audio.astype(np.float32)
        old_times = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
        new_times = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float64)
        resampled = np.interp(new_times, old_times, audio)
        return resampled.astype(np.float32)


__all__ = ["VoiceActivityDetector", "VadResult"]
