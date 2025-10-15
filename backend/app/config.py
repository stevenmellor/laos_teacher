"""Application configuration handling."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - import resolution differs across Pydantic versions
    from pydantic import BaseSettings, Field
except Exception:  # Pydantic v2 raises a custom error; fall back to the v1 compatibility layer
    from pydantic.v1 import BaseSettings, Field  # type: ignore


class Settings(BaseSettings):
    """Central configuration for the tutor backend."""

    app_name: str = Field("Lao Tutor Backend", description="Human readable application name")
    environment: str = Field("development", description="Current execution environment")
    data_dir: Path = Field(Path("data"), description="Directory for persistent data files")
    model_dir: Path = Field(Path("models"), description="Directory containing ML models")
    whisper_model_size: str = Field(
        "small", description="Default Whisper model size identifier (tiny, base, small, medium, large)"
    )
    sqlite_path: Path = Field(Path("data/tutor.db"), description="Path to SQLite database file")
    enable_pitch_feedback: bool = Field(
        False,
        description="Whether to compute pitch contours for pronunciation feedback. Requires librosa and numpy",
    )
    # Audio processing
    sample_rate: int = Field(16000, description="Target sample rate for audio processing")
    vad_threshold: float = Field(
        0.35,
        description="Decision threshold (0-1) for VAD probability across WebRTC, Silero, or energy fallback",
    )
    # Conversational LLM configuration
    llm_model_name: str = Field(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Default Hugging Face identifier for the conversational tutor model",
    )
    llm_device: str = Field(
        "cpu",
        description="Device identifier passed to transformers pipeline (cpu, cuda, mps)",
    )
    llm_max_new_tokens: int = Field(
        256,
        description="Maximum number of tokens to generate for each conversational turn",
    )
    llm_temperature: float = Field(
        0.7,
        description="Sampling temperature applied during conversational generation",
    )
    tts_model_name: str = Field(
        "facebook/mms-tts-lao",
        description="Hugging Face identifier for the Lao text-to-speech voice",
    )
    tts_device: str = Field(
        "cpu",
        description="Preferred device for TTS inference (cpu, cuda, mps)",
    )

    class Config:
        env_prefix = "LAO_TUTOR_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return memoized application settings."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


SettingsType = Settings
