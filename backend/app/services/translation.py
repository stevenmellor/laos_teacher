"""Translation helper for Lao -> English glosses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import get_settings
from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("Translation service module loaded")

try:  # pragma: no cover - optional dependency path
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class TranslationResult:
    """Structured translation output."""

    text: str
    backend: str


class TranslationService:
    """Lazy wrapper around Hugging Face seq2seq models for Lao->English glosses."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.translation_model_name
        self._source_lang = settings.translation_source_lang
        self._target_lang = settings.translation_target_lang
        self._device = settings.translation_device
        self._translator = None

        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers unavailable; translation service disabled")
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=settings.model_dir)
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = self._source_lang
            model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, cache_dir=settings.model_dir)
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos = tokenizer.lang_code_to_id.get(self._target_lang)
                if forced_bos is not None:
                    model.config.forced_bos_token_id = forced_bos
            device_index = -1
            if self._device.startswith("cuda"):
                device_index = 0
            elif self._device == "mps":
                device_index = 0
            self._translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=self._source_lang,
                tgt_lang=self._target_lang,
                max_length=256,
                device=device_index,
            )
            logger.info(
                "Translation model loaded",
                extra={"model": self._model_name, "device": self._device},
            )
        except Exception as exc:  # pragma: no cover - optional failure path
            logger.warning(
                "Translation model unavailable; glosses will fall back to placeholders",
                exc_info=exc,
            )
            self._translator = None

    @property
    def is_ready(self) -> bool:
        return self._translator is not None

    def translate(self, text: str) -> Optional[TranslationResult]:
        if not text:
            logger.debug("Translation requested for empty text")
            return None
        if self._translator is None:
            logger.debug("Translation skipped; backend not initialised")
            return None
        try:
            outputs = self._translator(text)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Translation failed", extra={"text": text}, exc_info=exc)
            return None
        if not outputs:
            logger.debug("Translation produced no output", extra={"text": text})
            return None
        translation = outputs[0].get("translation_text", "").strip()
        if not translation:
            logger.debug("Translation output empty", extra={"text": text})
            return None
        logger.debug(
            "Translation completed",
            extra={"source": text, "translation": translation},
        )
        return TranslationResult(text=translation, backend=self._model_name)


__all__ = ["TranslationResult", "TranslationService"]
