"""Translation helper for Lao -> English glosses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import SettingsType, get_settings
from ..logging_utils import get_logger
from .nlp import contains_lao_characters

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
    direction: str


class TranslationService:
    """Lazy wrapper around Hugging Face seq2seq models for Lao->English glosses."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.translation_model_name
        self._source_lang = settings.translation_source_lang
        self._target_lang = settings.translation_target_lang
        self._reverse_model_name = settings.translation_reverse_model_name
        self._reverse_source_lang = settings.translation_reverse_source_lang
        self._reverse_target_lang = settings.translation_reverse_target_lang
        self._device = settings.translation_device
        self._translator_lo_en = None
        self._translator_en_lo = None

        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers unavailable; translation service disabled")
            return

        self._translator_lo_en = self._load_pipeline(
            model_name=self._model_name,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            settings=settings,
        )
        self._translator_en_lo = self._load_pipeline(
            model_name=self._reverse_model_name,
            source_lang=self._reverse_source_lang,
            target_lang=self._reverse_target_lang,
            settings=settings,
        )

    def _load_pipeline(
        self,
        *,
        model_name: str,
        source_lang: str,
        target_lang: str,
        settings: SettingsType,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=settings.model_dir)
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = source_lang
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=settings.model_dir)
            if hasattr(tokenizer, "lang_code_to_id"):
                forced_bos = tokenizer.lang_code_to_id.get(target_lang)
                if forced_bos is not None:
                    model.config.forced_bos_token_id = forced_bos
            device_index = -1
            if self._device.startswith("cuda"):
                device_index = 0
            elif self._device == "mps":
                device_index = 0
            translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=source_lang,
                tgt_lang=target_lang,
                max_length=256,
                device=device_index,
            )
            logger.info(
                "Translation model loaded",
                extra={"model": model_name, "device": self._device, "src": source_lang, "tgt": target_lang},
            )
            return translator
        except Exception as exc:  # pragma: no cover - optional failure path
            logger.warning(
                "Translation model unavailable",
                extra={"model": model_name, "src": source_lang, "tgt": target_lang},
                exc_info=exc,
            )
            return None

    @property
    def is_ready(self) -> bool:
        return bool(self._translator_lo_en or self._translator_en_lo)

    def translate(self, text: str) -> Optional[TranslationResult]:
        if not text:
            logger.debug("Translation requested for empty text")
            return None
        if contains_lao_characters(text):
            translator = self._translator_lo_en
            direction = "lo->en"
            backend = self._model_name
        else:
            translator = self._translator_en_lo
            direction = "en->lo"
            backend = self._reverse_model_name

        if translator is None:
            logger.debug(
                "Translation skipped; backend not initialised",
                extra={"direction": direction},
            )
            return None

        try:
            outputs = translator(text)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Translation failed",
                extra={"text": text, "direction": direction},
                exc_info=exc,
            )
            return None
        if not outputs:
            logger.debug(
                "Translation produced no output",
                extra={"text": text, "direction": direction},
            )
            return None
        translation = outputs[0].get("translation_text", "").strip()
        if not translation:
            logger.debug(
                "Translation output empty",
                extra={"text": text, "direction": direction},
            )
            return None
        logger.debug(
            "Translation completed",
            extra={"source": text, "translation": translation, "direction": direction},
        )
        return TranslationResult(text=translation, backend=backend, direction=direction)


__all__ = ["TranslationResult", "TranslationService"]
