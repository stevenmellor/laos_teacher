"""Lao-specific NLP helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..logging_utils import get_logger

logger = get_logger(__name__)
logger.debug("NLP service module loaded")


def contains_lao_characters(text: str) -> bool:
    """Return True if the string includes Lao script characters."""
    return any("\u0E80" <= char <= "\u0EFF" for char in text)


_LAO_RUN = re.compile(r"[\u0E80-\u0EFF]+")


def extract_first_lao_segment(text: str) -> Optional[str]:
    """Extract the first contiguous Lao substring from *text* if present."""

    if not text:
        logger.debug("extract_first_lao_segment received empty text")
        return None
    match = _LAO_RUN.search(text)
    if match:
        segment = match.group(0)
        logger.debug(
            "extract_first_lao_segment located substring",
            extra={"segment": segment, "length": len(segment)},
        )
        return segment
    logger.debug("extract_first_lao_segment found no Lao substring")
    return None

try:  # pragma: no cover - optional dependency
    from laonlp import LaoTokenizer  # type: ignore

    _LAONLP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LaoTokenizer = None  # type: ignore
    _LAONLP_AVAILABLE = False

# Simplified romanisation fallback mapping
_ROMAN_MAP = {
    "ກ": "k", "ຂ": "kh", "ຄ": "kh", "ງ": "ng", "ຈ": "ch", "ສ": "s",
    "ຊ": "s", "ດ": "d", "ຕ": "t", "ນ": "n", "ບ": "b", "ປ": "p",
    "ຜ": "ph", "ຝ": "f", "ພ": "ph", "ຟ": "f", "ມ": "m", "ຢ": "y",
    "ຣ": "r", "ລ": "l", "ວ": "w", "ຫ": "h", "ອ": "o", "ຮ": "h",
    "ະ": "a", "າ": "aa", "ິ": "i", "ີ": "ii", "ຸ": "u", "ູ": "uu",
    "ເ": "e", "ແ": "ae", "ໂ": "o", "ໄ": "ai", "ໃ": "ai", "ັ": "a",
}


@dataclass
class SegmentedText:
    tokens: List[str]
    romanised: str


class LaoTextProcessor:
    """Provides Lao segmentation and romanisation."""

    def __init__(self) -> None:
        self._tokenizer = LaoTokenizer() if _LAONLP_AVAILABLE else None
        if self._tokenizer is None:
            logger.info("LaoTokenizer unavailable; using simple character segmentation")
        else:
            logger.info("LaoTokenizer initialised")

    def segment(self, text: str) -> SegmentedText:
        if not text:
            logger.debug("Segmentation requested for empty text")
            return SegmentedText(tokens=[], romanised="")
        if self._tokenizer is not None:
            tokens = self._tokenizer.tokenize(text)
        else:
            tokens = list(text)
        romanised_tokens = [self._romanise_token(tok) for tok in tokens]
        romanised = " ".join(filter(None, romanised_tokens))
        logger.debug(
            "Segmentation complete",
            extra={"token_count": len(tokens), "romanised": romanised},
        )
        return SegmentedText(tokens=tokens, romanised=romanised)

    def _romanise_token(self, token: str) -> str:
        roman_chars = [_ROMAN_MAP.get(char, char) for char in token]
        return "".join(roman_chars)


__all__ = [
    "LaoTextProcessor",
    "SegmentedText",
    "contains_lao_characters",
    "extract_first_lao_segment",
]
