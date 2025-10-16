"""Core tutoring logic for handling learner utterances."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import re

import numpy as np

from ..config import get_settings
from ..logging_utils import get_logger
from ..models.schemas import SegmentFeedback
from .asr import AsrService
from .nlp import LaoTextProcessor, contains_lao_characters, extract_first_lao_segment
from .srs import SrsRepository
from .tts import TtsResult, TtsService
from .translation import TranslationService
from .vad import VoiceActivityDetector

logger = get_logger(__name__)
logger.debug("Tutor engine module loaded")


@dataclass
class TutorState:
    current_task: str = "day1_greetings"
    turn_count: int = 0


class TutorEngine:
    """High-level orchestration of the Lao tutor."""

    def __init__(self) -> None:
        settings = get_settings()
        self.vad = VoiceActivityDetector(sample_rate=settings.sample_rate, threshold=settings.vad_threshold)
        self.asr = AsrService()
        self.tts = TtsService(settings.tts_model_name, settings.tts_device)
        self.text_processor = LaoTextProcessor()
        self.srs = SrsRepository(settings.sqlite_path)
        self.translator = TranslationService()
        self.state = TutorState()
        self._phrase_bank = self._load_phrase_bank()
        self._romanised_cache: Dict[str, str] = {}
        logger.info(
            "Tutor engine initialised",
            extra={
                "task": self.state.current_task,
                "vad_backend": self.vad.backend_name,
            },
        )

    def _load_phrase_bank(self) -> Dict[str, Dict[str, str]]:
        # Minimal seed content; in real usage load from JSON/DB
        bank = {
            "day1_greetings": {
                "ສະບາຍດີ": "Hello",
                "ຂອບໃຈ": "Thank you",
                "ຈົ່ງພູດຊ້າ": "Please speak slowly",
                "ທ່ານສະບາຍດີບໍ?": "How are you?",
                "ຂ້ອຍສຸກສະບາຍ": "I'm doing well",
                "ສະບາຍດີຕອນເຊົ້າ": "Good morning",
                "ສະບາຍດີຕອນແລງ": "Good evening",
                "ຂໍໂທດ": "Sorry",
                "ບໍ່ເປັນຫຍັງ": "You're welcome",
            },
            "numbers_0_10": {
                "ສູນ": "0",
                "ໜຶ່ງ": "1",
                "ສອງ": "2",
            },
        }
        logger.debug("Phrase bank loaded", extra={"tasks": list(bank.keys())})
        return bank

    def _normalize_key(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower().strip()
        return re.sub(r"[^a-z0-9\s]", "", lowered)

    def _normalize_romanised(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"[^a-z]", "", text.lower())

    def _get_cached_romanised(self, phrase: str) -> str:
        cached = self._romanised_cache.get(phrase)
        if cached is not None:
            return cached
        romanised = self.text_processor.segment(phrase).romanised
        normalized = self._normalize_romanised(romanised)
        self._romanised_cache[phrase] = normalized
        logger.debug(
            "Cached romanised key",
            extra={"phrase": phrase, "normalized": normalized},
        )
        return normalized

    def _find_phrase_by_translation(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Return the best Lao phrase whose English gloss resembles *text*."""

        if not text:
            return None, None

        normalized = self._normalize_key(text)
        if not normalized:
            return None, None

        best_phrase: Optional[str] = None
        best_translation: Optional[str] = None
        best_score = 0

        for task, phrases in self._phrase_bank.items():
            for phrase, translation in phrases.items():
                if not translation:
                    continue

                candidate = self._normalize_key(translation)
                if not candidate:
                    continue

                exact_match = candidate == normalized
                subset_match = candidate in normalized or normalized in candidate

                if not exact_match and not subset_match:
                    continue

                score = len(candidate)
                if score > best_score:
                    best_phrase = phrase
                    best_translation = translation
                    best_score = score
                    logger.debug(
                        "Matched English phrase to Lao target",
                        extra={
                            "task": task,
                            "phrase": phrase,
                            "translation": translation,
                            "score": score,
                            "normalized_query": normalized,
                        },
                    )

        return best_phrase, best_translation

    def _find_phrase_by_romanised(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Match ASCII/romanised learner input back to a Lao focus phrase."""

        normalized_query = self._normalize_romanised(text)
        if not normalized_query:
            return None, None

        best_phrase: Optional[str] = None
        best_translation: Optional[str] = None
        best_score = 0.0

        for task, phrases in self._phrase_bank.items():
            for phrase, translation in phrases.items():
                romanised_key = self._get_cached_romanised(phrase)
                if not romanised_key:
                    continue
                ratio = SequenceMatcher(None, normalized_query, romanised_key).ratio()
                if ratio < 0.55:
                    continue
                if ratio > best_score:
                    best_phrase = phrase
                    best_translation = translation
                    best_score = ratio
                    logger.debug(
                        "Matched romanised input to Lao phrase",
                        extra={
                            "phrase": phrase,
                            "translation": translation,
                            "ratio": ratio,
                            "normalized_query": normalized_query,
                            "task": task,
                        },
                    )

        return best_phrase, best_translation

    def process_audio(self, audio: np.ndarray, sample_rate: int, task_id: Optional[str] = None) -> SegmentFeedback:
        if task_id:
            self.state.current_task = task_id
            logger.debug("Task updated", extra={"task_id": task_id})
        if not self.asr.is_ready:
            logger.error(
                "ASR backend unavailable", extra={"task": self.state.current_task}
            )
            return SegmentFeedback(
                lao_text="",
                romanised="",
                translation=None,
                corrections=[
                    "ລະບົບກຳລັງຕຽມການຮັບຟັງ – The speech recogniser is still preparing. "
                    "Install the speech extras with `uv pip install '.[speech]'` and wait for the Whisper model download to finish.",
                ],
                praise=None,
            )

        vad_result = self.vad.detect(audio, sample_rate)
        if not vad_result.has_speech:
            logger.debug(
                "No speech detected",
                extra={"probability": vad_result.probability, "task": self.state.current_task},
            )
            return SegmentFeedback(
                lao_text="",
                romanised="",
                translation=None,
                corrections=["ລອງເວົ້າອີກຄັ້ງ – I didn't catch anything."],
                praise=None,
            )

        asr_result = self.asr.transcribe(audio, sample_rate)
        segmented = self.text_processor.segment(asr_result.text)
        translation = self._phrase_bank.get(self.state.current_task, {}).get(asr_result.text)
        focus_from_translation: Optional[str] = None
        if translation is None and asr_result.text:
            translation_result = self.translator.translate(asr_result.text)
            if translation_result:
                if translation_result.direction == "lo->en":
                    translation = translation_result.text
                else:
                    lao_focus = extract_first_lao_segment(translation_result.text)
                    if lao_focus:
                        focus_from_translation = lao_focus
                    else:
                        logger.debug(
                            "Reverse translation produced non-Lao text",
                            extra={
                                "source": asr_result.text,
                                "translation": translation_result.text,
                                "direction": translation_result.direction,
                            },
                        )
                    translation = translation or asr_result.text
                logger.info(
                    "Translation generated",
                    extra={
                        "source": asr_result.text,
                        "translation": translation_result.text,
                        "direction": translation_result.direction,
                        "backend": translation_result.backend,
                    },
                )

        corrections: List[str] = []
        praise: Optional[str] = None
        focus_phrase: Optional[str] = None
        focus_translation: Optional[str] = None
        focus_romanised: Optional[str] = None

        if not asr_result.text:
            corrections.append("ຂໍໃຫ້ເວົ້າອີກຄັ້ງ – Try repeating the target phrase.")
        else:
            if contains_lao_characters(asr_result.text):
                focus_phrase = asr_result.text
                focus_translation = translation
                focus_romanised = segmented.romanised
            else:
                romanised_match, romanised_translation = self._find_phrase_by_romanised(asr_result.text)
                matched_phrase, matched_translation = self._find_phrase_by_translation(asr_result.text)
                if romanised_match:
                    focus_phrase = romanised_match
                    focus_translation = romanised_translation or translation or asr_result.text
                    focus_romanised = self.text_processor.segment(romanised_match).romanised
                    corrections.append(
                        (
                            f"Let's learn to say '{focus_translation}' in Lao: {focus_phrase}"
                            f" ({focus_romanised})."
                        )
                    )
                    translation = focus_translation
                elif matched_phrase:
                    focus_phrase = matched_phrase
                    focus_translation = matched_translation or translation or asr_result.text
                    focus_romanised = self.text_processor.segment(matched_phrase).romanised
                    corrections.append(
                        (
                            f"Let's learn to say '{focus_translation}' in Lao: {matched_phrase}"
                            f" ({focus_romanised})."
                        )
                    )
                    translation = focus_translation
                elif focus_from_translation:
                    focus_phrase = focus_from_translation
                    focus_translation = translation or asr_result.text
                    focus_romanised = self.text_processor.segment(focus_phrase).romanised
                    corrections.append(
                        (
                            f"Let's practise '{focus_translation}' in Lao: {focus_phrase}"
                            f" ({focus_romanised})."
                        )
                    )
                    translation = focus_translation
                elif translation:
                    corrections.append(
                        f"We'll map your English phrase to Lao together. Try saying the Lao for '{translation}'."
                    )
                else:
                    corrections.append(
                        "I heard English audio. Let's try speaking the Lao phrase slowly together."
                    )

            if translation is None:
                hint = "Let's focus on today's target phrase. Repeat after me."
                if not self.translator.is_ready:
                    hint += (
                        " Our translation model is still downloading – run `uv pip install '.[llm]'` "
                        "and allow the NLLB weights to finish syncing."
                    )
                corrections.append(hint)
            else:
                praise = "ດີຫຼາຍ! Great job!"
                card_id = focus_phrase or asr_result.text
                self.srs.log_review(card_id=card_id, ease=1.0)
                logger.info(
                    "Learner phrase recognised",
                    extra={
                        "text": asr_result.text,
                        "card_id": card_id,
                        "task": self.state.current_task,
                    },
                )

        if focus_phrase and not focus_romanised:
            focus_romanised = self.text_processor.segment(focus_phrase).romanised

        review_ids: List[str] = []
        if focus_phrase:
            review_ids.append(focus_phrase)
        elif translation:
            review_ids.append(asr_result.text)

        return SegmentFeedback(
            lao_text=asr_result.text,
            romanised=segmented.romanised,
            translation=translation,
            corrections=corrections,
            praise=praise,
            review_card_ids=review_ids,
            focus_phrase=focus_phrase,
            focus_translation=focus_translation or translation,
            focus_romanised=focus_romanised,
        )

    def dependency_status(self) -> Dict[str, bool]:
        return {
            "asr_ready": self.asr.is_ready,
            "tts_ready": self.tts.is_ready,
            "translation_ready": self.translator.is_ready,
        }

    def prepare_teacher_audio(
        self,
        feedback: Optional[SegmentFeedback] = None,
        *,
        text_override: Optional[str] = None,
    ) -> Optional[TtsResult]:
        text = (text_override or (
            feedback.focus_phrase
            if feedback and feedback.focus_phrase
            else (feedback.lao_text if feedback else "")
        )).strip()
        if not text:
            bank = self._phrase_bank.get(self.state.current_task, {})
            text = next(iter(bank.keys()), "").strip()
        if feedback and text and not contains_lao_characters(text):
            if feedback.focus_phrase and contains_lao_characters(feedback.focus_phrase):
                text = feedback.focus_phrase
            else:
                logger.debug(
                    "Skipping TTS for non-Lao text",
                    extra={"text": text, "task": self.state.current_task},
                )
                return None
        if not text:
            logger.debug("No text available for TTS", extra={"task": self.state.current_task})
            return None
        tts_result = self.tts.synthesize(text)
        if tts_result is None:
            logger.warning("TTS synthesis failed", extra={"text": text})
            return None
        logger.debug(
            "Prepared teacher audio",
            extra={"text": text, "sample_rate": tts_result.sample_rate},
        )
        return tts_result

    def get_phrase_bank(self, task_id: Optional[str] = None) -> Dict[str, str]:
        if task_id:
            return self._phrase_bank.get(task_id, {})
        return self._phrase_bank.get(self.state.current_task, {})

    def get_focus_phrase(self, task_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        bank = self.get_phrase_bank(task_id)
        if not bank:
            return None, None
        phrase, translation = next(iter(bank.items()))
        return phrase, translation

    def export_phrase_banks(self) -> Dict[str, Dict[str, str]]:
        return {task: phrases.copy() for task, phrases in self._phrase_bank.items()}


__all__ = ["TutorEngine", "TutorState"]
