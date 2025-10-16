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
    last_focus_phrase: Optional[str] = None
    last_focus_translation: Optional[str] = None
    awaiting_repeat: bool = False


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
        self._focus_indices: Dict[str, int] = {self.state.current_task: 0}
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

    def _current_focus(self, task_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        task = task_id or self.state.current_task
        bank = self._phrase_bank.get(task, {})
        if not bank:
            return None, None
        if task not in self._focus_indices:
            self._focus_indices[task] = 0
        index = self._focus_indices[task]
        phrases = list(bank.items())
        index = min(index, len(phrases) - 1)
        phrase, translation = phrases[index]
        return phrase, translation

    def _advance_focus(self, task_id: Optional[str] = None) -> None:
        task = task_id or self.state.current_task
        bank = self._phrase_bank.get(task, {})
        if not bank:
            return
        total = len(bank)
        current_index = self._focus_indices.get(task, 0)
        if current_index + 1 < total:
            self._focus_indices[task] = current_index + 1
        logger.debug(
            "Advanced focus phrase",
            extra={"task": task, "index": self._focus_indices.get(task, 0)},
        )

    def mark_focus_prompted(self, phrase: Optional[str], translation: Optional[str]) -> None:
        if not phrase:
            return
        self.state.last_focus_phrase = phrase
        self.state.last_focus_translation = translation
        self.state.awaiting_repeat = True
        logger.debug(
            "Marked focus prompt",
            extra={"phrase": phrase, "translation": translation},
        )

    def _register_success(self, task_id: Optional[str] = None) -> None:
        self.state.awaiting_repeat = False
        self._advance_focus(task_id)
        logger.info(
            "Learner completed focus phrase",
            extra={
                "phrase": self.state.last_focus_phrase,
                "task": task_id or self.state.current_task,
            },
        )

    def process_audio(self, audio: np.ndarray, sample_rate: int, task_id: Optional[str] = None) -> SegmentFeedback:
        if task_id:
            self.state.current_task = task_id
            logger.debug("Task updated", extra={"task_id": task_id})
        logger.info(
            "Processing learner audio",
            extra={
                "frames": int(audio.size),
                "sample_rate": sample_rate,
                "task": self.state.current_task,
            },
        )
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

        expected_phrase, expected_translation = self._current_focus(task_id)
        expected_romanised = (
            self.text_processor.segment(expected_phrase).romanised if expected_phrase else ""
        )
        expected_normalised = self._normalize_romanised(expected_romanised)

        vad_result = self.vad.detect(audio, sample_rate)
        logger.info(
            "VAD evaluated audio",
            extra={
                "backend": self.vad.backend_name,
                "has_speech": vad_result.has_speech,
                "probability": round(vad_result.probability, 3),
            },
        )
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
        logger.info(
            "ASR completed",
            extra={
                "text": asr_result.text,
                "language": asr_result.language,
                "confidence": round(asr_result.confidence, 3),
            },
        )
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
        success = False
        completed_focus: Optional[str] = None

        if not asr_result.text:
            corrections.append("ຂໍໃຫ້ເວົ້າອີກຄັ້ງ – Try repeating the target phrase.")
        else:
            heard_normalised = self._normalize_romanised(segmented.romanised)
            if contains_lao_characters(asr_result.text):
                if (
                    expected_phrase
                    and heard_normalised
                    and expected_normalised
                    and SequenceMatcher(None, heard_normalised, expected_normalised).ratio() >= 0.75
                ):
                    focus_phrase = expected_phrase
                    focus_translation = expected_translation or translation
                    focus_romanised = expected_romanised or segmented.romanised
                    praise = "ຍອດເຢັ່ຍ! Your pronunciation is improving."
                    success = True
                    completed_focus = expected_phrase
                else:
                    focus_phrase = asr_result.text
                    focus_translation = translation
                    focus_romanised = segmented.romanised
            else:
                romanised_match, romanised_translation = self._find_phrase_by_romanised(asr_result.text)
                matched_phrase, matched_translation = self._find_phrase_by_translation(asr_result.text)
                heard_ascii = self._normalize_romanised(asr_result.text)
                if (
                    self.state.awaiting_repeat
                    and expected_phrase
                    and expected_normalised
                    and heard_ascii
                    and SequenceMatcher(None, heard_ascii, expected_normalised).ratio() >= 0.65
                ):
                    focus_phrase = expected_phrase
                    focus_translation = expected_translation or translation or asr_result.text
                    focus_romanised = expected_romanised or self.text_processor.segment(expected_phrase).romanised
                    praise = "ດີຫຼາຍ! Great job repeating the Lao phrase."
                    translation = focus_translation
                    success = True
                    completed_focus = expected_phrase
                elif romanised_match:
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

            if success:
                self._register_success(task_id)
                next_phrase, next_translation = self._current_focus(task_id)
                if next_phrase:
                    focus_phrase = next_phrase
                    focus_translation = next_translation or focus_translation
                    focus_romanised = self.text_processor.segment(next_phrase).romanised
                    translation = focus_translation or translation
                    next_translation_text = focus_translation or "let's keep going."
                    corrections.append(
                        (
                            "Great! Let's build on that. Our next phrase is "
                            f"“{focus_phrase}” ({focus_romanised}) – {next_translation_text}"
                        )
                    )
                else:
                    translation = focus_translation or translation
            elif translation is None:
                hint = "Let's focus on today's target phrase. Repeat after me."
                if not self.translator.is_ready:
                    hint += (
                        " Our translation model is still downloading – run `uv pip install '.[llm]'` "
                        "and allow the NLLB weights to finish syncing."
                    )
                corrections.append(hint)
            else:
                card_id = completed_focus or focus_phrase or asr_result.text
                if success:
                    self.srs.log_review(card_id=card_id, ease=1.0)
                logger.info(
                    "Learner phrase recognised",
                    extra={
                        "text": asr_result.text,
                        "card_id": card_id,
                        "task": self.state.current_task,
                        "success": success,
                    },
                )

        if focus_phrase and not focus_romanised:
            focus_romanised = self.text_processor.segment(focus_phrase).romanised

        review_ids: List[str] = []
        if focus_phrase:
            review_ids.append(focus_phrase)
        elif translation:
            review_ids.append(asr_result.text)

        if not focus_phrase and expected_phrase:
            focus_phrase = expected_phrase
            focus_translation = expected_translation or translation
            focus_romanised = expected_romanised or focus_romanised
            translation = focus_translation or translation
            if not success:
                corrections.append(
                    (
                        f"Let's stay with '{focus_translation}' for now. Repeat {focus_phrase}"
                        f" ({focus_romanised})."
                    )
                )

        logger.info(
            "Segment feedback prepared",
            extra={
                "focus_phrase": focus_phrase,
                "translation": translation,
                "corrections": len(corrections),
                "praise": bool(praise),
            },
        )
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
            fallback = feedback.focus_phrase or self.state.last_focus_phrase
            if fallback and contains_lao_characters(fallback):
                text = fallback
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
        return self._current_focus(task_id)

    def export_phrase_banks(self) -> Dict[str, Dict[str, str]]:
        return {task: phrases.copy() for task, phrases in self._phrase_bank.items()}


__all__ = ["TutorEngine", "TutorState"]
