"""Core tutoring logic for handling learner utterances."""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..config import get_settings
from ..models.schemas import SegmentFeedback
from .asr import AsrService
from .nlp import LaoTextProcessor
from .srs import SrsRepository
from .tts import TtsResult, TtsService
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


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
        self.tts = TtsService()
        self.text_processor = LaoTextProcessor()
        self.srs = SrsRepository(settings.sqlite_path)
        self.state = TutorState()
        self._phrase_bank = self._load_phrase_bank()

    def _load_phrase_bank(self) -> Dict[str, Dict[str, str]]:
        # Minimal seed content; in real usage load from JSON/DB
        return {
            "day1_greetings": {
                "ສະບາຍດີ": "Hello",
                "ຂອບໃຈ": "Thank you",
                "ຈົ່ງພູດຊ້າ": "Please speak slowly",
            },
            "numbers_0_10": {
                "ສູນ": "0",
                "ໜຶ່ງ": "1",
                "ສອງ": "2",
            },
        }

    def process_audio(self, audio: np.ndarray, sample_rate: int, task_id: Optional[str] = None) -> SegmentFeedback:
        if task_id:
            self.state.current_task = task_id
        vad_result = self.vad.detect(audio, sample_rate)
        if not vad_result.has_speech:
            logger.debug("No speech detected (prob=%.2f)", vad_result.probability)
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

        corrections: List[str] = []
        praise: Optional[str] = None
        if not asr_result.text:
            corrections.append("ຂໍໃຫ້ເວົ້າອີກຄັ້ງ – Try repeating the target phrase.")
        elif translation is None:
            corrections.append("Let's focus on today's target phrase. Repeat after me.")
        else:
            praise = "ດີຫຼາຍ! Great job!"
            self.srs.log_review(card_id=asr_result.text, ease=1.0)

        return SegmentFeedback(
            lao_text=asr_result.text,
            romanised=segmented.romanised,
            translation=translation,
            corrections=corrections,
            praise=praise,
            review_card_ids=[asr_result.text] if translation else [],
        )

    def prepare_teacher_audio(
        self,
        feedback: Optional[SegmentFeedback] = None,
        *,
        text_override: Optional[str] = None,
    ) -> Optional[TtsResult]:
        text = text_override or (feedback.lao_text if feedback else "")
        if not text:
            bank = self._phrase_bank.get(self.state.current_task, {})
            text = next(iter(bank.keys()), "")
        if not text:
            return None
        tts_result = self.tts.synthesize(text)
        if tts_result is None:
            return None
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
