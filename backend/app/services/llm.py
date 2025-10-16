"""Conversational LLM orchestration for the Lao tutor."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import get_settings
from ..logging_utils import get_logger
from ..models.schemas import SegmentFeedback
from .nlp import LaoTextProcessor, contains_lao_characters, extract_first_lao_segment

logger = get_logger(__name__)
logger.debug("Conversation service module loaded")

try:  # pragma: no cover - optional dependency path
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

try:  # pragma: no cover - optional dependency path
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


ChatHistory = List[Dict[str, str]]


@dataclass
class ConversationResult:
    """Structured output of a conversation turn."""

    reply_text: str
    history: ChatHistory
    focus_phrase: Optional[str]
    focus_translation: Optional[str]
    spoken_text: Optional[str]
    debug: Dict[str, str]


class ConversationService:
    """Wrap an LLM (or fallback heuristics) to keep the tutor conversational."""

    _SYSTEM_PROMPT = (
        "You are a warm, encouraging Lao language teacher who is fluent in English. "
        "Always present Lao phrases alongside a romanisation and an English translation so the learner understands the meaning."
    )

    def __init__(self, phrase_bank: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        settings = get_settings()
        self._phrase_bank = phrase_bank or {}
        self._model_name = settings.llm_model_name
        self._temperature = settings.llm_temperature
        self._max_new_tokens = settings.llm_max_new_tokens
        self._device = settings.llm_device
        self._generator = None
        self._tokenizer = None
        self._text_processor = LaoTextProcessor()

        if _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name, cache_dir=settings.model_dir
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    cache_dir=settings.model_dir,
                )
                device = 0 if self._device.startswith("cuda") else (0 if self._device == "mps" else -1)
                self._generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self._tokenizer,
                    device=device,
                )
                logger.info(
                    "Loaded conversational model",
                    extra={"model": self._model_name, "device": self._device},
                )
            except Exception as exc:  # pragma: no cover - optional failure path
                logger.warning(
                    "Conversation model unavailable; falling back to scripted replies",
                    exc_info=exc,
                )
        else:
            logger.warning("Transformers/torch unavailable; using scripted conversation fallback")

    @property
    def is_ready(self) -> bool:
        return self._generator is not None

    def _select_focus_phrase(self, task_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if task_id and task_id in self._phrase_bank:
            bank = self._phrase_bank[task_id]
        else:
            bank = next(iter(self._phrase_bank.values()), {})
        if not bank:
            return None, None
        phrase, translation = next(iter(bank.items()))
        return phrase, translation

    def _romanise(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return self._text_processor.segment(text).romanised

    def _build_summary(
        self,
        observation: Optional[SegmentFeedback],
        focus_phrase: Optional[str],
        focus_translation: Optional[str],
    ) -> str:
        lines: List[str] = []
        if observation and observation.corrections:
            for hint in observation.corrections:
                if "speech recogniser" in hint.lower():
                    lines.append(
                        "I'm still downloading the speech models so I might miss what you say. Keep the server running and I'll start listening as soon as Whisper finishes syncing."
                    )
                    break

        if observation and observation.lao_text:
            observed_translation = observation.translation or "We'll learn this meaning together."
            observed_romanised = observation.romanised or self._romanise(observation.lao_text)
            if contains_lao_characters(observation.lao_text):
                romanised_note = f" ({observed_romanised})" if observed_romanised else ""
                lines.append(
                    f"I heard you say \"{observation.lao_text}\"{romanised_note}. In English, that means \"{observed_translation}\"."
                )
            else:
                lines.append(
                    f"I heard you say \"{observation.lao_text}\" in English. Let's express that in Lao together."
                )
            if not focus_phrase:
                focus_phrase = observation.focus_phrase or observation.lao_text
                focus_translation = observation.focus_translation or observation.translation or focus_translation

        if focus_phrase:
            focus_romanised = observation.focus_romanised if observation else None
            if not focus_romanised:
                focus_romanised = self._romanise(focus_phrase)
            romanised_note = f" ({focus_romanised})" if focus_romanised else ""
            english_gloss = focus_translation or "our target phrase today."
            lines.append(
                f"Let's practise \"{focus_phrase}\"{romanised_note} – \"{english_gloss}\"."
            )

        lines.append("Repeat it aloud and pay close attention to the tone contour.")
        return " ".join(lines)

    def _format_prompt(
        self, history: ChatHistory, user_message: str, focus_phrase: Optional[str], focus_translation: Optional[str]
    ) -> str:
        conversation: ChatHistory = []
        for message in history[-8:]:
            if message.get("role") in {"user", "assistant"} and message.get("content"):
                conversation.append({"role": message["role"], "content": message["content"]})
        conversation.append({"role": "user", "content": user_message})

        prompt_parts = [f"System: {self._SYSTEM_PROMPT}"]
        if focus_phrase:
            prompt_parts.append(
                "System: Today's focus phrase is '"
                f"{focus_phrase}' which means '{focus_translation or '...'}'."
            )
        for item in conversation:
            role = item["role"].capitalize()
            prompt_parts.append(f"{role}: {item['content']}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        if not text:
            return ""
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^(system|user|assistant)\s*:\s*", stripped, re.IGNORECASE):
                continue
            lines.append(stripped)
        cleaned = " ".join(lines)
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentences) > 2:
            cleaned = " ".join(sentences[:2]).strip()
        return cleaned.strip()

    @staticmethod
    def _extract_lao_line(text: str) -> Optional[str]:
        lao_pattern = re.compile(r"([\u0E80-\u0EFF]+(?:[\u0E80-\u0EFF\s]+)?)")
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            match = lao_pattern.search(candidate)
            if match:
                return match.group(1)
        return None

    def _fallback_reply(
        self, user_message: str, focus_phrase: Optional[str], focus_translation: Optional[str]
    ) -> tuple[str, Optional[str]]:
        if not focus_phrase:
            reply = (
                "ສະບາຍດີ! I can guide you through Lao basics once vocabulary is loaded. "
                "Ask me about greetings or numbers to begin."
            )
            return reply, None
        reply = (
            f"ມາຝຶກກັນ! Repeat after me: {focus_phrase}. "
            f"It means {focus_translation or 'this is our target phrase today'}. "
            "Focus on the tone contour and say it once more."
        )
        return reply, focus_phrase

    def generate(
        self,
        history: ChatHistory,
        user_message: str,
        task_id: Optional[str] = None,
        observation: Optional[SegmentFeedback] = None,
    ) -> ConversationResult:
        if not user_message.strip():
            raise ValueError("Message must not be empty")

        focus_phrase, focus_translation = self._select_focus_phrase(task_id)

        if observation and observation.focus_phrase:
            focus_phrase = observation.focus_phrase
            if observation.focus_translation:
                focus_translation = observation.focus_translation
            elif observation.translation:
                focus_translation = observation.translation
        elif observation and observation.lao_text and contains_lao_characters(observation.lao_text):
            focus_phrase = observation.lao_text
            if observation.translation:
                focus_translation = observation.translation

        logger.info(
            "Generating tutor response",
            extra={
                "history_turns": len(history),
                "task_id": task_id,
                "focus_phrase": focus_phrase,
            },
        )

        spoken_text: Optional[str] = None

        summary = self._build_summary(observation, focus_phrase, focus_translation)
        introduction = ""
        if not history:
            intro_focus = focus_phrase or (observation.focus_phrase if observation else None)
            intro_translation = focus_translation or (observation.focus_translation if observation else None)
            introduction = (
                "Sabaidee! I'm your Lao tutor. We'll take things step by step."
            )
            if intro_focus:
                intro_romanised = self._romanise(intro_focus)
                introduction += (
                    f" Our first focus is “{intro_focus}” ({intro_romanised}) – "
                    f"{intro_translation or 'a friendly greeting.'}"
                )
            introduction = introduction.strip()

        reply_parts = [part for part in [introduction, summary] if part]
        reply = "\n\n".join(reply_parts)
        debug: Dict[str, str] = {"backend": "summary"}

        if self._generator and self._tokenizer:
            prompt = self._format_prompt(history, user_message, focus_phrase, focus_translation)
            prompt += (
                "\nSystem: Craft one or two additional encouraging English sentences with specific pronunciation or tone tips."
                " Always restate the Lao focus phrase with its romanisation and English meaning."
            )
            try:
                outputs = self._generator(
                    prompt,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                generated = outputs[0]["generated_text"]
                llm_tail = generated[len(prompt) :].strip() or generated.strip()
                llm_tail = self._clean_generated_text(llm_tail)
                if llm_tail:
                    reply = f"{summary}\n\n{llm_tail}"
                spoken_text = self._extract_lao_line(reply)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning("Generation failed; using fallback response", exc_info=exc)
                fallback_reply, fallback_spoken = self._fallback_reply(user_message, focus_phrase, focus_translation)
                reply = f"{summary}\n\n{fallback_reply}"
                spoken_text = fallback_spoken or spoken_text
                debug = {"backend": "fallback", "reason": str(exc)}
            else:
                debug = {"backend": "transformers", "model": self._model_name}
                logger.debug(
                    "Generated response via transformers",
                    extra={"reply_chars": len(reply), "spoken": bool(spoken_text)},
                )
        else:
            fallback_reply, fallback_spoken = self._fallback_reply(user_message, focus_phrase, focus_translation)
            reply = f"{summary}\n\n{fallback_reply}" if fallback_reply else summary
            spoken_text = fallback_spoken
            debug = {"backend": "fallback"}
            logger.debug("Using scripted fallback reply")

        if not spoken_text:
            spoken_text = self._extract_lao_line(reply)
        if not spoken_text:
            spoken_text = extract_first_lao_segment(focus_phrase or "") or focus_phrase or None
        if spoken_text and not contains_lao_characters(spoken_text) and focus_phrase and contains_lao_characters(focus_phrase):
            spoken_text = focus_phrase

        updated_history = history[-8:].copy()
        updated_history.append({"role": "user", "content": user_message})
        updated_history.append({"role": "assistant", "content": reply})

        logger.debug(
            "Conversation response prepared",
            extra={
                "history_length": len(updated_history),
                "spoken_text": spoken_text,
                "focus_phrase": focus_phrase,
            },
        )

        return ConversationResult(
            reply_text=reply,
            history=updated_history,
            focus_phrase=focus_phrase,
            focus_translation=focus_translation,
            spoken_text=spoken_text,
            debug=debug,
        )


__all__ = ["ConversationResult", "ConversationService"]
