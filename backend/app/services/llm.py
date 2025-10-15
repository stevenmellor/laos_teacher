"""Conversational LLM orchestration for the Lao tutor."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)

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
        "You are a warm, encouraging Lao language teacher. Always include Lao script, a simple transliteration, "
        "and an English gloss when presenting phrases. Encourage the learner to repeat the Lao focus phrase."
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
                logger.info("Loaded conversational model %s", self._model_name)
            except Exception as exc:  # pragma: no cover - optional failure path
                logger.warning("Conversation model unavailable (%s); falling back to scripted replies", exc)
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
    def _extract_lao_line(text: str) -> Optional[str]:
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if any("຀" <= char <= "໿" for char in candidate):
                return candidate
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
        self, history: ChatHistory, user_message: str, task_id: Optional[str] = None
    ) -> ConversationResult:
        if not user_message.strip():
            raise ValueError("Message must not be empty")

        focus_phrase, focus_translation = self._select_focus_phrase(task_id)

        spoken_text: Optional[str] = None

        if self._generator and self._tokenizer:
            prompt = self._format_prompt(history, user_message, focus_phrase, focus_translation)
            try:
                outputs = self._generator(
                    prompt,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                generated = outputs[0]["generated_text"]
                reply = generated[len(prompt) :].strip() or generated.strip()
                spoken_text = self._extract_lao_line(reply)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.warning("Generation failed (%s); using fallback response", exc)
                reply, spoken_text = self._fallback_reply(user_message, focus_phrase, focus_translation)
                debug = {"backend": "fallback", "reason": str(exc)}
            else:
                debug = {"backend": "transformers", "model": self._model_name}
        else:
            reply, spoken_text = self._fallback_reply(user_message, focus_phrase, focus_translation)
            debug = {"backend": "fallback"}

        if not spoken_text:
            spoken_text = self._extract_lao_line(reply)
        if not spoken_text:
            spoken_text = focus_phrase or None

        updated_history = history[-8:].copy()
        updated_history.append({"role": "user", "content": user_message})
        updated_history.append({"role": "assistant", "content": reply})

        return ConversationResult(
            reply_text=reply,
            history=updated_history,
            focus_phrase=focus_phrase,
            focus_translation=focus_translation,
            spoken_text=spoken_text,
            debug=debug,
        )


__all__ = ["ConversationResult", "ConversationService"]
