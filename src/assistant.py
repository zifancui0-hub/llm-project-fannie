"""LingoBridge: a themed translation assistant built on the instruct model.

LingoBridge turns the generic open-weight chat model into a focused
translation application. It wraps ``src.model`` with a translator persona
system prompt and a structured task prompt (source language, target
language, tone), so the same model behaves as a dedicated translator
instead of a general chatbot.
"""

from __future__ import annotations

from typing import List, Optional

SYSTEM_PROMPT = (
    "You are LingoBridge, an expert professional translator. "
    "Translate the user's text accurately and naturally into the requested "
    "target language, matching the requested tone. When the target is "
    "中文（繁體）, use Traditional Chinese characters only; when it is "
    "中文（簡體）, use Simplified Chinese characters only. Preserve the "
    "original meaning, formatting, numbers, and proper nouns. Output only "
    "the translation itself: no explanations, no notes, no quotation marks, "
    "and no repetition of the source text. If the source text is already "
    "in the target language, return it unchanged. If the text is ambiguous, "
    "choose the most common, natural interpretation."
)

SUPPORTED_LANGUAGES = [
    "Auto-detect",
    "English",
    "中文（繁體）",
    "中文（簡體）",
    "日本語",
    "한국어",
    "Português",
    "Français",
    "Deutsch",
    "Español",
]

FORMALITY_LEVELS = ["Neutral", "Formal", "Casual"]

Message = dict


class TranslatorAssistant:
    """Task-oriented translator over a loaded instruct model."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def build_messages(
        self,
        text: str,
        source_language: str = "Auto-detect",
        target_language: str = "English",
        formality: str = "Neutral",
    ) -> List[Message]:
        """Assemble the system prompt plus one structured translation task."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if target_language not in SUPPORTED_LANGUAGES or target_language == "Auto-detect":
            raise ValueError("target_language must be a concrete supported language")
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError("source_language must be one of SUPPORTED_LANGUAGES")
        if formality not in FORMALITY_LEVELS:
            raise ValueError("formality must be one of FORMALITY_LEVELS")

        task = (
            f"Translate the following text from {source_language} to "
            f"{target_language}. Use a {formality.lower()} tone. "
            f"Translate every word; leave nothing untranslated.\n\n"
            f"Text:\n{text.strip()}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

    def translate(
        self,
        text: str,
        source_language: str = "Auto-detect",
        target_language: str = "English",
        formality: str = "Neutral",
        max_length: int = 512,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> str:
        """Translate text and return only the model's translation.

        A low default temperature keeps translations stable; only the tokens
        generated after the formatted prompt are returned, so the instruction
        never leaks into the output.
        """
        import torch  # Imported lazily so UI/tests can run without a GPU stack.

        messages = self.build_messages(text, source_language, target_language, formality)
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        )
        device = getattr(self.model, "device", None) or next(
            self.model.parameters()
        ).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_tokens = encoded["input_ids"].shape[1]

        if seed is not None:
            torch.manual_seed(seed)

        generation = {
            "max_new_tokens": max_length,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation.update(temperature=temperature, top_p=0.9)

        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation)

        new_tokens = output[0, prompt_tokens:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
