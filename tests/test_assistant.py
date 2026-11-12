"""Lightweight tests for the LingoBridge translator persona logic.

These tests exercise prompt assembly and input validation without loading
the real model, so they run fast on any machine.
"""

import unittest

from src.assistant import (
    FORMALITY_LEVELS,
    SUPPORTED_LANGUAGES,
    SYSTEM_PROMPT,
    TranslatorAssistant,
)


class FakeTokenizer:
    """Minimal tokenizer stand-in; only the chat template is needed here."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class TestBuildMessages(unittest.TestCase):
    def setUp(self):
        self.translator = TranslatorAssistant(model=None, tokenizer=FakeTokenizer())

    def test_system_prompt_comes_first(self):
        messages = self.translator.build_messages("Hello", target_language="中文（繁體）")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertIn("translator", SYSTEM_PROMPT.lower())

    def test_task_prompt_contains_languages_tone_and_text(self):
        messages = self.translator.build_messages(
            "Good morning",
            source_language="English",
            target_language="Français",
            formality="Formal",
        )
        task = messages[-1]["content"]
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("English", task)
        self.assertIn("Français", task)
        self.assertIn("formal", task)
        self.assertIn("Good morning", task)

    def test_supported_language_list_has_no_duplicate_targets(self):
        targets = [l for l in SUPPORTED_LANGUAGES if l != "Auto-detect"]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertIn("English", targets)

    def test_blank_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.build_messages("   ", target_language="English")

    def test_auto_detect_target_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.build_messages("Hello", target_language="Auto-detect")

    def test_unknown_language_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.build_messages("Hello", source_language="Klingon")

    def test_unknown_formality_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.build_messages(
                "Hello", target_language="English", formality="Pirate"
            )

    def test_all_formality_levels_accepted(self):
        for level in FORMALITY_LEVELS:
            messages = self.translator.build_messages(
                "Hi", target_language="日本語", formality=level
            )
            self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
