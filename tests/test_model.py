"""Integration tests using real model inference with a fixed seed."""

import os
import unittest

from src.model import get_response, load_model


TEST_SEED = int(os.environ.get("TEST_SEED", "6"))


class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load_model()

    def ask(self, prompt, tokens):
        return get_response(
            prompt,
            self.model,
            self.tokenizer,
            max_length=tokens,
            seed=TEST_SEED,
        )

    def test_factual_query(self):
        response = self.ask("What is the capital of France?", 60)
        self.assertIn("paris", response.lower())

    def test_code_generation(self):
        response = self.ask(
            "Write a Python function named factorial that calculates the factorial of a non-negative integer.",
            220,
        )
        self.assertIn("def", response)
        self.assertIn("factorial", response.lower())
        self.assertIn("return", response)
        self.assertTrue(
            "for" in response or "range" in response or "factorial(n - 1)" in response
        )

    def test_simple_reasoning(self):
        response = self.ask(
            "If I have 3 apples and give 1 to Mary, how many do I have left? Answer briefly.",
            70,
        )
        self.assertRegex(response, r"\b2\b|\btwo\b", response)


class TestInputValidation(unittest.TestCase):
    def test_blank_prompt_is_rejected_before_inference(self):
        with self.assertRaises(ValueError):
            get_response("   ", None, None)


if __name__ == "__main__":
    unittest.main()
