"""LingoBridge web app: a themed translation UI for the Assignment 1B model.

Run the real application (downloads the open-weight model on first use):

    python app.py

Run a lightweight UI preview without torch or the model (for layout checks):

    python app.py --mock
"""

from __future__ import annotations

import argparse
import time

import gradio as gr

from src.assistant import FORMALITY_LEVELS, SUPPORTED_LANGUAGES

APP_TITLE = "LingoBridge — Local AI Translator"
APP_DESCRIPTION = (
    "A themed translation app powered by an open-weight instruct language "
    "model running entirely on your own machine — no cloud API, no data "
    "leaving your computer. Paste text, pick a language pair and a tone, "
    "and get an instant translation."
)

EXAMPLES = [
    ["The nightlife in Macau blends Portuguese heritage with modern energy."],
    ["人工智能正在改變我們學習語言的方式。"],
    ["I would be grateful if you could confirm the meeting by Friday."],
    ["這間茶餐廳嘅豬扒包好好味，你一定要試吓！"],
]

# Language pairs that make sense for the one-click swap button.
SWAPPABLE = {"English", "中文（繁體）", "中文（簡體）"}


class MockTranslator:
    """Stand-in for TranslatorAssistant so the UI can be previewed offline."""

    def translate(self, text, source_language="Auto-detect",
                  target_language="English", formality="Neutral",
                  max_length=512, temperature=0.3, seed=None):
        time.sleep(0.3)  # Simulate generation latency.
        return (
            f"[mock mode — no model loaded]\n"
            f"{source_language} → {target_language} ({formality}): {text}\n"
            f"Run `python app.py` without --mock for real translations."
        )


def create_demo(translator) -> gr.Blocks:
    """Build the Gradio UI around any object with a translate() method."""

    def run_translation(text, source, target, formality, temperature, seed_text):
        seed = int(seed_text) if str(seed_text).strip() else None
        return translator.translate(
            text,
            source_language=source,
            target_language=target,
            formality=formality,
            temperature=float(temperature),
            seed=seed,
        )

    def swap_languages(source, target):
        if source in SWAPPABLE and target in SWAPPABLE:
            return target, source
        return source, target

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# 🌉 {APP_TITLE}\n\n{APP_DESCRIPTION}")

        with gr.Row():
            source = gr.Dropdown(
                SUPPORTED_LANGUAGES, value="Auto-detect", label="From"
            )
            swap_button = gr.Button("⇄ Swap", scale=1, min_width=80)
            target = gr.Dropdown(
                [l for l in SUPPORTED_LANGUAGES if l != "Auto-detect"],
                value="English",
                label="To",
            )

        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(
                    label="Source text", lines=8,
                    placeholder="Type or paste the text to translate…",
                )
                formality = gr.Radio(
                    FORMALITY_LEVELS, value="Neutral", label="Tone"
                )
                translate_button = gr.Button("Translate", variant="primary")
            with gr.Column():
                output_box = gr.Textbox(label="Translation", lines=8, interactive=False)
                copy_hint = gr.Markdown(
                    "*Select the text above to copy your translation.*"
                )

        with gr.Accordion("Generation settings", open=False):
            temperature = gr.Slider(
                0.0, 1.0, value=0.3, step=0.05,
                label="Temperature (low = stable, high = creative)",
            )
            seed_text = gr.Textbox(
                label="Seed (optional, for reproducible output)", value=""
            )

        gr.Examples(examples=EXAMPLES, inputs=input_box)

        inputs = [input_box, source, target, formality, temperature, seed_text]
        translate_button.click(run_translation, inputs=inputs, outputs=output_box)
        input_box.submit(run_translation, inputs=inputs, outputs=output_box)
        swap_button.click(swap_languages, inputs=[source, target], outputs=[source, target])

        gr.Markdown(
            "*LingoBridge runs a quantized open-weight LLM locally; "
            "for legal, medical, or safety-critical documents, use a certified translator.*"
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock", action="store_true", help="Preview the UI without loading the model"
    )
    parser.add_argument("--model", default=None, help="Hugging Face model ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.mock:
        translator = MockTranslator()
    else:
        from src.assistant import TranslatorAssistant
        from src.model import load_model

        print("Loading the language model (first run downloads it) ...")
        model, tokenizer = load_model(args.model)
        translator = TranslatorAssistant(model, tokenizer)
        print(f"Model ready on {model.device}. Starting the web UI ...")

    demo = create_demo(translator)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
