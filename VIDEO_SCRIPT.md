# Video Recording Script — LingoBridge (p2319777)

Project title changed from `LLM-Project` to **LingoBridge**. Rename files consistently:

- `p2319777_LingoBridge_1-installation.mp4`
- `p2319777_LingoBridge_2-implementation.mp4`
- `p2319777_LingoBridge_3-testing.mp4`

Only **Video 2 must be re-recorded** (the app is new). Videos 1 and 3 can stay as they are; optionally re-record the short segments noted below.

---

## Video 2 — Implementation & Demo (re-record, ~4–5 min)

Recording tool: macOS `Cmd + Shift + 5` → "Record Entire Screen".

### Shot 1 — Code walkthrough, `src/assistant.py` (~2 min)

Open in VS Code, scroll slowly, and say roughly:

1. "This is my themed application: LingoBridge, a local AI translator."
2. Point at `SYSTEM_PROMPT`: "The translator persona — output only the translation, Traditional vs Simplified Chinese is controlled here."
3. Point at `build_messages()`: "The app builds a structured task prompt — source language, target language, tone — so the user never has to write prompts."
4. Point at `translate()`: "Only tokens generated after the prompt are returned, so the instruction never leaks into the output. Temperature defaults to 0.3 for stable translations."

### Shot 2 — Code walkthrough, `app.py` (~1 min)

1. "Gradio UI: language dropdowns, a swap button, tone radio buttons, and generation settings."
2. Point at `--mock` flag: "The UI can be previewed without the model."

### Shot 3 — Live demo (~2 min)

In the terminal:

```bash
cd llm-project-fannie
source venv/bin/activate
python app.py
```

Open http://127.0.0.1:7860 and demonstrate these three (verified outputs, seed 6):

| # | Input | Settings | Expected output |
| --- | --- | --- | --- |
| 1 | `The nightlife in Macau blends Portuguese heritage with modern energy.` | Auto-detect → 中文（繁體）, Neutral | 夜生活在澳门融合了葡萄牙風味與現代活力。 |
| 2 | `人工智能正在改變我們學習語言的方式。` | Auto-detect → English, Neutral | Artificial intelligence is transforming how we learn languages. |
| 3 | `Where is the nearest bus stop?` | Auto-detect → 日本語, Neutral | どこに最近のバス停がありますか？ |

Then click ⇄ Swap once and show the Formal/Casual tone toggle on any sentence.

---

## Video 1 — Installation (optional touch-up)

If you re-record any part: `requirements.txt` now includes `gradio`, so show `pip install -r requirements.txt` installing it. Everything else is unchanged.

## Video 3 — Testing (optional addition)

After the existing segments, you can add 10 seconds:

```bash
python -m unittest tests.test_assistant -v
```

"These eight tests check the translator's prompt assembly and validation without loading the model."

---

## Known limitation to mention (optional, shows critical thinking)

The 1.5B model sometimes mixes Simplified characters (e.g. 澳门) into Traditional Chinese output. This is a model-capacity limitation, documented honestly in the demo.
