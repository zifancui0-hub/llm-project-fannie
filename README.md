# LLM Project — Fannie (p2319777)

Assignment 1B: loading, interacting with, and evaluating an open-weight instruction-tuned language model.

The project uses **Qwen/Qwen2.5-1.5B-Instruct** by default. It is openly downloadable and does not require accepting a gated model licence. The implementation selects CUDA, Apple MPS, or CPU automatically and supports reproducible generation for testing.

## Repository

https://github.com/zifancui0-hub/llm-project-fannie

## Installation

Requirements: Python 3.8 or newer (Python 3.10+ recommended), Git, and approximately 4 GB of free disk space for the model and its download cache.

```bash
git clone https://github.com/zifancui0-hub/llm-project-fannie.git
cd llm-project-fannie
python3 -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Hugging Face token is needed for the default Qwen model. If you switch to a gated model such as Gemma, accept its licence on Hugging Face and authenticate with:

```bash
hf auth login
```

The older `huggingface-cli login` command is deprecated in current Hugging Face tooling.

## Hardware-aware model loading

| Runtime | Loading strategy |
| --- | --- |
| NVIDIA CUDA + bitsandbytes | 4-bit NF4 with double quantization and FP16 computation |
| NVIDIA CUDA without bitsandbytes | FP16 with automatic device mapping |
| Apple Silicon MPS | FP16 on the Metal Performance Shaders device |
| CPU | FP32 portable fallback |

Set `LLM_MODEL_ID` to override the default model globally, or pass `--model` for one run.

## Demo

Interactive mode:

```bash
python demo.py --seed 6 --max-new-tokens 200
```

Type `quit` to exit. One or more non-interactive prompts can also be supplied:

```bash
python demo.py --seed 6 --max-new-tokens 100 \
  --prompt "What is the capital of France?" \
  --prompt "Write a short haiku about artificial intelligence."
```

`--seed` makes each independent prompt reproducible. `--temperature 0` selects greedy decoding; the default `0.7` uses top-p sampling.

## Tests

Run from the repository root after installing the requirements:

```bash
python -m unittest discover tests -v
# or
pytest -q
```

The test suite loads the model once, checks factual recall, Python factorial generation, explicitly worded arithmetic reasoning, and blank-input validation. Model assertions use a fixed seed (`TEST_SEED`, default `6`). The first run downloads the model, so it can take substantially longer than later runs.

On the recorded Apple M5 environment, both runners passed all four tests. See [RESULTS.md](RESULTS.md) for the observed test output.

## Qualitative evaluation

See [RESULTS.md](RESULTS.md) for verbatim outputs and analysis of factual accuracy, creativity, reasoning, bias and safety, and a hallucination failure case. Reproduce the six recorded prompts (five qualitative prompts plus factorial generation) with:

```bash
python -m scripts.evaluate --seeds 6 --output outputs/qualitative-results.json
```

The JSON contains full responses and runtime metadata. The generated `outputs/` directory is excluded from Git.

## Project structure

```text
llm-project-fannie/
├── src/
│   ├── __init__.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
├── scripts/
│   ├── __init__.py
│   └── evaluate.py
├── videos/
│   └── README.md
├── demo.py
├── requirements.txt
├── RESULTS.md
├── README.md
└── .gitignore
```

## Videos

| Required demonstration | File |
| --- | --- |
| Installation | [p2319777_LLM-Project_1-installation.mp4](videos/p2319777_LLM-Project_1-installation.mp4) |
| Implementation and demo | [p2319777_LLM-Project_2-implementation.mp4](videos/p2319777_LLM-Project_2-implementation.mp4) |
| Testing and results | [p2319777_LLM-Project_3-testing.mp4](videos/p2319777_LLM-Project_3-testing.mp4) |

## Submission checklist

- [x] Original `load_model()` and `get_response()` implementation
- [x] Open-weight instruct model with hardware-aware quantization/loading
- [x] Interactive `demo.py`
- [x] Unit tests for factual and code-generation behaviour
- [x] Real model outputs documented in `RESULTS.md`
- [x] Unit tests pass locally with `unittest` and `pytest`
- [ ] Three MP4 videos posted and linked above
- [ ] Public GitHub repository synchronized with the final submission
