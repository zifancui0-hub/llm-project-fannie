"""Hardware-aware loading and reproducible inference for an instruct LLM."""

from __future__ import annotations

import importlib.util
import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_ID = os.environ.get(
    "LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"
)


def _pick_device() -> str:
    """Return the fastest supported inference device on this machine."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _cuda_load_options() -> dict:
    """Use NF4 when bitsandbytes is available, with a portable FP16 fallback."""
    if importlib.util.find_spec("bitsandbytes") is None:
        return {"device_map": "auto", "torch_dtype": torch.float16}

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return {
        "device_map": "auto",
        "quantization_config": quantization,
        "torch_dtype": torch.float16,
    }


def load_model(model_id: Optional[str] = None) -> Tuple[object, object]:
    """Load an open-weight instruct model and its tokenizer.

    CUDA uses 4-bit NF4 quantization when bitsandbytes is installed. Apple
    Silicon uses FP16 on MPS, while CPU inference remains in FP32. Set the
    LLM_MODEL_ID environment variable or pass model_id to try another model.
    """
    selected_model = model_id or DEFAULT_MODEL_ID
    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained(selected_model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        options = _cuda_load_options()
        model = AutoModelForCausalLM.from_pretrained(selected_model, **options)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            selected_model, low_cpu_mem_usage=True
        )

    model.eval()
    return model, tokenizer


def get_response(
    prompt: str,
    model: object,
    tokenizer: object,
    max_length: int = 150,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> str:
    """Generate only the assistant's reply to one user prompt.

    max_length is the maximum number of *new* tokens. A temperature of zero
    selects greedy decoding; positive values enable top-p sampling. Supplying
    a seed makes each independent prompt reproducible for tests and demos.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    if seed is not None:
        torch.manual_seed(seed)

    conversation = [{"role": "user", "content": prompt.strip()}]
    formatted = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )
    device = getattr(model, "device", None) or next(model.parameters()).device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_tokens = encoded["input_ids"].shape[1]

    generation = {
        "max_new_tokens": max_length,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        generation.update(temperature=temperature, top_p=0.9)

    with torch.inference_mode():
        output = model.generate(**encoded, **generation)

    new_tokens = output[0, prompt_tokens:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
