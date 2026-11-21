"""Generate responses using a locally archived chat model."""

from __future__ import annotations

from typing import cast

from ._model_config import load_local_model_config
from ._types import Device, PathInput
from .qwen_model import format_chat, load_model, load_tokenizer


def generate(model_dir: PathInput, prompt: str, max_new_tokens: int, device: Device) -> str:
    """Return the generated response, loading weights only from model_dir."""
    if not prompt.strip():
        raise ValueError("--prompt must not be empty")
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    from copy import deepcopy

    import torch
    from transformers import GenerationConfig

    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; run Docker with --gpus all or select --device cpu."
        )
    config = load_local_model_config(model_dir)
    tokenizer = load_tokenizer(model_dir)
    encoded = format_chat(tokenizer, prompt)
    model = load_model(model_dir, selected, config=config)
    inputs = {name: torch.tensor([value], device=selected) for name, value in encoded.items()}
    input_length = inputs["input_ids"].shape[1]
    if input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus --max-new-tokens exceeds the model context window.")
    generation = deepcopy(model.generation_config) or GenerationConfig.from_model_config(config)
    generation.do_sample = False
    generation.top_p, generation.top_k, generation.temperature = 1.0, 50, 1.0
    with torch.inference_mode():
        result = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            generation_config=generation,
            stop_words_ids=[
                [cast(int, tokenizer.convert_tokens_to_ids("<|im_end|>"))],
                [cast(int, tokenizer.convert_tokens_to_ids("<|im_start|>"))],
            ],
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(result[0, input_length:], skip_special_tokens=True)
