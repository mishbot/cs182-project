"""Generate responses using a locally archived chat model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from ._model_config import load_local_model_config
from ._types import Device, PathInput
from .qwen_model import format_chat, load_model, load_tokenizer


def generate(model_dir: PathInput, prompt: str, max_new_tokens: int, device: Device) -> str:
    """Return the generated response, loading weights only from model_dir."""
    return generate_many(model_dir, [prompt], max_new_tokens, device)[0]


def generate_many(
    model_dir: PathInput, prompts: Sequence[str], max_new_tokens: int, device: Device
) -> list[str]:
    """Generate one response per prompt, loading the model once."""
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("--prompt must not be empty")
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not prompts:
        return []
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
    model = load_model(model_dir, selected, config=config)
    generation = deepcopy(model.generation_config) or GenerationConfig.from_model_config(config)
    generation.do_sample = False
    generation.top_p, generation.top_k, generation.temperature = 1.0, 50, 1.0
    stop_words = [
        [cast(int, tokenizer.convert_tokens_to_ids("<|im_end|>"))],
        [cast(int, tokenizer.convert_tokens_to_ids("<|im_start|>"))],
    ]
    responses = []
    with torch.inference_mode():
        for index, prompt in enumerate(prompts, start=1):
            encoded = format_chat(tokenizer, prompt)
            inputs = {name: torch.tensor([value], device=selected) for name, value in encoded.items()}
            input_length = inputs["input_ids"].shape[1]
            if input_length + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError(f"Prompt {index} plus --max-new-tokens exceeds the model context window.")
            result = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                generation_config=generation,
                stop_words_ids=stop_words,
                pad_token_id=tokenizer.pad_token_id,
            )
            responses.append(tokenizer.decode(result[0, input_length:], skip_special_tokens=True))
    return responses
