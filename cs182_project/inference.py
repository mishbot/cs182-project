"""Generate responses using a locally archived chat model."""

from __future__ import annotations

from ._model_config import load_local_model_config
from ._types import Device, PathInput


def generate(model_dir: PathInput, prompt: str, max_new_tokens: int, device: Device) -> str:
    """Return the generated response, loading weights only from model_dir."""
    if not prompt.strip():
        raise ValueError("--prompt must not be empty")
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; run Docker with --gpus all or select --device cpu."
        )
    config = load_local_model_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16 if selected == "cuda" else torch.float32,
        )
        .to(selected)
        .eval()
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(selected)
    input_length = inputs["input_ids"].shape[1]
    if input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus --max-new-tokens exceeds the model context window.")
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(result[0, input_length:], skip_special_tokens=True)
