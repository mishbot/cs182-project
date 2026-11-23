"""Original Qwen model access, local loading, and canonical single-turn ChatML inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ._model_config import load_local_model_config
from ._types import PathInput

if TYPE_CHECKING:
    from torch.nn import Embedding, Linear, Module, ModuleList
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerBase,
    )
    from transformers.tokenization_utils_base import BatchEncoding

SYSTEM_MESSAGE = "You are a helpful assistant."
PROMPT_FORMAT = "original-qwen-chatml-v1"


def load_tokenizer(model_dir: PathInput) -> PreTrainedTokenizer:
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    formatter = get_class_from_dynamic_module(
        "qwen_generation_utils.make_context", str(model_dir), local_files_only=True
    )
    vars(tokenizer)["_cs182_make_context"] = formatter
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return tokenizer


def format_chat(tokenizer: PreTrainedTokenizerBase, prompt: str) -> BatchEncoding:
    """Keep upstream token IDs: re-tokenizing the rendered text changes reserved-token handling."""
    from transformers.tokenization_utils_base import BatchEncoding

    formatter = cast(Callable[..., tuple[str, list[int]]], vars(tokenizer)["_cs182_make_context"])
    _, tokens = formatter(
        tokenizer,
        prompt,
        history=[],
        system=SYSTEM_MESSAGE,
        max_window_size=8192,
        chat_format="chatml",
    )
    return BatchEncoding({"input_ids": tokens, "attention_mask": [1] * len(tokens)})


def load_model(
    model_dir: PathInput,
    device: str,
    *,
    config: PretrainedConfig | None = None,
    preserve_dtype: bool = False,
    for_training: bool = False,
) -> PreTrainedModel:
    import torch
    from transformers import AutoModelForCausalLM

    config = config or load_local_model_config(model_dir)
    dtype = torch.float16 if device == "cuda" else torch.float32
    if for_training:
        dtype = (
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        )
    if preserve_dtype:
        from pathlib import Path

        from safetensors import safe_open

        first = next(iter(sorted(Path(model_dir).glob("*.safetensors"))))
        with safe_open(first, framework="pt") as weights:
            stored = weights.get_slice(next(iter(weights.keys()))).get_dtype()
        dtype = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
        }[stored]
    config.bf16, config.fp16, config.fp32 = (
        dtype == torch.bfloat16,
        dtype == torch.float16,
        dtype in {torch.float32, torch.float64},
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    return model.to(device).eval()


def backbone(model: PreTrainedModel) -> Module:
    return model.get_submodule("transformer")


def decoder_blocks(transformer: Module) -> ModuleList:
    return cast("ModuleList", transformer.get_submodule("h"))


def token_embeddings(model: PreTrainedModel) -> Embedding:
    return cast("Embedding", model.get_submodule("transformer.wte"))


def output_head(model: PreTrainedModel) -> Linear:
    return cast("Linear", model.get_submodule("lm_head"))


def residual_writers(model: PreTrainedModel) -> list[tuple[str, Linear]]:
    return [
        (f"transformer.h.{index}.{suffix}", cast("Linear", layer.get_submodule(suffix)))
        for index, layer in enumerate(decoder_blocks(backbone(model)))
        for suffix in ("attn.c_proj", "mlp.c_proj")
    ]
