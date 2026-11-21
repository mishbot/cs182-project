"""Local original-Qwen configuration for the supported model runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import PathInput

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


def load_local_model_config(model_dir: PathInput) -> PretrainedConfig:
    from transformers import AutoConfig

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    if raw.get("model_type") != "qwen" or raw.get("quantization_config"):
        raise ValueError(
            "The model runtime supports unquantized original Qwen (model_type='qwen')."
        )
    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    config.use_flash_attn = False
    config.use_cache_quantization = False
    config.use_cache_kernel = False
    return config
