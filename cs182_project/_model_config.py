"""Local model configuration compatibility for the pinned Transformers runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import PathInput

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig


def load_local_model_config(model_dir: PathInput) -> PretrainedConfig:
    """Normalize a disabled Qwen sliding window without modifying the snapshot.

    Transformers 4.51.3 checks the window length, ignoring use_sliding_window,
    both in its warning and some causal-mask paths. Disabled windows should be
    None. Preserve the window when the checkpoint actually enables it.
    """
    from transformers.models.auto.configuration_auto import AutoConfig

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type == "qwen2" and getattr(config, "use_sliding_window", True) is False:
        config.sliding_window = None
    return config
