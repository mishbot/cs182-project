"""Global parameters for the CS 182 project.

common/, defense/ and modal_app.py import from here.
"""
import os

import torch as th

# Model
MODEL_ID = "qwen/Qwen-1_8B-Chat"
PAD_TOKEN = "<|extra_0|>"       # Qwen-1 has no pad token; this one is distinct from <|endoftext|>
SYSTEM_PROMPT = "You are a helpful assistant."   # used by LLM.chat only

# Same template Arditi et al. use for Qwen (no system prompt). Use this one
# template everywhere: training, the attack, and evaluation.
PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
EOI_STR = "<|im_end|>\n<|im_start|>assistant\n"      # post-instruction tokens

# Device and model cache
DEVICE = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR")  # "/models" on Modal; None uses the default modelscope cache

# Smoke test
LAYER_IDX = 12      # layer the Modal smoke test reads; index into hidden_states
                    # (0 is the embeddings, i is the input to block i)

# Defender training (defense/train.py)
DEFENSE_CFG = dict(
    data_dir="data",        # prepare_data() downloads and writes every split here
    n_harmless_train=2000,  # sampled from the (much larger) harmless_train split
    k=4,                    # leave the last k residual layers unobfuscated; sweep {0, 2, 4, 8}
    batch_per_class=16,     # raise to 32+ if memory allows (lower noise floor in L_obf)
    steps=1500,
    lr=1e-4,
    alpha=1.0,
    lam_max=1.0,            # check initial L_obf vs L_refuse in the logs and adjust
    warmup_steps=100,
    ramp_steps=300,
    lora_r=32,
    lora_alpha=64,
    grad_checkpointing=True,
    log_every=20,
    eval_every=250,
    n_val=128,
    out_dir="qwen_obfuscated_lora",
    export_merged=True,     # also save full merged fp16 weights to <out_dir>/merged for the attack
    seed=0,
)

# Overrides for a short end-to-end check: reaches full lambda, one mid-run eval and
# checkpoint, the final eval, and the fresh-probe AUROC. Minutes instead of hours.
DRY_RUN_CFG = dict(
    steps=20,
    warmup_steps=5,
    ramp_steps=5,
    log_every=2,
    eval_every=10,
    n_harmless_train=64,
    n_val=32,
)

# Modal
APP_NAME = "cs182project"
GPU = "A100-80GB"               # full fine-tuning of 1.8B params with Adam needs ~30 GB before activations
PYTHON_VERSION = "3.11"         # matches the local environment
MODELS_VOLUME = "cs182-models"  # Volume that caches the Qwen weights
MODELS_DIR = "/models"          # where that Volume is mounted inside Modal containers
RUNS_VOLUME = "cs182-runs"      # Volume for the prepared data splits and training outputs
RUNS_DIR = "/runs"              # where that Volume is mounted inside Modal containers
