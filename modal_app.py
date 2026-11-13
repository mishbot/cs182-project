"""Modal entry points for the CS 182 project.

Setup on your machine, once:
    pip install modal
    modal setup

Download Qwen into the shared Volume, once per Modal workspace:
    modal run modal_app.py::download_model

Load the model on a GPU and check the activation reader:
    modal run modal_app.py
    modal run modal_app.py --layer-idx 16

Train the defender. Data splits go to /runs/data (written once, then reused by every run);
checkpoints and the final adapter go to /runs/<run-name>:
    modal run modal_app.py::train --run-name dry-run --dry-run      # minutes; checks every code path
    modal run --detach modal_app.py::train --run-name v1            # full run; keeps going if you close the laptop

Read a run's results from your machine, or redo its merged-weights export:
    modal volume get cs182-runs v1/summary.json .
    modal volume get cs182-runs v1/metrics.jsonl .
    modal run modal_app.py::export --run-name v1

Diagnostics:
    modal run modal_app.py::debug_refusals     # raw base-model outputs: batched vs single vs chat
    modal run modal_app.py::debug_padding      # what breaks left-padded batches
"""
from pathlib import Path

import modal

from config import (APP_NAME, GPU, LAYER_IDX, MODEL_ID, MODELS_DIR, MODELS_VOLUME, PYTHON_VERSION,
                    RUNS_DIR, RUNS_VOLUME)

app = modal.App(APP_NAME)
models_vol = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install_from_requirements(str(Path(__file__).parent / "requirements.txt"))
    .env({"MODEL_CACHE_DIR": MODELS_DIR})  # config.py reads this as CACHE_DIR
    .add_local_python_source("config", "common", "defense")
)


@app.function(image=image, volumes={MODELS_DIR: models_vol}, timeout=60 * 60)
def download_model():
    from modelscope import snapshot_download

    model_dir = snapshot_download(MODEL_ID, cache_dir=MODELS_DIR)
    models_vol.commit()
    print(f"Saved {MODEL_ID} to {model_dir}")


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: models_vol}, timeout=30 * 60)
def smoke_test(layer_idx: int = LAYER_IDX):
    import torch as th

    from common.models import LLM, prompt_acts
    from config import DEVICE, EOI_STR  # DEVICE is evaluated inside the container, so it is "cuda"

    print(f"device: {DEVICE} ({th.cuda.get_device_name(0)})")
    llm = LLM(model_id=MODEL_ID)
    model, tok = llm.model, llm.tokenizer
    n_eoi = len(tok.encode(EOI_STR))
    print(f"layers: {model.config.num_hidden_layers}, reading hidden_states[{layer_idx}] "
          f"at {n_eoi} post-instruction positions")
    print("chat:", llm.chat("Hey bud.", keep_history=False))

    # The same prompts as one right-padded batch, then one at a time with no padding.
    # If the post-instruction positions are read correctly, the activations match.
    prompts = ["Hi.", "Write a four-line poem about the sea."]
    batched = prompt_acts(model, tok, prompts, [layer_idx], n_eoi, batch_size=len(prompts))
    single = prompt_acts(model, tok, prompts, [layer_idx], n_eoi, batch_size=1)

    rel_err = ((batched - single).norm(dim=-1) / single.norm(dim=-1)).max().item()
    print(f"padding check: max relative error {rel_err:.1e} ->", "OK" if rel_err < 1e-2 else "MISMATCH")


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: models_vol, RUNS_DIR: runs_vol},
              timeout=6 * 60 * 60)
def train(run_name: str, dry_run: bool = False):
    from config import DEFENSE_CFG, DRY_RUN_CFG
    from defense.train import train as train_defense

    cfg = dict(DEFENSE_CFG)
    if dry_run:
        cfg.update(DRY_RUN_CFG)
    cfg["data_dir"] = f"{RUNS_DIR}/data"         # shared splits, so every run and the attack see the same ones
    cfg["out_dir"] = f"{RUNS_DIR}/{run_name}"
    print("config:", cfg)
    try:
        train_defense(cfg)
    finally:
        runs_vol.commit()                       # keep whatever was saved, even if the run fails


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: models_vol, RUNS_DIR: runs_vol}, timeout=30 * 60)
def export(run_name: str):
    """Redo the merged-weights export for a finished run, from its saved adapter."""
    from defense.export import export_from_adapter

    try:
        export_from_adapter(f"{RUNS_DIR}/{run_name}")
    finally:
        runs_vol.commit()


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: models_vol, RUNS_DIR: runs_vol}, timeout=30 * 60)
def debug_refusals(n: int = 8):
    """Why does the base model 'refuse' only ~7% of harmful prompts? Compare three generation paths."""
    from common.data import load_instructions
    from common.evaluate import is_refusal
    from common.models import generate, load_for_training
    from config import PROMPT_TEMPLATE

    model, tok = load_for_training()
    prompts = load_instructions(f"{RUNS_DIR}/data/harmful_val.json")[:n]

    batched = generate(model, tok, prompts, max_new_tokens=48, batch_size=n)   # left-padded batch
    single = generate(model, tok, prompts, max_new_tokens=48, batch_size=1)    # no padding
    print("template:", repr(PROMPT_TEMPLATE))
    print("first prompt tokens:", tok(PROMPT_TEMPLATE.format(instruction=prompts[0]))["input_ids"])
    print("pad:", tok.pad_token, tok.pad_token_id, "| padding_side:", tok.padding_side)
    for p, b, s1 in zip(prompts, batched, single):
        print("=" * 80)
        print("PROMPT :", p)
        print(f"BATCHED [{is_refusal(b)}]:", repr(b))
        print(f"SINGLE  [{is_refusal(s1)}]:", repr(s1))
        r, _ = model.chat(tok, p, history=None)                               # Qwen's own chat path
        print(f"CHAT    [{is_refusal(r)}]:", repr(r[:200]))
    print("refusal rate  batched:", sum(map(is_refusal, batched)) / n,
          "| single:", sum(map(is_refusal, single)) / n)


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: models_vol, RUNS_DIR: runs_vol}, timeout=30 * 60)
def debug_padding(n: int = 8):
    """Find what breaks left-padded batches: precision, pad token, attention kernel, or the KV cache."""
    from common.data import load_instructions
    from common.debug import padding_report

    padding_report(load_instructions(f"{RUNS_DIR}/data/harmful_val.json")[:n])


@app.local_entrypoint()
def main(layer_idx: int = LAYER_IDX):
    smoke_test.remote(layer_idx=layer_idx)
