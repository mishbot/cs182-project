"""Modal entry points for the CS 182 project.

Setup on your machine, once:
    pip install modal
    modal setup

Download Qwen into the shared Volume, once per Modal workspace:
    modal run modal_app.py::download_model

Load the model on a GPU and check the activation reader:
    modal run modal_app.py
    modal run modal_app.py --layer-idx 16
"""
from pathlib import Path

import modal

from config import APP_NAME, GPU, LAYER_IDX, MODEL_ID, MODELS_DIR, MODELS_VOLUME, PYTHON_VERSION

app = modal.App(APP_NAME)
models_vol = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)

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


@app.local_entrypoint()
def main(layer_idx: int = LAYER_IDX):
    smoke_test.remote(layer_idx=layer_idx)
