"""Export the fine-tuned model as full merged weights. The weight-edit attack needs whole weight matrices,
not a LoRA adapter."""
import json
import os
import shutil

import torch as th

WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".h5", ".msgpack", ".index.json")
LOAD_TIME_KEYS = ("bf16", "fp16", "fp32", "use_flash_attn")     # set by whoever loads the model, not by us


@th.no_grad()
def export_merged(model, tok, base_dir, out_dir, check_ids):
    """Merge the LoRA adapter into the base weights and save a standalone model to out_dir.

    The merge runs in fp32: bf16 keeps about 3 significant digits, too few to add a small LoRA update
    to a weight. The result is saved in fp16, the format of the original Qwen checkpoint.
    Returns (merged model, largest logit difference between adapter model and merged model on check_ids).
    Load the export with AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).
    """
    model.eval().float()
    before = model(input_ids=check_ids, use_cache=False).logits
    merged = model.merge_and_unload()
    after = merged(input_ids=check_ids, use_cache=False).logits
    diff = (before - after).abs().max().item()

    merged.half().save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    _finish_export(base_dir, out_dir)
    return merged, diff


def export_from_adapter(run_dir, check_prompt="Hi."):
    """Redo the export for a finished run, from the adapter saved in run_dir."""
    from peft import PeftModel

    from common.models import load_for_training, model_path
    from config import PROMPT_TEMPLATE

    model, tok = load_for_training()
    model = PeftModel.from_pretrained(model, run_dir)
    check_ids = th.tensor([tok.encode(PROMPT_TEMPLATE.format(instruction=check_prompt))], device=model.device)
    _, diff = export_merged(model, tok, model_path(), os.path.join(run_dir, "merged"), check_ids)
    print(f"merged weights saved to {run_dir}/merged (max logit diff vs adapter model: {diff:.4f})")
    return diff


def _finish_export(base_dir, out_dir):
    """Make out_dir loadable by itself: copy the code and tokenizer files that save_pretrained left out,
    and reset the load-time precision flags in config.json to the base model's values."""
    for name in sorted(os.listdir(base_dir)):
        src, dst = os.path.join(base_dir, name), os.path.join(out_dir, name)
        if name.startswith(".") or name.endswith(WEIGHT_SUFFIXES) or not os.path.isfile(src) or os.path.exists(dst):
            continue
        shutil.copy(src, dst)

    with open(os.path.join(base_dir, "config.json")) as f:
        base_cfg = json.load(f)
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for key in LOAD_TIME_KEYS:
        if key in base_cfg:
            cfg[key] = base_cfg[key]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
