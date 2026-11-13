"""Defender fine-tuning loop. Run from the project root: python -m defense.train"""
import random

import torch as th
from peft import LoraConfig, get_peft_model

from common.data import collate, load_instructions, prepare_data
from common.evaluate import is_refusal
from common.models import eoi_positions, gather_resid, generate, load_for_training, prompt_acts
from config import DEFENSE_CFG, EOI_STR
from defense.data import make_examples
from defense.evaluate import evaluate, probe_auroc_per_layer
from defense.losses import mean_gap_loss, refusal_loss, retain_loss

def training_step(model, harm_batch, benign_batch, step, cfg):
    out_h = model(input_ids=harm_batch["input_ids"], attention_mask=harm_batch["attention_mask"],
                  output_hidden_states=True, use_cache=False)
    out_b = model(input_ids=benign_batch["input_ids"], attention_mask=benign_batch["attention_mask"],
                  output_hidden_states=True, use_cache=False)

    L_refuse = refusal_loss(out_h.logits, harm_batch)
    L_retain = retain_loss(model, out_b.logits, benign_batch)

    pos_h = eoi_positions(harm_batch["prompt_len"], cfg["n_eoi"])
    pos_b = eoi_positions(benign_batch["prompt_len"], cfg["n_eoi"])
    acts_h = gather_resid(out_h.hidden_states, cfg["layers"], pos_h)
    acts_b = gather_resid(out_b.hidden_states, cfg["layers"], pos_b)
    L_obf = mean_gap_loss(acts_h, acts_b)

    t = (step - cfg["warmup_steps"]) / cfg["ramp_steps"]
    lam = cfg["lam_max"] * min(max(t, 0.0), 1.0)

    loss = L_refuse + cfg["alpha"] * L_retain + lam * L_obf
    logs = {"loss": loss.item(), "L_refuse": L_refuse.item(), "L_retain": L_retain.item(),
            "L_obf": L_obf.item(), "lambda": lam}
    return loss, logs

def train(cfg=DEFENSE_CFG):
    random.seed(cfg["seed"])
    th.manual_seed(cfg["seed"])
    model, tok = load_for_training()

    # layers: hidden_states[0 .. n_layers - k]; with k >= 1 the post-ln_f output is excluded
    n_layers = model.config.num_hidden_layers
    cfg["layers"] = list(range(0, n_layers + 1 - cfg["k"]))
    cfg["n_eoi"] = len(tok.encode(EOI_STR))
    print(f"layers={cfg['layers'][0]}..{cfg['layers'][-1]}  n_eoi={cfg['n_eoi']}")

    # ---- data (splits are disjoint by construction; see prepare_data) ----
    paths = prepare_data(cfg["data_dir"])
    harm_train = load_instructions(paths["harmful_train"])
    ben_train = load_instructions(paths["harmless_train"])
    ben_train = random.sample(ben_train, min(cfg["n_harmless_train"], len(ben_train)))
    harm_val = load_instructions(paths["harmful_val"])[:cfg["n_val"]]
    ben_val = load_instructions(paths["harmless_val"])[:cfg["n_val"]]

    # ---- targets from the base model itself ----
    print("generating base-model refusals and benign completions...")
    refusals = generate(model, tok, harm_train, max_new_tokens=64)
    kept = [(x, r) for x, r in zip(harm_train, refusals) if is_refusal(r)]
    print(f"kept {len(kept)}/{len(harm_train)} harmful prompts the base model refuses")
    ben_resp = generate(model, tok, ben_train, max_new_tokens=128)
    harm_ex = make_examples(tok, [x for x, _ in kept], [r for _, r in kept], max_target=64)
    ben_ex = make_examples(tok, ben_train, ben_resp, max_target=128)

    # ---- LoRA ----
    model.config.use_cache = False
    if cfg["grad_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except TypeError:                       # older Qwen remote code + newer transformers
            model.transformer.gradient_checkpointing = True
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "w1", "w2"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])

    print("baseline:", evaluate(model, tok, harm_val, ben_val, cfg))

    # ---- train ----
    B = cfg["batch_per_class"]
    for step in range(cfg["steps"]):
        model.train()
        hb = collate(tok, random.sample(harm_ex, B), model.device)
        bb = collate(tok, random.sample(ben_ex, B), model.device)

        loss, logs = training_step(model, hb, bb, step, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg["log_every"] == 0:
            print(step, {k: round(v, 4) for k, v in logs.items()})
        if step > 0 and step % cfg["eval_every"] == 0:
            print(f"eval @ {step}:", evaluate(model, tok, harm_val, ben_val, cfg))
            model.save_pretrained(f"{cfg['out_dir']}/step{step}")

    print("final:", evaluate(model, tok, harm_val, ben_val, cfg))
    model.save_pretrained(cfg["out_dir"])

    # ---- Q1: fresh linear probes on the obfuscated model ----
    acts_h = prompt_acts(model, tok, harm_val, cfg["layers"], cfg["n_eoi"])
    acts_b = prompt_acts(model, tok, ben_val, cfg["layers"], cfg["n_eoi"])
    print("probe AUROC per layer:", [round(a, 3) for a in probe_auroc_per_layer(acts_h, acts_b)])
    return model, tok

if __name__ == "__main__":
    model, tok = train()
