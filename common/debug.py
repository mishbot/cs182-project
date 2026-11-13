"""Diagnostic for the left-padding failure: padded rows of a batched generation came out as '!' (token 0).

Arditi et al. and Qwen's README both left-pad this model, so something in our setup breaks it.
padding_report() varies one thing at a time: precision, pad token, attention kernel, and the KV cache.
"""
import contextlib

import torch as th
from transformers import AutoModelForCausalLM

from common.models import load_tokenizer, model_path, prompt_acts
from config import EOI_STR, PROMPT_TEMPLATE

DTYPES = {"bf16": th.bfloat16, "fp16": th.float16, "fp32": th.float32}


def _load(name):
    model_dir = model_path()
    tok = load_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="cuda", torch_dtype=DTYPES[name],
        bf16=name == "bf16", fp16=name == "fp16", fp32=name == "fp32",
        use_flash_attn=False, trust_remote_code=True,
    ).eval()
    return model, tok


def _left_pad(seqs, pad_id, device):
    T = max(map(len, seqs))
    ids = th.full((len(seqs), T), pad_id, dtype=th.long)
    mask = th.zeros((len(seqs), T), dtype=th.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = th.tensor(s)
        mask[i, T - len(s):] = 1
    return ids.to(device), mask.to(device)


def _first_bad(hidden_states, select=None):
    """Index of the first hidden state with NaN or inf (optionally only at selected positions), or None."""
    for i, h in enumerate(hidden_states):
        if not th.isfinite(h if select is None else h[select]).all():
            return i
    return None


@th.no_grad()
def _forward_check(model, seqs, pad_id):
    """One forward pass, no generation: left-padded batch vs each prompt alone, at the last position."""
    ids, mask = _left_pad(seqs, pad_id, model.device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    batched = out.logits[:, -1].float()
    padded = (mask == 0).any(dim=1).cpu()
    same, diff = [], []
    for i, s in enumerate(seqs):
        one = th.tensor([s], device=model.device)
        single = model(input_ids=one, attention_mask=th.ones_like(one), use_cache=False).logits[0, -1].float()
        same.append(bool(batched[i].argmax() == single.argmax()))
        diff.append((batched[i] - single).abs().max().item())
    same, diff = th.tensor(same), th.tensor(diff)
    return {
        "rows with NaN logits": int(batched.isnan().any(dim=1).sum()),
        "first bad layer (any position)": _first_bad(out.hidden_states),
        "first bad layer (real tokens)": _first_bad(out.hidden_states, mask.bool()),
        "top-1 matches single, padded rows": f"{int(same[padded].sum())}/{int(padded.sum())}",
        "top-1 matches single, unpadded rows": f"{int(same[~padded].sum())}/{int((~padded).sum())}",
        "max logit diff, padded rows": round(th.nan_to_num(diff[padded], nan=float("inf")).max().item(), 3),
    }


@th.no_grad()
def _generate_check(model, tok, seqs, pad_id, use_cache, max_new_tokens=16):
    """Greedy generation, left-padded batch vs each prompt alone. Same arguments as common.models.generate."""
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=use_cache,
              eos_token_id=[tok.im_end_id, tok.eod_id], pad_token_id=pad_id)
    ids, mask = _left_pad(seqs, pad_id, model.device)
    batched = model.generate(input_ids=ids, attention_mask=mask, **kw)[:, ids.shape[1]:]
    padded = (mask == 0).any(dim=1).cpu()
    first, prefix, garbage = [], [], []
    for i, s in enumerate(seqs):
        one = th.tensor([s], device=model.device)
        single = model.generate(input_ids=one, attention_mask=th.ones_like(one), **kw)[0, one.shape[1]:]
        n = min(len(single), 8)
        first.append(bool(batched[i, 0] == single[0]))
        prefix.append(bool((batched[i, :n] == single[:n]).all()))
        garbage.append(bool((batched[i] == 0).all()))        # token 0 decodes to '!'
    first, prefix, garbage = map(th.tensor, (first, prefix, garbage))
    k = int(padded.sum())
    return {
        "padded rows, first token matches single": f"{int(first[padded].sum())}/{k}",
        "padded rows, first 8 tokens match single": f"{int(prefix[padded].sum())}/{k}",
        "padded rows that are all '!'": f"{int(garbage[padded].sum())}/{k}",
        "sample": repr(tok.decode(batched[int(padded.int().argmax())], skip_special_tokens=True)[:60]),
    }


@th.no_grad()
def _right_pad_check(model, tok, instructions):
    """The activation path used by training and evaluation: right-padded batch vs one prompt at a time."""
    layers = list(range(model.config.num_hidden_layers + 1))
    n_eoi = len(tok.encode(EOI_STR))
    batched = prompt_acts(model, tok, instructions, layers, n_eoi, batch_size=len(instructions))
    single = prompt_acts(model, tok, instructions, layers, n_eoi, batch_size=1)
    rel = (batched - single).norm(dim=-1) / single.norm(dim=-1).clamp_min(1e-6)       # [L, P, N]
    return {
        "all finite": bool(th.isfinite(batched).all()),
        "max relative error vs single": float(f"{rel.max().item():.2e}"),
        "worst layer": int(rel.amax(dim=(1, 2)).argmax()),
    }


def _show(title, result):
    print(f"  {title}")
    for k, v in result.items():
        print(f"      {k:42s} {v}")


def padding_report(instructions, dtypes=("bf16", "fp16", "fp32")):
    math_only = lambda: th.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    for name in dtypes:
        model, tok = _load(name)
        seqs = [tok.encode(PROMPT_TEMPLATE.format(instruction=x)) for x in instructions]
        longest = max(map(len, seqs))
        print(f"\n{'=' * 30} {name} {'=' * 30}")
        print(f"  pad tokens per row under left padding: {[longest - len(s) for s in seqs]}")

        pads = {"<|extra_0|>": tok.pad_token_id, "<|endoftext|>": tok.eod_id}
        for pad_name, pad_id in pads.items():
            for kernel, ctx in [("default kernel", contextlib.nullcontext), ("math kernel", math_only)]:
                with ctx():
                    _show(f"FORWARD, left-padded, pad={pad_name} ({pad_id}), {kernel}", _forward_check(model, seqs, pad_id))
        for use_cache in (True, False):
            _show(f"GENERATE, left-padded, pad=<|extra_0|>, use_cache={use_cache}",
                  _generate_check(model, tok, seqs, tok.pad_token_id, use_cache))
        _show("GENERATE, left-padded, pad=<|endoftext|>, use_cache=True",
              _generate_check(model, tok, seqs, tok.eod_id, True))
        _show("ACTIVATIONS, right-padded (training/eval path)", _right_pad_check(model, tok, instructions))

        del model
        if th.cuda.is_available():
            th.cuda.empty_cache()
