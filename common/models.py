"""Shared Qwen loaders, generation, and activation readers. Used by the defense and the attack."""
import warnings

import torch as th
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import collate
from config import CACHE_DIR, DEVICE, MODEL_ID, PAD_TOKEN, PROMPT_TEMPLATE, SYSTEM_PROMPT

warnings.filterwarnings("ignore", message="MPS: no support for int64 min/max ops")
warnings.filterwarnings("ignore", message="MPS: no support for int64 repeats mask")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok.pad_token = PAD_TOKEN          # Qwen-1 has no pad token by default
    return tok

def load_for_training(model_id=MODEL_ID, device=DEVICE):
    """bf16 on Ampere+ GPUs, fp32 otherwise (fp16 training is unstable).
    Flash attention is off: Qwen-1's flash path and padded batches don't mix well."""
    model_dir = snapshot_download(model_id, cache_dir=CACHE_DIR)
    tok = load_tokenizer(model_dir)
    use_bf16 = device == "cuda" and th.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device,
        torch_dtype=th.bfloat16 if use_bf16 else th.float32,
        bf16=use_bf16,
        fp32=not use_bf16,
        use_flash_attn=False,
        trust_remote_code=True,
    )
    return model, tok

class LLM:
    """Chat wrapper. Loads Qwen for interactive use, or wraps an existing model,
    e.g. the fine-tuned one: LLM(model=model, tokenizer=tok)."""

    def __init__(self, model_id=MODEL_ID, device=DEVICE, model=None, tokenizer=None):
        if model is None:
            model_dir = snapshot_download(model_id, cache_dir=CACHE_DIR)
            tokenizer = load_tokenizer(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map=device,
                torch_dtype=th.float32,
                fp32=True,
                trust_remote_code=True,
                use_flash_attn=(device == "cuda" and th.cuda.get_device_capability()[0] >= 8),
            )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.history = None

    def chat(self, prompt, system=SYSTEM_PROMPT, keep_history=True):
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history if keep_history else None,
            system=system,
        )
        if keep_history:
            self.history = history
        return response

    def reset(self):
        self.history = None

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@th.no_grad()
def generate(model, tok, instructions, max_new_tokens=64, batch_size=16):
    """Greedy completions with the chat template (left padding for generation)."""
    model.eval()
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(instructions), batch_size):
        prompts = [PROMPT_TEMPLATE.format(instruction=x) for x in instructions[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=[tok.im_end_id, tok.eod_id],
            pad_token_id=tok.pad_token_id,
        )
        for g in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(g, skip_special_tokens=True).strip())
    return outs

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def eoi_positions(prompt_len, n_eoi):
    """Indices of the last n_eoi prompt tokens, per example. -> [B, n_eoi]"""
    offsets = th.arange(n_eoi, device=prompt_len.device)
    return prompt_len[:, None] - n_eoi + offsets[None, :]

def gather_resid(hidden_states, layers, pos):
    """Qwen-1 hidden_states: [0] = embeddings, [i] = output of block i-1,
    [-1] = final output AFTER ln_f. -> [n_layers, n_pos, B, d] (fp32)"""
    B, P = pos.shape
    out = []
    for l in layers:
        h = hidden_states[l]
        idx = pos[..., None].expand(B, P, h.size(-1))
        out.append(th.gather(h, 1, idx).transpose(0, 1))
    return th.stack(out).float()

@th.no_grad()
def prompt_acts(model, tok, instructions, layers, n_eoi, batch_size=32):
    """Post-instruction activations for prompts only. -> [L, P, N, d] on CPU."""
    model.eval()
    chunks = []
    for i in range(0, len(instructions), batch_size):
        ex = [(tok.encode(PROMPT_TEMPLATE.format(instruction=x)), [])
              for x in instructions[i:i + batch_size]]
        b = collate(tok, ex, model.device)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    output_hidden_states=True, use_cache=False)
        pos = eoi_positions(b["prompt_len"], n_eoi)
        chunks.append(gather_resid(out.hidden_states, layers, pos).cpu())
    return th.cat(chunks, 2)
