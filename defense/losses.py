"""
Fine-tune Qwen-1.8B-Chat so the Arditi et al. difference-in-means refusal
direction can't be recovered, while keeping refusal on harmful prompts and
base-model behaviour on benign prompts.

    L = L_refuse + alpha * L_retain + lambda * L_obf

    L_refuse : cross-entropy on the base model's own refusals (harmful prompts)
    L_retain : KL(base || model) on the base model's own completions (benign prompts)
    L_obf    : squared gap between harmful / harmless batch-mean activations,
               normalised by activation scale, averaged over layers & post-instruction
               positions (plain batch means; plateaus above 0 by design)
"""
import torch as th
import torch.nn.functional as F

# Only target positions go through log_softmax, to save memory: Qwen's vocab is ~152k.

def target_mask(batch):
    """True where logits[:, t] predict a target token (aligned with logits[:, :-1])."""
    T = batch["input_ids"].size(1)
    ar = th.arange(T, device=batch["input_ids"].device)[None, :]
    m = (ar >= batch["prompt_len"][:, None]) & batch["attention_mask"].bool()
    return m[:, 1:]

def refusal_loss(logits, batch):
    m = target_mask(batch)
    lg = logits[:, :-1][m].float()                     # [N, V]
    tgt = batch["input_ids"][:, 1:][m]                 # [N]
    return F.cross_entropy(lg, tgt)

def retain_loss(model, logits, batch):
    """KL(base || model). Base logits come from the same weights with LoRA off."""
    m = target_mask(batch)
    with th.no_grad(), model.disable_adapter():
        base_logits = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False).logits
    lp_model = F.log_softmax(logits[:, :-1][m].float(), dim=-1)
    lp_base = F.log_softmax(base_logits[:, :-1][m].float(), dim=-1)
    return F.kl_div(lp_model, lp_base, log_target=True, reduction="batchmean")

def _scale(acts_h, acts_b):
    """Mean squared activation norm over both classes, no gradient. -> [L, P]"""
    return th.cat([acts_h, acts_b], 2).pow(2).sum(-1).mean(2).detach()

def mean_gap_loss(acts_h, acts_b):
    """Plain batch-mean gap. Mean (not sum) over layers & positions, so lambda
    doesn't need retuning when you change k."""
    gap = (acts_h.mean(2) - acts_b.mean(2)).pow(2).sum(-1)          # [L, P]
    return (gap / _scale(acts_h, acts_b)).mean()
