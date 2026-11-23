"""Main's defender objective: refusal CE + alpha * retain KL + lambda * mean gap."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import JsonObject

if TYPE_CHECKING:
    from torch import Tensor


def target_mask(batch: dict[str, Tensor]) -> Tensor:
    import torch

    positions = torch.arange(batch["input_ids"].size(1), device=batch["input_ids"].device)
    return ((positions[None, :] >= batch["prompt_len"][:, None]) & batch["attention_mask"].bool())[
        :, 1:
    ]


def gather_resid(hidden_states: tuple[Tensor, ...], layers: list[int], pos: Tensor) -> Tensor:
    import torch

    gathered = []
    for layer in layers:
        hidden = hidden_states[layer]
        index = pos[..., None].expand(*pos.shape, hidden.size(-1))
        gathered.append(torch.gather(hidden, 1, index).transpose(0, 1))
    return torch.stack(gathered).float()


def mean_gap_loss(acts_h: Tensor, acts_b: Tensor) -> Tensor:
    import torch

    scale = torch.cat([acts_h, acts_b], 2).pow(2).sum(-1).mean(2).detach()
    gap = (acts_h.mean(2) - acts_b.mean(2)).pow(2).sum(-1)
    return (gap / scale).mean()


def training_step(
    model: Any,
    harm_batch: dict[str, Tensor],
    benign_batch: dict[str, Tensor],
    step: int,
    cfg: JsonObject,
) -> tuple[Tensor, JsonObject]:
    import torch
    from torch.nn import functional

    outputs = [
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        for batch in (harm_batch, benign_batch)
    ]
    hm, bm = target_mask(harm_batch), target_mask(benign_batch)
    refuse = functional.cross_entropy(
        outputs[0].logits[:, :-1][hm].float(), harm_batch["input_ids"][:, 1:][hm]
    )
    with torch.no_grad(), model.disable_adapter():
        base = model(
            input_ids=benign_batch["input_ids"],
            attention_mask=benign_batch["attention_mask"],
            use_cache=False,
        ).logits
    retain = functional.kl_div(
        functional.log_softmax(outputs[1].logits[:, :-1][bm].float(), dim=-1),
        functional.log_softmax(base[:, :-1][bm].float(), dim=-1),
        log_target=True,
        reduction="batchmean",
    )
    acts = []
    for output, batch in zip(outputs, (harm_batch, benign_batch)):
        offsets = torch.arange(cfg["n_eoi"], device=batch["prompt_len"].device)
        pos = batch["prompt_len"][:, None] - cfg["n_eoi"] + offsets[None, :]
        acts.append(gather_resid(output.hidden_states, cfg["layers"], pos))
    obfuscate = mean_gap_loss(*acts)
    ramp = (step - cfg["warmup_steps"]) / cfg["ramp_steps"]
    weight = cfg["lam_max"] * min(max(ramp, 0.0), 1.0)
    loss = refuse + cfg["alpha"] * retain + weight * obfuscate
    return loss, {
        "loss": loss.item(),
        "L_refuse": refuse.item(),
        "L_retain": retain.item(),
        "L_obf": obfuscate.item(),
        "alpha": cfg["alpha"],
        "lambda": weight,
    }
