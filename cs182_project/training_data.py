"""Training-only prompt selection and canonical Qwen prompt/target batches."""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

from ._types import JsonObject, PathInput
from .activations import next_token_statistics
from .dataset_artifacts import download_dataset, read_prompt_rows
from .dataset_partition import load_partition, partition_uri
from .model_artifacts import MANIFEST, sha256
from .qwen_model import format_chat

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from torch import Tensor

Example = tuple[list[int], list[int]]


def training_prompts(
    s3: S3Client,
    dataset_uri: str,
    partition: str | None,
    cache_dir: PathInput,
    seed: int,
    n_harmless: int,
) -> tuple[list[str], list[str], JsonObject]:
    directory = download_dataset(s3, dataset_uri, cache_dir)
    manifest = json.loads((directory / MANIFEST).read_text())
    rows = read_prompt_rows(directory, manifest, "train")
    uri = partition or partition_uri(dataset_uri, manifest)
    value, digest = load_partition(s3, uri, manifest, sha256(directory / MANIFEST), len(rows))
    allowed = set(value["splits"]["train"])
    # Group all rows before selecting membership, so contradictory labels cannot
    # silently become training labels. The existing partition groups exact prompts.
    labels: dict[str, set[str]] = {}
    for row in rows:
        prompt, label = row.get("prompt"), row.get("prompt_harm_label")
        if isinstance(prompt, str) and prompt.strip() and label in {"harmful", "unharmful"}:
            labels.setdefault(prompt, set()).add(str(label))
    groups: dict[str, list[str]] = {"harmful": [], "unharmful": []}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if index not in allowed or not isinstance(prompt, str) or prompt in seen:
            continue
        seen.add(prompt)
        observed = labels.get(prompt, set())
        if len(observed) == 1:
            groups[next(iter(observed))].append(prompt)
    benign = groups["unharmful"]
    benign = random.Random(seed).sample(benign, min(n_harmless, len(benign)))
    return (
        groups["harmful"],
        benign,
        {
            "dataset_s3_uri": dataset_uri,
            "dataset_manifest_sha256": sha256(directory / MANIFEST),
            "partition_s3_uri": uri,
            "partition_sha256": digest,
        },
    )


def base_target(
    model: Any, tokenizer: Any, prompt: str, *, harmful: bool, max_target: int
) -> Example | None:
    import torch

    encoded = format_chat(tokenizer, prompt)["input_ids"]
    if len(encoded) > 256:
        return None
    ids = torch.tensor([encoded], device=model.device)
    with torch.no_grad():
        if harmful:
            output = model(input_ids=ids, use_cache=False)
            stats = next_token_statistics(output.logits[:, -1], (40, 2121))[0]
            if stats["refusal_log_probability"] - stats["nonrefusal_log_probability"] <= 0:
                return None
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_target,
            use_cache=True,
            eos_token_id=[tokenizer.im_end_id, tokenizer.eod_id],
            pad_token_id=tokenizer.pad_token_id,
        )
    target = generated[0, ids.shape[1] :].tolist()
    if not target:
        return None
    if len(target) < max_target and target[-1] not in {tokenizer.im_end_id, tokenizer.eod_id}:
        target.append(tokenizer.im_end_id)
    return list(encoded), target


def collate(tokenizer: Any, examples: list[Example], device: Any) -> dict[str, Tensor]:
    import torch

    lengths = [len(prompt) + len(target) for prompt, target in examples]
    ids = torch.full((len(examples), max(lengths)), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (prompt, target) in enumerate(examples):
        sequence = prompt + target
        ids[index, : len(sequence)] = torch.tensor(sequence)
        mask[index, : len(sequence)] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "prompt_len": torch.tensor([len(p) for p, _ in examples], device=device),
    }
