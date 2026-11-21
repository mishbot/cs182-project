"""Sample WildGuardMix prompts and upload final-token layer activations to S3."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from ._model_config import load_local_model_config
from ._types import (
    ActivationRow,
    Device,
    HarmLabel,
    NextTokenStatistics,
    PathInput,
    PreparedInput,
    PreparedPrompt,
    PromptGroup,
    PromptRow,
    RunManifest,
    RunMetadata,
    Shard,
    TensorBatch,
)
from .dataset_artifacts import download_dataset
from .dataset_partition import load_partition, partition_uri
from .model_artifacts import MANIFEST, download_model, parse_s3_uri, sha256
from .qwen_model import (
    PROMPT_FORMAT,
    SYSTEM_MESSAGE,
    backbone,
    decoder_blocks,
    format_chat,
    load_model,
    load_tokenizer,
    output_head,
)

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa
    from mypy_boto3_s3 import S3Client
    from numpy.typing import NDArray
    from torch import Tensor
    from torch.nn import Module
    from transformers import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SHARD_ROWS = 128
ACTIVATION_SEMANTICS = "decoder_block_output_before_final_norm_at_last_chat_formatted_input_token"


def sample_prompts(
    rows: Iterable[Mapping[str, object]],
    sample_size: int,
    seed: int,
    allowed_indices: set[int] | None = None,
) -> list[PromptRow]:
    """Deduplicate exact strings, drop ambiguous groups, and balance without replacement."""
    if sample_size < 4:
        raise ValueError("--sample-size must be at least 4 to sample all four groups")
    prompts: dict[str, PromptGroup] = {}
    for index, row in enumerate(rows):
        prompt, label = row.get("prompt"), row.get("prompt_harm_label")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or label not in {"harmful", "unharmful"}
            or type(row.get("adversarial")) is not bool
        ):
            continue
        if prompt not in prompts:
            prompts[prompt] = {
                "source_row_index": index,
                "prompt": prompt,
                "labels": set(),
                "adversarial_values": set(),
            }
        prompts[prompt]["labels"].add(cast(HarmLabel, label))
        prompts[prompt]["adversarial_values"].add(cast(bool, row["adversarial"]))
    keys = [(label, adv) for adv in (False, True) for label in ("harmful", "unharmful")]
    groups: dict[tuple[str, bool], list[PromptRow]] = {key: [] for key in keys}
    for item in prompts.values():
        if len(item["labels"]) == len(item["adversarial_values"]) == 1 and (
            allowed_indices is None or item["source_row_index"] in allowed_indices
        ):
            label = next(iter(item["labels"]))
            adversarial = next(iter(item["adversarial_values"]))
            groups[label, adversarial].append(
                {
                    "source_row_index": item["source_row_index"],
                    "prompt": item["prompt"],
                    "prompt_harm_label": label,
                    "harmful": label == "harmful",
                    "adversarial": adversarial,
                }
            )
    rng = random.Random(seed)
    selected: list[PromptRow] = []
    for position, key in enumerate(keys):
        count = sample_size // 4 + (position < sample_size % 4)
        if count > len(groups[key]):
            raise ValueError(
                f"Need {count} unique {key} prompts, but only {len(groups[key])} are eligible."
            )
        selected.extend(rng.sample(groups[key], count))
    rng.shuffle(selected)
    return selected


def prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    selected: Iterable[PromptRow],
    max_input_tokens: int,
    context_window: int,
) -> list[PreparedInput]:
    """Preflight every input before any model forwards or activation uploads."""
    limit = min(max_input_tokens, context_window)
    prepared: list[PreparedInput] = []
    for row in selected:
        encoded = format_chat(tokenizer, row["prompt"])
        count = len(cast(list[int], encoded["input_ids"]))
        if count > limit:
            raise ValueError(
                f"Source row {row['source_row_index']} has {count} formatted tokens; limit is {limit} "
                f"(--max-input-tokens={max_input_tokens}, model context={context_window}). "
                "Increase the limit if the model supports it; inputs are never truncated or resampled."
            )
        prepared.append((PreparedPrompt(**row, formatted_input_token_count=count), encoded))
    return prepared


def capture_batch(transformer: Module, batch: TensorBatch) -> tuple[NDArray[np.float32], Tensor]:
    """Capture only [batch, hidden] per layer; do not retain full hidden-state histories."""
    import torch

    layers = decoder_blocks(transformer)
    captured: list[Tensor | None] = [None] * len(layers)
    positions = batch["attention_mask"].sum(dim=1) - 1
    indices = torch.arange(positions.shape[0], device=positions.device)
    handles = []

    def hook_for(
        index: int,
    ) -> Callable[[Module, tuple[Tensor, ...], Tensor | tuple[Tensor, ...]], None]:
        def capture(
            _module: Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor | tuple[Tensor, ...],
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[indices, positions].detach()

        return capture

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook_for(index)))
        with torch.inference_mode():
            output = transformer(
                **batch, use_cache=False, output_hidden_states=False, output_attentions=False
            )
            final_hidden = output.last_hidden_state[indices, positions]
        if any(value is None for value in captured):
            raise RuntimeError("Not all decoder layers produced an activation.")
        # Copy the small selected vectors once, avoiding a GPU synchronization per layer.
        return (
            torch.stack(cast("list[Tensor]", captured), dim=1)
            .to(device="cpu", dtype=torch.float32)
            .numpy(),
            final_hidden,
        )
    finally:
        for handle in handles:
            handle.remove()


def next_token_statistics(
    logits: Tensor, refusal_token_ids: Sequence[int]
) -> list[NextTokenStatistics]:
    """Full-vocabulary probabilities and stable refusal log masses, not top-k renormalization."""
    import torch

    values = logits.float()
    ids = list(refusal_token_ids)
    if (
        values.ndim != 2
        or not torch.isfinite(values).all()
        or not ids
        or len(set(ids)) != len(ids)
        or min(ids) < 0
        or max(ids) >= values.shape[1]
        or len(ids) == values.shape[1]
    ):
        raise ValueError(
            "Expected finite logits and a nonempty proper subset of refusal token IDs."
        )
    normalizer = values.logsumexp(dim=-1)
    refusal = values[:, ids].logsumexp(dim=-1) - normalizer
    complement = values.clone()
    complement[:, ids] = -torch.inf
    nonrefusal = complement.logsumexp(dim=-1) - normalizer
    top_logits, top_ids = values.topk(min(100, values.shape[1]), dim=-1, sorted=True)
    probabilities = (top_logits - normalizer[:, None]).exp().cpu().tolist()
    token_ids = top_ids.cpu().tolist()
    return [
        NextTokenStatistics(
            top_token_ids=indices,
            top_token_probabilities=probs,
            refusal_log_probability=ref,
            nonrefusal_log_probability=nonref,
        )
        for indices, probs, ref, nonref in zip(
            token_ids, probabilities, refusal.cpu().tolist(), nonrefusal.cpu().tolist()
        )
    ]


def activation_batches(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prepared: Sequence[PreparedInput],
    batch_size: int,
    device: Device,
    refusal_token_ids: Sequence[int] = (40, 2121),
) -> Iterator[list[ActivationRow]]:
    import torch

    tokenizer.padding_side = "right"
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        batch = cast(
            TensorBatch,
            tokenizer.pad([encoded for _, encoded in chunk], padding=True, return_tensors="pt").to(
                device
            ),
        )
        with torch.inference_mode():
            vectors, final_hidden = capture_batch(backbone(model), batch)
            statistics = next_token_statistics(output_head(model)(final_hidden), refusal_token_ids)
        yield [
            ActivationRow(**row, activations=vector.tolist(), **stats)
            for (row, _), vector, stats in zip(chunk, vectors, statistics)
        ]


def activation_schema(num_layers: int, hidden_size: int) -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            ("source_row_index", pa.int64()),
            ("prompt", pa.string()),
            ("prompt_harm_label", pa.string()),
            ("harmful", pa.bool_()),
            ("adversarial", pa.bool_()),
            ("top_token_ids", pa.list_(pa.int32(), 100)),
            ("top_token_probabilities", pa.list_(pa.float64(), 100)),
            ("refusal_log_probability", pa.float64()),
            ("nonrefusal_log_probability", pa.float64()),
            ("formatted_input_token_count", pa.int32()),
            ("activations", pa.list_(pa.list_(pa.float32(), hidden_size), num_layers)),
        ]
    )


def upload_run(
    s3: S3Client,
    output_uri: str,
    batches: Iterable[Iterable[ActivationRow]],
    metadata: RunMetadata,
    schema: pa.Schema,
) -> str:
    """Bound memory to one shard plus one batch; only manifest.json signals success."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, root = parse_s3_uri(output_uri)
    run_id = uuid.uuid4().hex
    prefix = f"{root}/{run_id}"
    shards: list[Shard] = []
    pending: list[ActivationRow] = []

    def flush(rows: list[ActivationRow]) -> None:
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), buffer, compression="zstd")
        data = buffer.getvalue()
        key = f"{prefix}/part-{len(shards):05d}.parquet"
        s3.upload_fileobj(
            io.BytesIO(data.to_pybytes()),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
        )
        shards.append(
            {
                "key": key,
                "rows": len(rows),
                "size": data.size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        logger.info("Uploaded %s (%s rows)", key, len(rows))

    try:
        for batch in batches:
            for row in batch:
                pending.append(row)
                if len(pending) == SHARD_ROWS:
                    flush(pending)
                    pending = []
        if pending:
            flush(pending)
        total = sum(part["rows"] for part in shards)
        if total != metadata["sample_size"]:
            raise ValueError(f"Expected {metadata['sample_size']} rows but extracted {total}.")
        manifest = RunManifest(
            **metadata,
            schema_version=2,
            status="complete",
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            row_count=total,
            shards=shards,
        )
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/manifest.json",
            Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as error:
        raise RuntimeError(
            f"Incomplete run at s3://{bucket}/{prefix}; no completion manifest was published: {error}"
        ) from error
    return f"s3://{bucket}/{prefix}/manifest.json"


def collect(
    s3: S3Client,
    *,
    model_s3_uri: str,
    dataset_s3_uri: str,
    output_s3_uri: str,
    sample_size: int,
    split: Literal["train", "validation", "test"] = "train",
    partition_s3_uri: str | None = None,
    seed: int = 0,
    batch_size: int = 4,
    max_input_tokens: int = 2048,
    device: Device = "auto",
    cache_dir: PathInput = "/cache",
) -> str:
    """Collect a balanced sample and return its S3 completion manifest URI."""
    if sample_size < 4 or batch_size < 1 or max_input_tokens < 1:
        raise ValueError(
            "--sample-size must be >= 4; --batch-size and --max-input-tokens must be positive"
        )
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    for uri in [model_s3_uri, dataset_s3_uri, output_s3_uri]:
        parse_s3_uri(uri)
    import pyarrow.parquet as pq
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; run Docker with --gpus all or select --device cpu."
        )
    cache = Path(cache_dir).expanduser()
    dataset_dir = download_dataset(s3, dataset_s3_uri, cache / "datasets")
    dataset_manifest = json.loads((dataset_dir / MANIFEST).read_text())
    source_split = "test" if split == "test" else "train"
    dataset_path = dataset_dir / dataset_manifest["splits"][source_split]
    rows = pq.read_table(
        dataset_path, columns=["prompt", "prompt_harm_label", "adversarial"]
    ).to_pylist()
    membership = None
    partition: dict[str, object] = {}
    if split != "test":
        uri = partition_s3_uri or partition_uri(dataset_s3_uri, dataset_manifest)
        value, digest = load_partition(
            s3, uri, dataset_manifest, sha256(dataset_dir / MANIFEST), len(rows)
        )
        membership = set(value["splits"][split])
        partition = {"s3_uri": uri, "sha256": digest, "configuration": value["configuration"]}
    elif partition_s3_uri:
        raise ValueError("--partition-s3-uri is only applicable to train/validation.")
    selected = sample_prompts(rows, sample_size, seed, membership)
    model_dir = download_model(s3, model_s3_uri, cache / "models")
    config = load_local_model_config(model_dir)
    tokenizer = load_tokenizer(model_dir)
    refusal_ids = (40, 2121)
    if [tokenizer.decode([i]) for i in refusal_ids] != ["I", "As"]:
        raise ValueError(
            "The tokenizer does not match the paper's Qwen Chat refusal IDs (40, 2121)."
        )
    prepared = prepare_inputs(tokenizer, selected, max_input_tokens, config.max_position_embeddings)
    model = load_model(model_dir, device, config=config)
    model_manifest = json.loads((model_dir / MANIFEST).read_text())
    labels: tuple[HarmLabel, HarmLabel] = ("harmful", "unharmful")
    metadata: RunMetadata = {
        "model": {
            "s3_uri": model_s3_uri,
            "manifest": model_manifest,
            "manifest_sha256": sha256(model_dir / MANIFEST),
        },
        "dataset": {
            "s3_uri": dataset_s3_uri,
            "manifest": dataset_manifest,
            "manifest_sha256": sha256(dataset_dir / MANIFEST),
            "split": split,
            "partition": partition,
        },
        "sample_size": sample_size,
        "seed": seed,
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "device": device,
        "compute_dtype": str(next(model.parameters()).dtype),
        "activation_dtype": "float32",
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "layer_index_base": 0,
        "activation_semantics": ACTIVATION_SEMANTICS,
        "model_execution": {
            "architecture": "qwen",
            "prompt_format": PROMPT_FORMAT,
            "system_message": SYSTEM_MESSAGE,
            "use_flash_attn": False,
            "use_cache_quantization": False,
        },
        "sampling": "balanced_harmfulness_x_adversarial_unique_exact_prompts_without_ambiguous_labels",
        "selected_source_row_indices": [row["source_row_index"] for row in selected],
        "class_counts": {
            label: sum(row["prompt_harm_label"] == label for row in selected) for label in labels
        },
    }
    metadata["group_counts"] = {
        f"{label}_{'adversarial' if adv else 'non_adversarial'}": sum(
            row["prompt_harm_label"] == label and row["adversarial"] == adv for row in selected
        )
        for adv in (False, True)
        for label in labels
    }
    metadata["next_token"] = {
        "top_k": 100,
        "refusal_token_ids": list(refusal_ids),
        "refusal_tokens": ["I", "As"],
        "probability_normalization": "full_vocabulary",
        "refusal_metric_source": "https://arxiv.org/html/2406.11717v3",
    }
    return upload_run(
        s3,
        output_s3_uri,
        activation_batches(model, tokenizer, prepared, batch_size, device),
        metadata,
        activation_schema(config.num_hidden_layers, config.hidden_size),
    )
