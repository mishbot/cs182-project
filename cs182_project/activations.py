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

from ._types import (
    ActivationRow,
    Device,
    HarmLabel,
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
from .model_artifacts import MANIFEST, download_model, parse_s3_uri, sha256

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa
    from mypy_boto3_s3 import S3Client
    from numpy.typing import NDArray
    from torch import Tensor
    from torch.nn import Module
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SHARD_ROWS = 128
ACTIVATION_SEMANTICS = "decoder_block_output_before_final_norm_at_last_chat_formatted_input_token"


def sample_prompts(
    rows: Iterable[Mapping[str, object]],
    sample_size: int,
    seed: int,
) -> list[PromptRow]:
    """Deduplicate exact strings, drop ambiguous groups, and balance without replacement."""
    if sample_size < 2:
        raise ValueError("--sample-size must be at least 2")
    prompts: dict[str, PromptGroup] = {}
    for index, row in enumerate(rows):
        prompt, label = row.get("prompt"), row.get("prompt_harm_label")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or label not in {"harmful", "unharmful"}
        ):
            continue
        if prompt not in prompts:
            prompts[prompt] = {"source_row_index": index, "prompt": prompt, "labels": set()}
        prompts[prompt]["labels"].add(cast(HarmLabel, label))
    groups: dict[HarmLabel, list[PromptRow]] = {"harmful": [], "unharmful": []}
    for item in prompts.values():
        if len(item["labels"]) == 1:
            label = next(iter(item["labels"]))
            groups[label].append(
                {
                    "source_row_index": item["source_row_index"],
                    "prompt": item["prompt"],
                    "prompt_harm_label": label,
                    "harmful": label == "harmful",
                }
            )
    rng = random.Random(seed)
    selected: list[PromptRow] = []
    counts: list[tuple[HarmLabel, int]] = [
        ("harmful", (sample_size + 1) // 2),
        ("unharmful", sample_size // 2),
    ]
    for label, count in counts:
        if count > len(groups[label]):
            raise ValueError(
                f"Need {count} unique {label} prompts, but only {len(groups[label])} are eligible."
            )
        selected.extend(rng.sample(groups[label], count))
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
        text = cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        encoded = tokenizer(
            text, add_special_tokens=False, truncation=False, return_attention_mask=True
        )
        count = len(cast(list[int], encoded["input_ids"]))
        if count > limit:
            raise ValueError(
                f"Source row {row['source_row_index']} has {count} formatted tokens; limit is {limit} "
                f"(--max-input-tokens={max_input_tokens}, model context={context_window}). "
                "Increase the limit if the model supports it; inputs are never truncated or resampled."
            )
        prepared.append((PreparedPrompt(**row, formatted_input_token_count=count), encoded))
    return prepared


def capture_batch(backbone: Qwen2Model, batch: TensorBatch) -> NDArray[np.float32]:
    """Capture only [batch, hidden] per layer; do not retain full hidden-state histories."""
    import torch

    layers = backbone.layers
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
            backbone(**batch, use_cache=False, output_hidden_states=False, output_attentions=False)
        if any(value is None for value in captured):
            raise RuntimeError("Not all decoder layers produced an activation.")
        # Copy the small selected vectors once, avoiding a GPU synchronization per layer.
        return (
            torch.stack(cast("list[Tensor]", captured), dim=1)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
    finally:
        for handle in handles:
            handle.remove()


def activation_batches(
    backbone: Qwen2Model,
    tokenizer: PreTrainedTokenizerBase,
    prepared: Sequence[PreparedInput],
    batch_size: int,
    device: Device,
) -> Iterator[list[ActivationRow]]:
    tokenizer.padding_side = "right"
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        batch = cast(
            TensorBatch,
            tokenizer.pad([encoded for _, encoded in chunk], padding=True, return_tensors="pt").to(
                device
            ),
        )
        vectors = capture_batch(backbone, batch)
        yield [
            ActivationRow(**row, activations=vector.tolist())
            for (row, _), vector in zip(chunk, vectors)
        ]


def activation_schema(num_layers: int, hidden_size: int) -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            ("source_row_index", pa.int64()),
            ("prompt", pa.string()),
            ("prompt_harm_label", pa.string()),
            ("harmful", pa.bool_()),
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
            schema_version=1,
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
    split: Literal["train", "test"] = "train",
    seed: int = 0,
    batch_size: int = 4,
    max_input_tokens: int = 2048,
    device: Device = "auto",
    cache_dir: PathInput = "/cache",
) -> str:
    """Collect a balanced sample and return its S3 completion manifest URI."""
    if sample_size < 2 or batch_size < 1 or max_input_tokens < 1:
        raise ValueError(
            "--sample-size must be >= 2; --batch-size and --max-input-tokens must be positive"
        )
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    for uri in [model_s3_uri, dataset_s3_uri, output_s3_uri]:
        parse_s3_uri(uri)
    import pyarrow.parquet as pq
    import torch
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; run Docker with --gpus all or select --device cpu."
        )
    cache = Path(cache_dir).expanduser()
    dataset_dir = download_dataset(s3, dataset_s3_uri, cache / "datasets")
    dataset_manifest = json.loads((dataset_dir / MANIFEST).read_text())
    dataset_path = dataset_dir / dataset_manifest["splits"][split]
    rows = pq.read_table(dataset_path, columns=["prompt", "prompt_harm_label"]).to_pylist()
    selected = sample_prompts(rows, sample_size, seed)
    model_dir = download_model(s3, model_s3_uri, cache / "models")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen2":
        raise ValueError(
            "Activation extraction currently supports the Qwen2 architecture used by Qwen1.5."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    prepared = prepare_inputs(tokenizer, selected, max_input_tokens, config.max_position_embeddings)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
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
        "sampling": "balanced_unique_exact_prompts_without_missing_or_conflicting_labels; odd_extra_harmful",
        "selected_source_row_indices": [row["source_row_index"] for row in selected],
        "class_counts": {
            label: sum(row["prompt_harm_label"] == label for row in selected) for label in labels
        },
    }
    return upload_run(
        s3,
        output_s3_uri,
        activation_batches(model.model, tokenizer, prepared, batch_size, device),
        metadata,
        activation_schema(config.num_hidden_layers, config.hidden_size),
    )
