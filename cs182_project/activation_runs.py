"""Read completed activation runs from S3 without downloading model weights."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

from ._types import JsonObject, Shard
from .model_artifacts import parse_s3_uri

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np
    from boto3.session import Session
    from mypy_boto3_s3 import S3Client
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivationRun:
    manifest_uri: str
    manifest_sha256: str
    manifest: JsonObject
    source_row_indices: NDArray[np.int64]
    prompts: list[str]
    harmful: NDArray[np.bool_]
    activations: NDArray[np.float32]  # [row, decoder layer, hidden component]


def activation_root(
    session: Session,
    *,
    project: str = "cs182-cddr",
    region: str | None = None,
) -> str:
    """Resolve the standard runs prefix using only account identity and region."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,29}", project):
        raise ValueError("Project must follow the persistent stack's project naming rules.")
    region = region or session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise ValueError("Invalid AWS region.")
    account = session.client("sts", region_name=region).get_caller_identity()["Account"]
    if not re.fullmatch(r"[0-9]{12}", account):
        raise ValueError("AWS STS did not return a valid account ID.")
    return f"s3://{project}-{account}-{region}-activations/runs"


def _read_object(s3: S3Client, bucket: str, key: str) -> bytes:
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


def _parse_manifest(data: bytes) -> JsonObject:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("An activation manifest must be a JSON object.")
    return value


def latest_activation_run(s3: S3Client, root_uri: str) -> str:
    """Find the latest completion marker across all pages; ignore orphan shards."""
    bucket, prefix = parse_s3_uri(root_uri)
    candidates: list[tuple[datetime, str]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key, modified = item.get("Key", ""), item.get("LastModified")
            relative = key.removeprefix(f"{prefix}/").split("/")
            if (
                len(relative) == 2
                and relative[0] not in {"", ".", ".."}
                and relative[1] == "manifest.json"
            ):
                if modified is None:
                    raise ValueError(f"Completion marker {key} has no publication timestamp.")
                candidates.append((modified, key))
    for _, key in sorted(candidates, reverse=True):
        manifest = _parse_manifest(_read_object(s3, bucket, key))
        if manifest.get("status") == "complete":
            return f"s3://{bucket}/{key}"
    raise ValueError(f"No completed activation runs found under {root_uri}.")


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_manifest(manifest: JsonObject, prefix: str) -> tuple[int, int, int, list[Shard]]:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Expected a completed schema-version-1 activation run.")
    if not isinstance(manifest.get("run_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", manifest["run_id"]
    ):
        raise ValueError("The activation manifest has an invalid run_id.")
    rows = _positive_integer(manifest.get("row_count"), "row_count")
    layers = _positive_integer(manifest.get("num_layers"), "num_layers")
    hidden = _positive_integer(manifest.get("hidden_size"), "hidden_size")
    if manifest.get("sample_size") != rows or manifest.get("layer_index_base", 0) != 0:
        raise ValueError("Activation sample size or layer numbering is inconsistent.")
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ValueError("The activation manifest contains no shards.")
    shards: list[Shard] = []
    names: set[str] = set()
    for item in manifest["shards"]:
        key = item.get("key")
        if (
            not isinstance(key, str)
            or not key.startswith(f"{prefix}/")
            or not key.endswith(".parquet")
            or ".." in PurePosixPath(key).parts
            or str(PurePosixPath(key)) != key
            or key in names
        ):
            raise ValueError("Shard keys must be unique Parquet objects within the selected run.")
        checksum = item.get("sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid checksum for {key}.")
        shards.append(
            Shard(
                key=key,
                sha256=checksum,
                rows=_positive_integer(item.get("rows"), "shard rows"),
                size=_positive_integer(item.get("size"), "shard size"),
            )
        )
        names.add(key)
    if sum(shard["rows"] for shard in shards) != rows:
        raise ValueError("Shard row counts do not match the manifest row_count.")
    return rows, layers, hidden, shards


def load_activation_run(s3: S3Client, run_uri: str) -> ActivationRun:
    """Verify and load one run; shard tensors never become Python float lists."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket, key = parse_s3_uri(run_uri)
    if not key.endswith("/manifest.json"):
        key = f"{key}/manifest.json"
    prefix = key.rsplit("/", 1)[0]
    data = _read_object(s3, bucket, key)
    manifest = _parse_manifest(data)
    rows, layers, hidden, shards = _validate_manifest(manifest, prefix)
    activations = np.empty((rows, layers, hidden), dtype=np.float32)
    harmful = np.empty(rows, dtype=np.bool_)
    indices = np.empty(rows, dtype=np.int64)
    prompts: list[str] = []
    seen_indices: set[int] = set()
    offset = 0
    for shard in shards:
        logger.info("Reading %s (%s rows)", shard["key"], shard["rows"])
        raw = _read_object(s3, bucket, shard["key"])
        if len(raw) != shard["size"] or hashlib.sha256(raw).hexdigest() != shard["sha256"]:
            raise ValueError(f"Checksum/size mismatch for {shard['key']}.")
        table = pq.read_table(
            pa.BufferReader(raw),
            columns=[
                "source_row_index",
                "prompt",
                "prompt_harm_label",
                "harmful",
                "activations",
            ],
        )
        if table.num_rows != shard["rows"]:
            raise ValueError(f"Row count mismatch for {shard['key']}.")
        column = table.column("activations").combine_chunks()
        if not pa.types.is_fixed_size_list(column.type):
            raise ValueError("Activations must be fixed-size nested float32 arrays.")
        outer = cast(pa.FixedSizeListArray, column)
        outer_type = cast(pa.FixedSizeListType, outer.type)
        if outer_type.list_size != layers or not pa.types.is_fixed_size_list(outer_type.value_type):
            raise ValueError("Activation layer dimensions do not match the manifest.")
        inner = cast(pa.FixedSizeListArray, outer.flatten())
        inner_type = cast(pa.FixedSizeListType, inner.type)
        flat = inner.flatten()
        if inner_type.list_size != hidden or flat.type != pa.float32():
            raise ValueError("Activation hidden dimensions or dtype do not match the manifest.")
        if outer.null_count or inner.null_count or flat.null_count:
            raise ValueError("Activation tensors contain null values.")
        values = flat.to_numpy(zero_copy_only=False).reshape(shard["rows"], layers, hidden)
        if not np.isfinite(values).all():
            raise ValueError("Activation tensors contain nonfinite values.")
        activations[offset : offset + shard["rows"]] = values
        for index, row in enumerate(table.drop(["activations"]).to_pylist(), start=offset):
            source = row["source_row_index"]
            flag, label, prompt = row["harmful"], row["prompt_harm_label"], row["prompt"]
            if (
                type(source) is not int
                or source < 0
                or source in seen_indices
                or type(flag) is not bool
                or label != ("harmful" if flag else "unharmful")
                or not isinstance(prompt, str)
                or not prompt.strip()
            ):
                raise ValueError("Activation rows contain invalid indices, labels, or prompts.")
            seen_indices.add(source)
            indices[index], harmful[index] = source, flag
            prompts.append(prompt)
        offset += shard["rows"]
    selected = manifest.get("selected_source_row_indices")
    if selected is not None and selected != indices.tolist():
        raise ValueError("Source row ordering differs from the manifest.")
    counts = {"harmful": int(harmful.sum()), "unharmful": int((~harmful).sum())}
    if manifest.get("class_counts", counts) != counts:
        raise ValueError("Class counts differ from the manifest.")
    return ActivationRun(
        f"s3://{bucket}/{key}",
        hashlib.sha256(data).hexdigest(),
        manifest,
        indices,
        prompts,
        harmful,
        activations,
    )
