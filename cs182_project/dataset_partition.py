"""Immutable row membership for a pinned WildGuardMix training snapshot."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .dataset_artifacts import (
    CORPUS_ID,
    SPLITS,
    dataset_name,
    download_dataset,
    read_prompt_rows,
    source_files,
)
from .model_artifacts import MANIFEST, parse_s3_uri, sha256

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def partition_config(validation_fraction: float = 0.2, seed: int = 0) -> JsonObject:
    if not 0 < validation_fraction < 1 or seed < 0:
        raise ValueError("Validation fraction must be between 0 and 1; seed must be nonnegative.")
    return {
        "algorithm": "exact-prompt-group-stratified-v1",
        "validation_fraction": validation_fraction,
        "seed": seed,
    }


def partition_uri(
    dataset_uri: str, manifest: JsonObject, validation_fraction: float = 0.2, seed: int = 0
) -> str:
    bucket, _ = parse_s3_uri(dataset_uri)
    config = partition_config(validation_fraction, seed)
    identity = {
        "configuration": config,
        "dataset_uri": dataset_uri.rstrip("/"),
        "dataset_manifest": manifest,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"s3://{bucket}/partitions/{dataset_name(manifest)}/{manifest['revision']}/{digest}.json"


def split_rows(
    rows: Sequence[Mapping[str, object]], validation_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[int]]:
    """Group identical prompts, stratifying by all categories observed in each group."""
    partition_config(validation_fraction, seed)
    groups: dict[tuple[str, object], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        key = ("prompt", prompt) if isinstance(prompt, str) else ("missing", index)
        groups[key].append(index)
    strata: dict[str, list[list[int]]] = defaultdict(list)
    for indices in groups.values():
        categories = sorted(
            {
                json.dumps([rows[i].get("prompt_harm_label"), rows[i].get("adversarial")])
                for i in indices
            }
        )
        strata[json.dumps(categories)].append(indices)
    rng = random.Random(seed)
    result: dict[str, list[int]] = {"train": [], "validation": []}
    for category in sorted(strata):
        items = strata[category]
        rng.shuffle(items)
        count = (
            max(1, min(len(items) - 1, int(len(items) * validation_fraction)))
            if len(items) > 1
            else 0
        )
        for index, group in enumerate(items):
            result["validation" if index < count else "train"].extend(group)
    return {name: sorted(indices) for name, indices in result.items()}


def validate_partition(
    value: JsonObject, manifest: JsonObject, manifest_sha256: str, row_count: int
) -> dict[str, list[int]]:
    """Bind membership to the exact source and require a complete disjoint partition."""
    if (
        value.get("schema_version") != 1
        or value.get("dataset_revision") != manifest["revision"]
        or value.get("dataset_manifest_sha256") != manifest_sha256
        or any(value.get(key) != expected for key, expected in partition_source(manifest).items())
        or value.get("row_count") != row_count
    ):
        raise ValueError("Partition provenance does not match the pinned dataset snapshot.")
    configuration = value.get("configuration", {})
    expected = partition_config(
        configuration.get("validation_fraction", 0), configuration.get("seed", -1)
    )
    if configuration != expected:
        raise ValueError("Unsupported partition configuration.")
    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("Partition must contain train and validation row indices.")
    combined: list[int] = []
    for indices in splits.values():
        if not isinstance(indices, list) or any(type(i) is not int for i in indices):
            raise ValueError("Partition row indices must be integer lists.")
        combined.extend(indices)
    if sorted(combined) != list(range(row_count)):
        raise ValueError("Partition membership must cover every row exactly once.")
    return splits


def load_partition(
    s3: S3Client, uri: str, manifest: JsonObject, manifest_sha256: str, row_count: int
) -> tuple[JsonObject, str]:
    bucket, key = parse_s3_uri(uri)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey as error:
        raise ValueError(
            f"Missing partition at {uri}. Run infra/scripts/partition_dataset.py first."
        ) from error
    with response["Body"] as body:
        data = body.read()
    value = json.loads(data)
    validate_partition(value, manifest, manifest_sha256, row_count)
    return value, hashlib.sha256(data).hexdigest()


def publish_partition(
    s3: S3Client,
    dataset_s3_uri: str,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
    cache_dir: PathInput = ".cache",
) -> str:
    """Publish a deterministic sidecar; never rewrite the archived parquet or manifest."""
    directory = download_dataset(s3, dataset_s3_uri, Path(cache_dir) / "datasets")
    manifest = json.loads((directory / MANIFEST).read_text())
    return publish_partition_rows(
        s3,
        dataset_s3_uri,
        manifest,
        sha256(directory / MANIFEST),
        read_prompt_rows(directory, manifest),
        validation_fraction=validation_fraction,
        seed=seed,
    )


def partition_source(manifest: JsonObject) -> JsonObject:
    return (
        {"source_files": source_files(manifest)}
        if manifest["dataset_id"] == CORPUS_ID
        else {"source_file": SPLITS["train"]}
    )


def publish_partition_rows(
    s3: S3Client,
    dataset_s3_uri: str,
    manifest: JsonObject,
    manifest_sha256: str,
    rows: Sequence[Mapping[str, object]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> str:
    """Publish membership directly from prepared rows, without redownloading a newly seeded corpus."""
    from botocore.exceptions import ClientError

    configuration = partition_config(validation_fraction, seed)
    value = {
        "schema_version": 1,
        "dataset_revision": manifest["revision"],
        "dataset_manifest_sha256": manifest_sha256,
        **partition_source(manifest),
        "row_count": len(rows),
        "configuration": configuration,
        "splits": split_rows(rows, validation_fraction, seed),
    }
    uri = partition_uri(dataset_s3_uri, manifest, validation_fraction, seed)
    bucket, key = parse_s3_uri(uri)
    data = json.dumps(value, sort_keys=True, indent=2).encode()
    try:
        s3.put_object(
            Bucket=bucket, Key=key, Body=data, ContentType="application/json", IfNoneMatch="*"
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != "PreconditionFailed":
            raise
        with s3.get_object(Bucket=bucket, Key=key)["Body"] as body:
            if body.read() != data:
                raise ValueError(f"Existing partition differs at immutable URI {uri}.") from None
    return uri
