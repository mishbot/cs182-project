"""Pinned WildGuardMix snapshot format and checksummed S3 caching."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .model_artifacts import download_snapshot, validate_files

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

DATASET_ID = "allenai/wildguardmix"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SPLITS = {"train": "train/wildguard_train.parquet", "test": "test/wildguard_test.parquet"}


def validate_dataset_manifest(manifest: JsonObject) -> None:
    names = validate_files(manifest)
    revision = manifest.get("revision", "")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("Expected a pinned allenai/wildguardmix dataset manifest.")
    if manifest.get("splits") != SPLITS or not {*SPLITS.values(), "README.md"} <= names:
        raise ValueError("Dataset snapshot must contain both official splits and the dataset card.")


def download_dataset(s3: S3Client, uri: str, cache_dir: PathInput) -> Path:
    return download_snapshot(s3, uri, cache_dir, validate_dataset_manifest)
