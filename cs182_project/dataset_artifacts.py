"""Checksummed dataset snapshots and ordered prompt rows for collection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput

CORPUS_ID = "cs182-project/refusal-direction"
CORPUS_NAME = "refusal-direction"
CORPUS_FILES = [
    "advbench.parquet",
    "maliciousinstruct.parquet",
    "tdc2023.parquet",
    "alpaca.parquet",
]
PROMPT_COLUMNS = ["prompt", "prompt_harm_label", "adversarial"]
from .model_artifacts import download_snapshot, validate_files

if TYPE_CHECKING:
    import pyarrow as pa
    from mypy_boto3_s3 import S3Client

DATASET_ID = "allenai/wildguardmix"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SPLITS = {"train": "train/wildguard_train.parquet", "test": "test/wildguard_test.parquet"}


def validate_dataset_manifest(manifest: JsonObject) -> None:
    names = validate_files(manifest)
    revision = manifest.get("revision", "")
    if manifest.get("dataset_id") == CORPUS_ID:
        if (
            len(revision) != 64
            or any(c not in "0123456789abcdef" for c in revision)
            or manifest.get("source_files") != CORPUS_FILES
            or manifest.get("preprocessing", {}).get("version") != 1
            or not {*CORPUS_FILES, "README.md"} <= names
        ):
            raise ValueError("Expected a versioned four-source refusal-direction corpus.")
        sources = manifest.get("sources", [])
        offset = 0
        if len(sources) != len(CORPUS_FILES):
            raise ValueError("Corpus source metadata is incomplete.")
        for filename, source in zip(CORPUS_FILES, sources):
            if (
                source.get("file") != filename
                or source.get("row_offset") != offset
                or type(source.get("row_count")) is not int
                or source["row_count"] < 0
            ):
                raise ValueError("Corpus source offsets or counts are inconsistent.")
            offset += source["row_count"]
        if offset != manifest.get("row_count"):
            raise ValueError("Corpus total row count is inconsistent.")
        return
    if (
        manifest.get("dataset_id") != DATASET_ID
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("Expected a pinned WildGuardMix or refusal-direction dataset manifest.")
    if manifest.get("splits") != SPLITS or not {*SPLITS.values(), "README.md"} <= names:
        raise ValueError("Dataset snapshot must contain both official splits and the dataset card.")


def dataset_name(manifest: JsonObject) -> str:
    return CORPUS_NAME if manifest["dataset_id"] == CORPUS_ID else "wildguardmix"


def source_files(manifest: JsonObject, split: str = "train") -> list[str]:
    if manifest["dataset_id"] == CORPUS_ID:
        if split not in {"train", "validation"}:
            raise ValueError(
                "The refusal-direction corpus has train/validation membership, not a test split."
            )
        return list(CORPUS_FILES)
    return [manifest["splits"]["train" if split == "validation" else split]]


def prompt_schema() -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("prompt", pa.string(), nullable=False),
            pa.field("prompt_harm_label", pa.string(), nullable=False),
            pa.field("adversarial", pa.bool_(), nullable=False),
        ]
    )


def read_prompt_rows(
    directory: PathInput, manifest: JsonObject, split: str = "train"
) -> list[JsonObject]:
    """Global source indices are positions in this fixed file-order concatenation."""
    import pyarrow.parquet as pq

    rows: list[JsonObject] = []
    for filename in source_files(manifest, split):
        table = pq.read_table(Path(directory) / filename, columns=PROMPT_COLUMNS)
        if manifest["dataset_id"] == CORPUS_ID and not table.schema.equals(prompt_schema()):
            raise ValueError(f"Unexpected normalized prompt schema in {filename}.")
        rows.extend(table.to_pylist())
    return rows


def download_dataset(s3: S3Client, uri: str, cache_dir: PathInput) -> Path:
    return download_snapshot(s3, uri, cache_dir, validate_dataset_manifest)
