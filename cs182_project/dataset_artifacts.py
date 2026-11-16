"""Pinned WildGuardMix snapshot format and checksummed S3 caching."""

from .model_artifacts import download_snapshot, validate_files

DATASET_ID = "allenai/wildguardmix"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
SPLITS = {"train": "train/wildguard_train.parquet", "test": "test/wildguard_test.parquet"}


def validate_dataset_manifest(manifest):
    names = validate_files(manifest)
    revision = manifest.get("revision", "")
    if (manifest.get("dataset_id") != DATASET_ID or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("Expected a pinned allenai/wildguardmix dataset manifest.")
    if manifest.get("splits") != SPLITS or not {*SPLITS.values(), "README.md"} <= names:
        raise ValueError("Dataset snapshot must contain both official splits and the dataset card.")


def download_dataset(s3, uri, cache_dir):
    return download_snapshot(s3, uri, cache_dir, validate_dataset_manifest)
