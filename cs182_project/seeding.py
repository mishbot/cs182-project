"""Archive Hugging Face model and dataset snapshots in S3."""

import json
import logging
from pathlib import Path

from .dataset_artifacts import DATASET_ID, DATASET_URL, SPLITS, validate_dataset_manifest
from .model_artifacts import MANIFEST, MODEL_ID, MODEL_REVISION, sha256, validate_manifest

logger = logging.getLogger(__name__)


def validate_model_destination(revision, prefix):
    """Validate a pinned model revision and destination prefix."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("--revision must be a full lowercase 40-character commit SHA")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("--prefix must be a nonempty S3 snapshot prefix without '..'")


def seed_model(s3, bucket, revision=MODEL_REVISION, prefix=None, cache_dir="/cache/huggingface"):
    """Download a pinned model and publish its snapshot; return its S3 URI."""
    prefix = (prefix or f"models/Qwen1.5-1.8B-Chat/{revision}").strip("/")
    validate_model_destination(revision, prefix)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=MODEL_ID, revision=revision, cache_dir=cache_dir)
    return seed(s3, snapshot, bucket, prefix, revision)


def seed(s3, snapshot, bucket, prefix, revision):
    """Publish an existing local model snapshot and return its S3 URI."""
    validate_model_destination(revision, prefix)
    # Only an absent manifest may be seeded. Other AWS errors must not be hidden.
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        raise ValueError("Snapshot already published. Use a new prefix for a new model revision.")
    root = Path(snapshot)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"schema_version": 1, "model_id": MODEL_ID, "revision": revision, "files": files}
    validate_manifest(manifest)
    for entry in files:
        logger.info("Uploading %s", entry["path"])
        s3.upload_file(str(root / entry["path"]), bucket, f"{prefix}/{entry['path']}")
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/{MANIFEST}",
        Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
        ContentType="application/json", IfNoneMatch="*",
    )
    return f"s3://{bucket}/{prefix}"


def seed_dataset(s3, bucket, revision="main", cache_dir=".cache/huggingface", hub=None):
    """Resolve once, preserve upstream files, and publish the completion marker last."""
    from botocore.exceptions import ClientError

    if hub is None:
        import huggingface_hub as hub

    resolved = hub.HfApi().dataset_info(DATASET_ID, revision=revision).sha
    if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved):
        raise ValueError("Hugging Face did not return a full dataset commit SHA.")
    prefix = f"datasets/wildguardmix/{resolved}"
    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        raise ValueError(f"Dataset snapshot already published: s3://{bucket}/{prefix}")

    files = []
    paths = {}
    for filename in [*SPLITS.values(), "README.md"]:
        path = Path(hub.hf_hub_download(
            repo_id=DATASET_ID, repo_type="dataset", revision=resolved,
            filename=filename, cache_dir=cache_dir,
        ))
        paths[filename] = path
        files.append({"path": filename, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1, "dataset_id": DATASET_ID, "revision": resolved,
        "splits": SPLITS, "files": files,
        "attribution": {"source": DATASET_URL, "license": "ODC-By", "dataset_card": "README.md"},
    }
    validate_dataset_manifest(manifest)
    for filename, path in paths.items():
        logger.info("Uploading %s", filename)
        s3.upload_file(str(path), bucket, f"{prefix}/{filename}")
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/{MANIFEST}",
        Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
        ContentType="application/json", IfNoneMatch="*",
    )
    return f"s3://{bucket}/{prefix}"

