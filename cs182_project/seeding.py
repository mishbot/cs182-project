"""Archive the pinned ModelScope original model and Hugging Face datasets in S3."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from os import fspath
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._types import ArtifactFile, DatasetHub, JsonObject, PathInput
from .dataset_artifacts import DATASET_ID, DATASET_URL, SPLITS, validate_dataset_manifest
from .model_artifacts import (
    MANIFEST,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    publish_model_snapshot,
    sha256,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


def validate_model_destination(revision: str, prefix: str) -> None:
    """Validate a pinned model revision and destination prefix."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("--revision must be a full lowercase 40-character commit SHA")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("--prefix must be a nonempty S3 snapshot prefix without '..'")


def _check_unpublished(s3: S3Client, bucket: str, prefix: str) -> None:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        raise ValueError("Snapshot already published. Use a new prefix for a new model revision.")


def download_original_model(revision: str, cache_dir: PathInput) -> Path:
    """Checkout an immutable ModelScope commit and verify its Git LFS content identities."""
    root = Path(cache_dir).expanduser().resolve() / MODEL_NAME / revision
    root.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    if not (root / ".git").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                MODEL_REPOSITORY,
                str(root),
            ],
            env=environment,
            check=True,
            stdout=sys.stderr,
        )

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *arguments], env=environment)

    git("lfs", "version")
    git("fetch", "--depth", "1", "origin", revision)
    git("checkout", "--detach", revision)
    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("ModelScope checkout does not match the requested immutable revision.")
    git("lfs", "pull", "origin")
    for entry in git("lfs", "ls-files", "--long").decode().splitlines():
        digest, relative = entry.split(" ", 1)
        relative = relative[2:]  # '* path' or '- path' in git-lfs ls-files output.
        if sha256(root / relative) != digest:
            raise ValueError(f"Git LFS checksum mismatch: {relative}")
    return root


def seed_model(
    s3: S3Client,
    bucket: str,
    revision: str = MODEL_REVISION,
    prefix: str | None = None,
    cache_dir: PathInput = "/cache/modelscope",
) -> str:
    """Download the original ModelScope model; never seed from the fine-tuned import."""
    prefix = (prefix or f"models/{MODEL_NAME}/{revision}").strip("/")
    validate_model_destination(revision, prefix)
    _check_unpublished(s3, bucket, prefix)
    snapshot = download_original_model(revision, cache_dir)
    return seed(s3, snapshot, bucket, prefix, revision)


def seed(s3: S3Client, snapshot: PathInput, bucket: str, prefix: str, revision: str) -> str:
    """Publish model assets without Git/cache metadata, preserving all checkpoint bytes."""
    validate_model_destination(revision, prefix)
    _check_unpublished(s3, bucket, prefix)
    root = Path(snapshot)
    with tempfile.TemporaryDirectory(prefix="cs182-original-qwen-") as temporary:
        staging = Path(temporary)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and not any(part.startswith(".") for part in relative.parts):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        return publish_model_snapshot(
            s3,
            staging,
            bucket,
            prefix,
            model_id=MODEL_ID,
            revision=revision,
            artifact_kind="upstream_model",
            provenance={
                "model_format": "legacy_qwen",
                "upstream": {
                    "provider": "modelscope",
                    "repository": MODEL_REPOSITORY,
                    "revision": revision,
                },
            },
        )


def seed_dataset(
    s3: S3Client,
    bucket: str,
    revision: str = "main",
    cache_dir: PathInput = ".cache/huggingface",
    hub: DatasetHub | None = None,
    *,
    dataset: str = "refusal-direction",
) -> str:
    """Seed the default processed corpus, or explicitly archive WildGuardMix."""
    if dataset == "refusal-direction":
        from .processed_datasets import seed_corpus

        if revision != "main" or hub is not None:
            raise ValueError(
                "--revision/Hugging Face overrides apply only to --dataset wildguardmix."
            )
        return seed_corpus(s3, bucket, cache_dir)
    if dataset != "wildguardmix":
        raise ValueError("Dataset must be refusal-direction or wildguardmix.")
    from botocore.exceptions import ClientError

    if hub is None:
        import huggingface_hub

        hub = cast(DatasetHub, huggingface_hub)

    resolved = hub.HfApi().dataset_info(DATASET_ID, revision=revision).sha
    if (
        not isinstance(resolved, str)
        or len(resolved) != 40
        or any(c not in "0123456789abcdef" for c in resolved)
    ):
        raise ValueError("Hugging Face did not return a full dataset commit SHA.")
    prefix = f"datasets/wildguardmix/{resolved}"
    try:
        s3.head_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        raise ValueError(f"Dataset snapshot already published: s3://{bucket}/{prefix}")

    files: list[ArtifactFile] = []
    paths: dict[str, Path] = {}
    for filename in [*SPLITS.values(), "README.md"]:
        path = Path(
            hub.hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=resolved,
                filename=filename,
                cache_dir=fspath(cache_dir),
            )
        )
        paths[filename] = path
        files.append({"path": filename, "size": path.stat().st_size, "sha256": sha256(path)})
    manifest: JsonObject = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "revision": resolved,
        "splits": SPLITS,
        "files": files,
        "attribution": {"source": DATASET_URL, "license": "ODC-By", "dataset_card": "README.md"},
    }
    validate_dataset_manifest(manifest)
    for filename, path in paths.items():
        logger.info("Uploading %s", filename)
        s3.upload_file(str(path), bucket, f"{prefix}/{filename}")
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{MANIFEST}",
        Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
        ContentType="application/json",
        IfNoneMatch="*",
    )
    return f"s3://{bucket}/{prefix}"
