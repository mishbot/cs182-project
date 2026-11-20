"""Faithful checkpoint import from an existing mounted filesystem into the models bucket."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .model_artifacts import publish_model_snapshot

if TYPE_CHECKING:
    from boto3.session import Session
    from mypy_boto3_s3 import S3Client


def model_bucket(
    session: Session, *, project: str = "cs182-cddr", region: str | None = None
) -> str:
    """Resolve the existing stack's bucket without ECR or local OpenTofu state."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,29}", project):
        raise ValueError("Invalid project name.")
    region = region or session.region_name or "us-east-1"
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise ValueError("Invalid region.")
    identity = session.client("sts", region_name=region).get_caller_identity()
    account = identity["Account"]
    if not re.fullmatch(r"[0-9]{12}", account):
        raise ValueError("AWS STS did not return a valid account ID.")
    return f"{project}-{account}-{region}-models"


def import_model(
    s3: S3Client,
    directory: PathInput,
    bucket: str,
    *,
    name: str = "Qwen-1_8B-Chat-merged",
    source: JsonObject,
) -> str:
    """Copy a checkpoint unchanged, using bounded multipart transfers and an explicit format."""
    from boto3.s3.transfer import TransferConfig

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Model name must use letters, digits, '.', '_', or '-'.")
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    identifier = uuid.uuid4().hex
    model_name = f"{name}-{identifier}"
    metadata: JsonObject = {
        "source": source,
        "architecture": {
            key: config.get(key) for key in ("model_type", "architectures", "tokenizer_class")
        },
    }
    if config.get("model_type") == "qwen":
        metadata["model_format"] = "legacy_qwen"
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=4,
    )
    return publish_model_snapshot(
        s3,
        root,
        bucket,
        f"models/{model_name}",
        model_id=model_name,
        revision=identifier,
        artifact_kind="modal_volume_import",
        provenance=metadata,
        transfer_config=transfer,
        verify_uploaded_sizes=True,
    )
