"""Discover the project's published S3 snapshots and newest tagged ECR image."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from ._types import JsonObject
from .dataset_artifacts import CORPUS_ID, CORPUS_NAME, DATASET_ID, validate_dataset_manifest
from .model_artifacts import MANIFEST, MODEL_ID, MODEL_NAME, MODEL_REVISION, validate_manifest

if TYPE_CHECKING:
    from datetime import datetime

    from boto3.session import Session
    from mypy_boto3_ecr import ECRClient
    from mypy_boto3_s3 import S3Client


def _read_manifest(s3: S3Client, bucket: str, key: str) -> JsonObject:
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return json.load(response["Body"])
    finally:
        response["Body"].close()


def _latest_dataset(s3: S3Client, bucket: str, dataset: str = "refusal-direction") -> str:
    """Select by completion-marker publication time, never by SHA ordering."""
    if dataset not in {CORPUS_NAME, "wildguardmix"}:
        raise ValueError("Dataset must be refusal-direction or wildguardmix.")
    size = 40 if dataset == "wildguardmix" else 64
    pattern = re.compile(rf"datasets/{dataset}/([0-9a-f]{{{size}}})/{re.escape(MANIFEST)}")
    candidates: list[tuple[datetime, str]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"datasets/{dataset}/"
    ):
        for entry in page.get("Contents", []):
            key = entry.get("Key", "")
            if pattern.fullmatch(key):
                modified = entry.get("LastModified")
                if modified is None:
                    raise ValueError("A dataset completion marker has no publication timestamp.")
                candidates.append((modified, key))
    if not candidates:
        raise ValueError(
            f"No completed {dataset} snapshots found in s3://{bucket}/datasets/{dataset}/. "
            "Run infra/scripts/seed_dataset.py with the selected --dataset first."
        )
    _, latest = max(candidates)
    manifest = _read_manifest(s3, bucket, latest)
    validate_dataset_manifest(manifest)
    expected_id = DATASET_ID if dataset == "wildguardmix" else CORPUS_ID
    if manifest["dataset_id"] != expected_id:
        raise ValueError("Dataset identity differs from the selected corpus.")
    match = pattern.fullmatch(latest)
    assert match is not None  # Candidates were filtered with this same pattern.
    revision = match[1]
    if manifest["revision"] != revision:
        raise ValueError("The newest dataset manifest's revision does not match its S3 prefix.")
    return f"s3://{bucket}/datasets/{dataset}/{revision}"


def _latest_image(ecr: ECRClient, repository: str) -> str:
    """Select the newest tagged image across all result pages."""
    candidates: list[tuple[datetime, str, list[str]]] = []
    for page in ecr.get_paginator("describe_images").paginate(
        repositoryName=repository,
        filter={"tagStatus": "TAGGED"},
    ):
        for image in page.get("imageDetails", []):
            tags = image.get("imageTags")
            if not tags:
                continue
            pushed_at, digest = image.get("imagePushedAt"), image.get("imageDigest")
            if pushed_at is None or digest is None:
                raise ValueError("A tagged ECR image has no publication timestamp or digest.")
            candidates.append((pushed_at, digest, tags))
    if not candidates:
        raise ValueError(f"No tagged runtime images found in ECR repository {repository}.")
    _, _, tags = max(candidates, key=lambda image: (image[0], image[1]))
    # Multiple tags for one image are equivalent. Choose deterministically.
    tag = min(tags)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("ECR returned an invalid image tag.")
    result = ecr.describe_repositories(repositoryNames=[repository])
    uri = result["repositories"][0].get("repositoryUri")
    if uri is None:
        raise ValueError("ECR did not return a repository URI.")
    return f"{uri}:{tag}"


def discover_modal_environment(
    session: Session,
    *,
    project: str = "cs182-cddr",
    region: str | None = None,
    dataset: str = "refusal-direction",
) -> dict[str, str]:
    """Resolve shell variables without requiring local OpenTofu state.

    Resource names follow infra/persistent. Dataset 'latest' means the newest
    completion marker in S3; image 'latest' means the newest ECR push with a tag.
    A malformed newest manifest or any AWS error is reported, never hidden by
    falling back to an older snapshot or a different account.
    """
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,29}", project):
        raise ValueError(
            "Project must be 3–30 lowercase letters, digits, or hyphens, beginning with a letter."
        )
    region = region or session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise ValueError("Invalid AWS region.")
    account = session.client("sts", region_name=region).get_caller_identity()["Account"]
    if not re.fullmatch(r"[0-9]{12}", account):
        raise ValueError("AWS STS did not return a 12-digit account ID.")
    prefix = f"{project}-{account}-{region}"
    s3 = session.client("s3", region_name=region)
    ecr = session.client("ecr", region_name=region)
    model_prefix = f"models/{MODEL_NAME}/{MODEL_REVISION}"
    model_bucket = f"{prefix}-models"
    model_key = f"{model_prefix}/{MANIFEST}"
    try:
        model = _read_manifest(s3, model_bucket, model_key)
    except s3.exceptions.NoSuchKey as error:
        raise ValueError(
            f"The pinned original Qwen model is not seeded: s3://{model_bucket}/{model_key}. "
            "Run infra/scripts/run_qwen_experiment.py with --seed-model to seed it on Modal "
            "before discovery, or run infra/modal/seed_model.py separately."
        ) from error
    validate_manifest(model)
    if (
        model.get("model_id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("upstream", {}).get("provider") != "modelscope"
    ):
        raise ValueError(
            "The model manifest does not match the project's pinned model and revision."
        )
    return {
        "CS182_MODAL_IMAGE": _latest_image(ecr, f"{project}-runtime"),
        "MODEL_URI": f"s3://{prefix}-models/{model_prefix}",
        "DATASET_URI": _latest_dataset(s3, f"{prefix}-datasets", dataset),
        "OUTPUT_URI": f"s3://{prefix}-activations/runs",
        "ANALYSIS_URI": f"s3://{prefix}-analyses/runs",
        "TRAINING_STATS_URI": f"s3://{prefix}-training-stats/runs",
    }
