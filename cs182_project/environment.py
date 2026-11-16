"""Discover the project's published S3 snapshots and newest tagged ECR image."""

import json
import os
import re

from .dataset_artifacts import validate_dataset_manifest
from .model_artifacts import MANIFEST, MODEL_ID, MODEL_REVISION, validate_manifest


def _read_manifest(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return json.load(response["Body"])
    finally:
        response["Body"].close()


def _latest_dataset(s3, bucket):
    """Select by completion-marker publication time, never by SHA ordering."""
    pattern = re.compile(rf"datasets/wildguardmix/([0-9a-f]{{40}})/{re.escape(MANIFEST)}")
    candidates = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="datasets/wildguardmix/"):
        for entry in page.get("Contents", []):
            if pattern.fullmatch(entry["Key"]):
                candidates.append(entry)
    if not candidates:
        raise ValueError(f"No completed WildGuardMix snapshots found in s3://{bucket}/datasets/wildguardmix/.")
    latest = max(candidates, key=lambda entry: (entry["LastModified"], entry["Key"]))
    manifest = _read_manifest(s3, bucket, latest["Key"])
    validate_dataset_manifest(manifest)
    revision = pattern.fullmatch(latest["Key"])[1]
    if manifest["revision"] != revision:
        raise ValueError("The newest dataset manifest's revision does not match its S3 prefix.")
    return f"s3://{bucket}/datasets/wildguardmix/{revision}"


def _latest_image(ecr, repository):
    """Select the newest tagged image across all result pages."""
    candidates = []
    for page in ecr.get_paginator("describe_images").paginate(
        repositoryName=repository, filter={"tagStatus": "TAGGED"},
    ):
        candidates.extend(image for image in page.get("imageDetails", []) if image.get("imageTags"))
    if not candidates:
        raise ValueError(f"No tagged runtime images found in ECR repository {repository}.")
    latest = max(candidates, key=lambda image: (image["imagePushedAt"], image["imageDigest"]))
    # Multiple tags for one image are equivalent. Choose deterministically.
    tag = sorted(latest["imageTags"])[0]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("ECR returned an invalid image tag.")
    result = ecr.describe_repositories(repositoryNames=[repository])
    return f"{result['repositories'][0]['repositoryUri']}:{tag}"


def discover_modal_environment(session, *, project="cs182-cddr", region=None):
    """Resolve four shell variables without requiring local OpenTofu state.

    Resource names follow infra/persistent. Dataset 'latest' means the newest
    completion marker in S3; image 'latest' means the newest ECR push with a tag.
    A malformed newest manifest or any AWS error is reported, never hidden by
    falling back to an older snapshot or a different account.
    """
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,29}", project):
        raise ValueError("Project must be 3–30 lowercase letters, digits, or hyphens, beginning with a letter.")
    region = region or session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
    if not re.fullmatch(r"[a-z0-9-]+", region):
        raise ValueError("Invalid AWS region.")
    account = session.client("sts", region_name=region).get_caller_identity()["Account"]
    if not re.fullmatch(r"[0-9]{12}", account):
        raise ValueError("AWS STS did not return a 12-digit account ID.")
    prefix = f"{project}-{account}-{region}"
    s3 = session.client("s3", region_name=region)
    ecr = session.client("ecr", region_name=region)
    model_prefix = f"models/Qwen1.5-1.8B-Chat/{MODEL_REVISION}"
    model = _read_manifest(s3, f"{prefix}-models", f"{model_prefix}/{MANIFEST}")
    validate_manifest(model)
    if model.get("model_id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("The model manifest does not match the project's pinned model and revision.")
    return {
        "CS182_MODAL_IMAGE": _latest_image(ecr, f"{project}-runtime"),
        "MODEL_URI": f"s3://{prefix}-models/{model_prefix}",
        "DATASET_URI": _latest_dataset(s3, f"{prefix}-datasets"),
        "OUTPUT_URI": f"s3://{prefix}-activations/runs",
    }
