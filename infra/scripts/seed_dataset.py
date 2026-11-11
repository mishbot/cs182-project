#!/usr/bin/env python3
"""Archive a pinned WildGuardMix revision in the existing S3 artifact bucket."""

import argparse
import json
from pathlib import Path
import sys

from dataset_artifacts import DATASET_ID, DATASET_URL, SPLITS, validate_dataset_manifest
from model_artifacts import MANIFEST, sha256


def seed_dataset(s3, bucket, revision, cache_dir, hub):
    """Resolve once, preserve upstream files, and publish the completion marker last."""
    from botocore.exceptions import ClientError

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
        print(f"Uploading {filename}", file=sys.stderr)
        s3.upload_file(str(path), bucket, f"{prefix}/{filename}")
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/{MANIFEST}",
        Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
        ContentType="application/json", IfNoneMatch="*",
    )
    return f"s3://{bucket}/{prefix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", default="main", help="Upstream ref resolved to a full SHA before downloading")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    args = parser.parse_args()
    try:
        import boto3
        import huggingface_hub

        print(seed_dataset(boto3.client("s3"), args.bucket, args.revision, args.cache_dir, huggingface_hub))
    except Exception as error:
        print(f"Dataset seeding failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
