#!/usr/bin/env python3
"""Archive a pinned Hugging Face model snapshot in S3, publishing the manifest last."""

import argparse
import json
from pathlib import Path
import sys

from model_artifacts import MANIFEST, MODEL_ID, MODEL_REVISION, sha256, validate_manifest


def seed(s3, snapshot, bucket, prefix, revision):
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
        print(f"Uploading {entry['path']}", file=sys.stderr)
        s3.upload_file(str(root / entry["path"]), bucket, f"{prefix}/{entry['path']}")
    s3.put_object(
        Bucket=bucket, Key=f"{prefix}/{MANIFEST}",
        Body=json.dumps(manifest, sort_keys=True, indent=2).encode(),
        ContentType="application/json", IfNoneMatch="*",
    )
    return f"s3://{bucket}/{prefix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION, help="Full 40-character Hugging Face commit SHA")
    parser.add_argument("--prefix", help="New S3 prefix; defaults to models/Qwen1.5-1.8B-Chat/REVISION")
    parser.add_argument("--cache-dir", default="/cache/huggingface")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full lowercase 40-character commit SHA")
    prefix = (args.prefix or f"models/Qwen1.5-1.8B-Chat/{args.revision}").strip("/")
    if not prefix or ".." in prefix.split("/"):
        parser.error("--prefix must be a nonempty S3 snapshot prefix without '..'")
    try:
        import boto3
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=MODEL_ID, revision=args.revision, cache_dir=args.cache_dir)
        print(seed(boto3.client("s3"), snapshot, args.bucket, prefix, args.revision))
    except Exception as error:
        print(f"Model seeding failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
