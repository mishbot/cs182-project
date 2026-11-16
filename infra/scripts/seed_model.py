#!/usr/bin/env python3
"""Archive a pinned Hugging Face model snapshot in S3, publishing the manifest last."""

import argparse
import logging
import sys

from cs182_project.model_artifacts import MODEL_REVISION
from cs182_project.seeding import seed_model, validate_model_destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION, help="Full 40-character Hugging Face commit SHA")
    parser.add_argument("--prefix", help="New S3 prefix; defaults to models/Qwen1.5-1.8B-Chat/REVISION")
    parser.add_argument("--cache-dir", default="/cache/huggingface")
    args = parser.parse_args()
    prefix = (args.prefix or f"models/Qwen1.5-1.8B-Chat/{args.revision}").strip("/")
    try:
        validate_model_destination(args.revision, prefix)
    except ValueError as error:
        parser.error(str(error))
    try:
        import boto3

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(seed_model(boto3.client("s3"), args.bucket, args.revision, prefix, args.cache_dir))
    except Exception as error:
        print(f"Model seeding failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
