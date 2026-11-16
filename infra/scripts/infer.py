#!/usr/bin/env python3
"""Load an archived Qwen chat model from S3 and print its generated response."""

import argparse
import sys

from cs182_project.inference import generate
from cs182_project.model_artifacts import download_model, parse_s3_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-s3-uri", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--cache-dir", default="/cache/models")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 1 and 4096")
    try:
        parse_s3_uri(args.model_s3_uri)
    except ValueError as error:
        parser.error(str(error))
    try:
        import boto3

        model_dir = download_model(boto3.client("s3"), args.model_s3_uri, args.cache_dir)
        print(generate(model_dir, args.prompt, args.max_new_tokens, args.device))
    except Exception as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
