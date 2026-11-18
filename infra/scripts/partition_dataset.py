#!/usr/bin/env python3
"""Publish reproducible train/validation row membership for an archived dataset."""

import argparse
import os
import sys

from cs182_project.dataset_partition import publish_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-s3-uri", default=os.environ.get("DATASET_URI"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=".cache")
    args = parser.parse_args()
    if not args.dataset_s3_uri:
        parser.error("Set DATASET_URI or pass --dataset-s3-uri")
    try:
        import boto3

        print(publish_partition(boto3.client("s3"), **vars(args)))
    except Exception as error:
        print(f"Dataset partition failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
