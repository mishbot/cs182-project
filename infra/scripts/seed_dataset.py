#!/usr/bin/env python3
"""Prepare and upload AdvBench, MaliciousInstruct, TDC2023, and Alpaca for collection."""

import argparse
import logging
import sys

from cs182_project.seeding import seed_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Dedicated datasets bucket (dataset_bucket OpenTofu output)")
    parser.add_argument("--dataset", choices=["refusal-direction", "wildguardmix"], default="refusal-direction")
    parser.add_argument("--revision", default="main", help="WildGuardMix only: upstream ref resolved to a full SHA")
    parser.add_argument("--cache-dir", default=".cache/datasets")
    args = parser.parse_args()
    try:
        import boto3

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(seed_dataset(boto3.client("s3"), args.bucket, args.revision, args.cache_dir, dataset=args.dataset))
    except Exception as error:
        print(f"Dataset seeding failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
