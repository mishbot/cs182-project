#!/usr/bin/env python3
"""Sample WildGuardMix prompts and upload final-token layer activations to S3."""

import argparse
import logging
import sys

from cs182_project.activations import collect
from cs182_project.model_artifacts import parse_s3_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["model-s3-uri", "dataset-s3-uri", "output-s3-uri"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--partition-s3-uri", help="Overrides the default 80/20, seed-0 membership file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", default="/cache")
    args = parser.parse_args()
    if args.sample_size < 4 or args.batch_size < 1 or args.max_input_tokens < 1:
        parser.error("--sample-size must be >= 4; --batch-size and --max-input-tokens must be positive")
    for uri in [args.model_s3_uri, args.dataset_s3_uri, args.output_s3_uri]:
        try:
            parse_s3_uri(uri)
        except ValueError as error:
            parser.error(str(error))
    try:
        import boto3

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        print(collect(boto3.client("s3"), **vars(args)))
    except Exception as error:
        print(f"Activation collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
