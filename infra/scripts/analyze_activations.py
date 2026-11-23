#!/usr/bin/env python3
"""Analyze a completed activation run and write a self-contained Plotly report."""

import argparse
import logging
import os
from pathlib import Path
import sys

from cs182_project.analysis_jobs import run_analysis
from cs182_project.model_artifacts import parse_s3_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-s3-uri", help="Exact run directory or manifest URI")
    source.add_argument("--activation-root-s3-uri", help="Runs prefix; defaults to OUTPUT_URI or the project's activation bucket")
    parser.add_argument("--project", default=os.environ.get("CS182_PROJECT", "cs182-cddr"))
    parser.add_argument("--region")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="Retain local artifacts in addition to the S3 export")
    parser.add_argument("--output-s3-uri", help="Results prefix; defaults to ANALYSIS_URI or the project's analyses bucket")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--include-ridge", action="store_true", help="Fit per-layer Ridge models for the refusal metric")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    if args.folds < 2 or args.seed < 0:
        parser.error("--folds must be at least 2; --seed must be nonnegative")
    for uri in [args.run_s3_uri, args.activation_root_s3_uri, args.output_s3_uri]:
        if uri:
            try:
                parse_s3_uri(uri)
            except ValueError as error:
                parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import boto3

        print(run_analysis(boto3.Session(region_name=args.region), **vars(args)))
    except Exception as error:
        print(f"Activation analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
