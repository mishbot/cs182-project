#!/usr/bin/env python3
"""Analyze a completed activation run and write a self-contained Plotly report."""

import argparse
import logging
import os
from pathlib import Path
import sys

from cs182_project.activation_analysis import analyze_activations
from cs182_project.activation_report import write_analysis_report
from cs182_project.activation_runs import activation_root, latest_activation_run, load_activation_run
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.folds < 2 or args.seed < 0:
        parser.error("--folds must be at least 2; --seed must be nonnegative")
    for uri in [args.run_s3_uri, args.activation_root_s3_uri]:
        if uri:
            try:
                parse_s3_uri(uri)
            except ValueError as error:
                parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import boto3

        session = boto3.Session(region_name=args.region)
        region = args.region or session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
        s3 = session.client("s3", region_name=region)
        run_uri = args.run_s3_uri
        if not run_uri:
            root = args.activation_root_s3_uri or os.environ.get("OUTPUT_URI")
            root = root or activation_root(session, project=args.project, region=region)
            run_uri = latest_activation_run(s3, root)
        run = load_activation_run(s3, run_uri)
        analysis = analyze_activations(run, folds=args.folds, seed=args.seed)
        output = args.output_dir or Path("reports/activations") / run.manifest["run_id"]
        print(write_analysis_report(analysis, output))
    except Exception as error:
        print(f"Activation analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
