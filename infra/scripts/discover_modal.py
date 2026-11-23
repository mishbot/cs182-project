#!/usr/bin/env python3
"""Resolve Modal configuration for the sourceable infra/modal/env.sh helper."""

import argparse
import os
import sys

from cs182_project.environment import discover_modal_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("CS182_PROJECT", "cs182-cddr"))
    parser.add_argument("--dataset", choices=["refusal-direction", "wildguardmix"], default="refusal-direction")
    parser.add_argument("--region", help="Defaults to the current AWS session's region")
    args = parser.parse_args()
    try:
        import boto3

        values = discover_modal_environment(boto3.Session(), project=args.project, region=args.region, dataset=args.dataset)
        ordered = [values[name] for name in ["CS182_MODAL_IMAGE", "MODEL_URI", "DATASET_URI", "OUTPUT_URI", "ANALYSIS_URI", "TRAINING_STATS_URI"]]
        if any(not value or any(char in value for char in "\t\r\n") for value in ordered):
            raise ValueError("Discovered values must be nonempty single-line strings.")
        print("\t".join(ordered))
    except Exception as error:
        print(f"Modal environment discovery failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
