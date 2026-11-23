#!/usr/bin/env python3
"""Train a defender locally, or retry statistics from a durable run directory."""

import argparse
import json
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-s3-uri", default=os.environ.get("MODEL_URI"))
    parser.add_argument("--dataset-s3-uri", default=os.environ.get("DATASET_URI"))
    parser.add_argument("--partition-s3-uri")
    parser.add_argument("--stats-s3-uri", default=os.environ.get("TRAINING_STATS_URI"))
    parser.add_argument("--model-output-s3-uri")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", default="reports/training")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config-json", default="{}", help="JSON training configuration overrides")
    parser.add_argument("--dry-run", action="store_true", help="Use a short, billable training run")
    parser.add_argument("--retry-dir", help="Upload pending statistics without loading a model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.retry_dir:
        from cs182_project.training import DRY_RUN, training_config
        from cs182_project.model_artifacts import parse_s3_uri

        for name in ("model_s3_uri", "dataset_s3_uri", "run_name"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required")
        model_bucket, _ = parse_s3_uri(args.model_s3_uri)
        parse_s3_uri(args.dataset_s3_uri)
        if not model_bucket.endswith("-models") and not args.stats_s3_uri:
            parser.error("--stats-s3-uri is required for a nonstandard models bucket")
        root = model_bucket.removesuffix("-models")
        args.stats_s3_uri = args.stats_s3_uri or f"s3://{root}-training-stats/runs"
        args.model_output_s3_uri = args.model_output_s3_uri or f"s3://{model_bucket}/models/trained"
        for uri in (args.stats_s3_uri, args.model_output_s3_uri, args.partition_s3_uri):
            if uri:
                parse_s3_uri(uri)
        overrides = {**(DRY_RUN if args.dry_run else {}), **json.loads(args.config_json)}
        training_config(overrides)
    import boto3
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(
        retries={"total_max_attempts": 3, "mode": "standard"},
        connect_timeout=5, read_timeout=10,
    ))
    if args.retry_dir:
        from cs182_project.training_stats import TrainingStats, write_json

        stats = TrainingStats(s3, args.retry_dir)
        pending = stats.flush()
        summary_path = Path(args.retry_dir) / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary.update(pending_batches=pending, statistics_delivery="pending" if pending else "complete")
        write_json(summary_path, summary)
        pending = stats.flush()
        print(json.dumps({"pending_batches": pending}))
        return 1 if pending else 0
    from cs182_project.training import train
    summary = train(
        s3, model_s3_uri=args.model_s3_uri, dataset_s3_uri=args.dataset_s3_uri,
        stats_s3_uri=args.stats_s3_uri,
        model_output_s3_uri=args.model_output_s3_uri,
        run_name=args.run_name, output_dir=args.output_dir, partition_s3_uri=args.partition_s3_uri,
        device=args.device, overrides=overrides,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
