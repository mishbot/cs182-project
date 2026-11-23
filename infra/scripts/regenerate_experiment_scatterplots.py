#!/usr/bin/env python3
"""Download HTML reports and regenerate scatterplots for a completed experiment."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from cs182_project.activation_report import (
    analysis_stages,
    download_analysis_plot_artifacts,
    render_scatterplot_pngs,
)
from cs182_project.model_artifacts import parse_s3_uri
from cs182_project.report_paths import model_role


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_json", type=Path, help="reports/activations experiment JSON")
    parser.add_argument("--output-root", type=Path, default=Path("report_v1/images"))
    args = parser.parse_args()
    try:
        experiment = json.loads(args.experiment_json.expanduser().read_text(encoding="utf-8"))
        experiment_id = experiment["experiment_id"]
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
            raise ValueError("Experiment JSON has no valid experiment_id.")
        role = model_role(experiment["original_model"])
        completed = analysis_stages(experiment, name_prefix=f"{role}qwen")

        import boto3

        session = boto3.Session(region_name=experiment.get("region") or os.environ.get("AWS_REGION"))
        s3 = session.client("s3")
        output_dir = args.output_root.expanduser() / f"{role}-experiment-{experiment_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stage, name_prefix in completed:
            bucket, key = parse_s3_uri(experiment[stage])
            if not key.endswith("/report.html"):
                key += "/report.html"
            report = output_dir / f"{name_prefix}_report.html"
            s3.download_file(bucket, key, str(report))
            print(report, flush=True)

        for stage, name_prefix in completed:
            metrics, directions = download_analysis_plot_artifacts(s3, experiment[stage])
            for path in render_scatterplot_pngs(
                metrics, directions, output_dir, name_prefix=name_prefix
            ):
                print(path)
    except Exception as error:
        print(f"Scatterplot regeneration failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
