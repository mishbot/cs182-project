#!/usr/bin/env python3
"""Download HTML reports and regenerate scatterplots for a completed experiment."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from cs182_project.activation_report import (
    download_analysis_plot_artifacts,
    render_scatterplot_pngs,
)
from cs182_project.model_artifacts import parse_s3_uri


def analysis_stages(experiment: dict) -> list[tuple[str, str]]:
    """Return both required analysis stages, failing clearly if either is unavailable."""
    stages = experiment.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("Experiment JSON has no stages object.")
    result = []
    for stage, name_prefix in (
        ("original_analysis", "originalqwen"),
        ("orthogonalized_analysis", "originalqwen_orthogonalized"),
    ):
        state = stages.get(stage)
        if not isinstance(state, dict) or state.get("status") != "complete":
            raise ValueError(f"Experiment stage {stage} is not complete.")
        if not isinstance(experiment.get(stage), str):
            raise ValueError(f"Experiment stage {stage} has no result URI.")
        result.append((stage, name_prefix))
    return result


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
        completed = analysis_stages(experiment)

        import boto3

        session = boto3.Session(region_name=experiment.get("region") or os.environ.get("AWS_REGION"))
        s3 = session.client("s3")
        output_dir = args.output_root.expanduser() / f"experiment-{experiment_id}"
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
