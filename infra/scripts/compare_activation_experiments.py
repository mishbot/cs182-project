#!/usr/bin/env python3
"""Compare two completed activation experiments in a local TeX report."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from cs182_project.activation_comparison import (
    MODEL_NAMES,
    comparison_figures,
    comparison_tex,
    load_panel,
    validate_model_role,
)
from cs182_project.activation_report import analysis_stages, download_analysis_plot_artifacts
from cs182_project.model_artifacts import parse_s3_uri
from cs182_project.report_paths import label_report_path


def read_experiment(path: Path, *, defended: bool) -> dict:
    experiment = json.loads(path.expanduser().read_text(encoding="utf-8"))
    label = MODEL_NAMES[int(defended)]
    if not isinstance(experiment, dict):
        raise ValueError(f"{label} checkpoint must be a JSON object.")
    identifier = experiment.get("experiment_id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError(f"{label} checkpoint has no valid experiment_id.")
    for field in ("original_model", "orthogonalized_model"):
        uri = experiment.get(field)
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"{label} checkpoint has no {field} snapshot URI.")
        parse_s3_uri(uri)
    validate_model_role(experiment["original_model"], defended=defended)
    try:
        for stage, _ in analysis_stages(experiment):
            parse_s3_uri(experiment[stage])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} checkpoint: {error}") from error
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_experiment_json", type=Path)
    parser.add_argument("defended_experiment_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        original = read_experiment(args.original_experiment_json, defended=False)
        defended = read_experiment(args.defended_experiment_json, defended=True)
        if args.output_dir:
            output = label_report_path(args.output_dir.expanduser(), "original-defended")
        else:
            output = Path("report_v1/images") / (
                f"compare-original-{original['experiment_id']}-defended-{defended['experiment_id']}"
            )

        import boto3

        clients = {}
        for experiment in (original, defended):
            region = experiment.get("region") or os.environ.get("AWS_REGION")
            if region not in clients:
                clients[region] = boto3.Session(region_name=region).client("s3")
        artifacts = []
        for stage, snapshot in (
            ("original_analysis", "original_model"),
            ("orthogonalized_analysis", "orthogonalized_model"),
        ):
            for experiment in (original, defended):
                region = experiment.get("region") or os.environ.get("AWS_REGION")
                metrics, directions = download_analysis_plot_artifacts(
                    clients[region], experiment[stage]
                )
                artifacts.append((metrics, directions, experiment[snapshot]))

        groups = {}
        for group in ("non_adversarial", "adversarial"):
            present = [group in metrics["groups"] for metrics, _, _ in artifacts]
            if not all(present):
                if group == "non_adversarial":
                    raise ValueError("All four analyses must include the non-adversarial group.")
                if any(present):
                    print("Skipping adversarial figures because some analyses omit that group.",
                          file=sys.stderr)
                continue
            groups[group] = [
                load_panel(metrics, directions, group, model)
                for metrics, directions, model in artifacts
            ]
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".comparison-", dir=output) as temporary:
            from kaleido import write_fig_from_object_sync

            staged = Path(temporary)
            figures = [
                {
                    "fig": figure,
                    "path": staged / f"{group}_{name}.png",
                    "opts": {"format": "png", "scale": 2},
                }
                for group, panels in groups.items()
                for name, figure in comparison_figures(panels).items()
            ]
            # Axis labels are plain text, so rendering needs no MathJax CDN request.
            write_fig_from_object_sync(figures, kopts={"mathjax": False})
            report = staged / "comparison.tex"
            report.write_text(comparison_tex(groups), encoding="utf-8")
            for figure_path in staged.glob("*.png"):
                figure_path.replace(output / figure_path.name)
            report.replace(output / report.name)
        print(output / "comparison.tex")
    except Exception as error:
        print(f"Experiment comparison failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
