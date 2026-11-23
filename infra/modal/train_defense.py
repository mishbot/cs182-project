"""Train with the prepared ECR runtime; outputs and upload queue persist on a Volume."""

import json
import os
import re

import modal

from cs182_project.model_artifacts import parse_s3_uri
from cs182_project.training import DRY_RUN, training_config
from infra.modal._runner import run_task

app = modal.App("cs182-modal-training-cli")


@app.local_entrypoint()
def main(run_name: str, model_s3_uri: str = "", dataset_s3_uri: str = "",
         stats_s3_uri: str = "", model_output_s3_uri: str = "", partition_s3_uri: str = "",
         config_json: str = "{}", dry_run: bool = False, device: str = "auto",
         gpu: str = "A100-80GB", timeout: int = 21600):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name):
        raise ValueError("Invalid run name")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if partition_s3_uri:
        parse_s3_uri(partition_s3_uri)
    model_s3_uri = model_s3_uri or os.environ.get("MODEL_URI", "")
    dataset_s3_uri = dataset_s3_uri or os.environ.get("DATASET_URI", "")
    bucket, _ = parse_s3_uri(model_s3_uri)
    parse_s3_uri(dataset_s3_uri)
    stats_s3_uri = stats_s3_uri or os.environ.get("TRAINING_STATS_URI", "")
    if not stats_s3_uri:
        if not bucket.endswith("-models"):
            raise ValueError("--stats-s3-uri is required for a nonstandard models bucket")
        stats_s3_uri = f"s3://{bucket.removesuffix('-models')}-training-stats/runs"
    model_output_s3_uri = model_output_s3_uri or f"s3://{bucket}/models/trained"
    parse_s3_uri(stats_s3_uri)
    parse_s3_uri(model_output_s3_uri)
    overrides = {**(DRY_RUN if dry_run else {}), **json.loads(config_json)}
    training_config(overrides)
    print(run_task("train_defense", dict(
        model_s3_uri=model_s3_uri, dataset_s3_uri=dataset_s3_uri, stats_s3_uri=stats_s3_uri,
        model_output_s3_uri=model_output_s3_uri, partition_s3_uri=partition_s3_uri or None,
        run_name=run_name, device=device, overrides=overrides,
    ), gpu=gpu, timeout=timeout))
