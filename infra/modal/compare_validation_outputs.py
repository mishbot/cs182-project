"""Compare two experiment snapshots on held-out prompts and write a local TeX report."""

import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import modal

from cs182_project.model_artifacts import parse_s3_uri
from cs182_project.validation_comparison import validation_comparison_tex
from infra.modal._runner import run_task

app = modal.App("cs182-validation-comparison-cli")


def comparison_inputs(experiment: dict) -> dict:
    snapshots = {}
    for field, option in (
        ("original_model", "model_s3_uri"),
        ("orthogonalized_model", "orthogonalized_model_s3_uri"),
        ("dataset", "dataset_s3_uri"),
    ):
        uri = experiment.get(field)
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"Experiment JSON has no {field} snapshot URI.")
        parse_s3_uri(uri)
        snapshots[option] = uri

    partitions = [experiment.get("partition")]
    stages = experiment.get("stages", {})
    for name in ("original_activations", "orthogonalized_activations"):
        stage = stages.get(name, {})
        partitions.append(stage.get("request", {}).get("options", {}).get("partition_s3_uri"))
    partitions = [uri for uri in partitions if uri is not None]
    for uri in partitions:
        if not isinstance(uri, str) or not uri:
            raise ValueError("Experiment JSON has an invalid partition URI.")
        parse_s3_uri(uri)
    if len(set(partitions)) > 1:
        raise ValueError("The experiment records different validation partitions.")
    pins = experiment.get("pinned_manifests", {})
    if not isinstance(pins, dict):
        raise ValueError("Experiment JSON has an invalid pinned_manifests object.")
    return {**snapshots, "partition_s3_uri": next(iter(partitions), None), "pinned_manifests": pins}


@app.local_entrypoint()
def main(
    experiment_json: str,
    samples_per_class: int = 5,
    seed: int = 0,
    gpu: str = "T4",
    output: str = "",
) -> None:
    if samples_per_class < 1 or seed < 0:
        raise ValueError("--samples-per-class must be positive and --seed must be nonnegative.")
    experiment = json.loads(Path(experiment_json).expanduser().read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError("Expected an experiment JSON object.")
    experiment_id = experiment.get("experiment_id")
    if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Experiment JSON has no valid experiment_id.")
    reference = experiment.get("runtime_image")
    if not isinstance(reference, str) or not reference:
        raise ValueError("Experiment JSON has no runtime_image.")
    options = comparison_inputs(experiment)
    destination = (
        Path(output).expanduser()
        if output
        else Path("report_v1") / f"validation-comparison-{experiment_id}.tex"
    )
    results = run_task(
        "compare_validation_outputs",
        {**options, "device": "cuda", "samples_per_class": samples_per_class, "seed": seed},
        gpu=gpu,
        reference=reference,
        region=experiment.get("region"),
    )
    document = validation_comparison_tex(experiment_id, seed, results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=destination.parent, encoding="utf-8", delete=False) as file:
            temporary = Path(file.name)
            file.write(document)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(destination)
