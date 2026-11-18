"""Collect archived-dataset activations on Modal and publish results to S3."""

import modal

from cs182_project.model_artifacts import parse_s3_uri
from infra.modal._runner import run_task

app = modal.App("cs182-modal-activations-cli")


@app.local_entrypoint()
def main(
    model_s3_uri: str, dataset_s3_uri: str, output_s3_uri: str, sample_size: int,
    split: str = "train", seed: int = 0, batch_size: int = 4,
    max_input_tokens: int = 2048, device: str = "auto",
    gpu: str = "T4", timeout: int = 3600, partition_s3_uri: str = "",
):
    for uri in [model_s3_uri, dataset_s3_uri, output_s3_uri]:
        parse_s3_uri(uri)
    if sample_size < 4 or batch_size < 1 or max_input_tokens < 1:
        raise ValueError("--sample-size must be >= 4; --batch-size and --max-input-tokens must be positive")
    if split not in {"train", "validation", "test"}:
        raise ValueError("--split must be train, validation, or test")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    print(run_task("collect_activations", dict(
        model_s3_uri=model_s3_uri, dataset_s3_uri=dataset_s3_uri,
        output_s3_uri=output_s3_uri, sample_size=sample_size, split=split,
        partition_s3_uri=partition_s3_uri or None, seed=seed, batch_size=batch_size, max_input_tokens=max_input_tokens,
        device=device,
    ), gpu=gpu, timeout=timeout))
