"""Analyze an S3 activation run on one Modal GPU and publish results to S3."""

import os

import modal

from cs182_project.model_artifacts import parse_s3_uri
from infra.modal._runner import aws_environment, execute_analysis, runtime_image

app = modal.App("cs182-modal-analysis-cli")


@app.local_entrypoint()
def main(run_s3_uri: str = "", activation_root_s3_uri: str = "", output_s3_uri: str = "",
         project: str = "", region: str = "", folds: int = 5, seed: int = 0,
         include_adversarial: bool = False, gpu: str = "A100-40GB",
         memory: int = 32768, timeout: int = 3600):
    if run_s3_uri and activation_root_s3_uri:
        raise ValueError("Select either --run-s3-uri or --activation-root-s3-uri")
    if folds < 2 or seed < 0 or memory < 1024 or not 1 <= timeout <= 86400 or not gpu.strip():
        raise ValueError("Require folds >= 2, seed >= 0, memory >= 1024 MiB, a GPU, and timeout 1–86400")
    for uri in (run_s3_uri, activation_root_s3_uri, output_s3_uri):
        if uri:
            parse_s3_uri(uri)
    reference = os.environ.get("CS182_MODAL_IMAGE", "")
    image = runtime_image(reference)
    import boto3

    credentials = aws_environment(boto3.Session(region_name=region or None))
    options = dict(run_s3_uri=run_s3_uri, activation_root_s3_uri=activation_root_s3_uri,
                   output_s3_uri=output_s3_uri or os.environ.get("ANALYSIS_URI", ""),
                   project=project or os.environ.get("CS182_PROJECT", "cs182-cddr"),
                   region=region or credentials["AWS_REGION"], folds=folds, seed=seed,
                   include_adversarial=include_adversarial)
    if not run_s3_uri and not activation_root_s3_uri:
        options["activation_root_s3_uri"] = os.environ.get("OUTPUT_URI", "")
    worker_app = modal.App("cs182-project-analysis")
    worker = worker_app.function(image=image, secrets=[modal.Secret.from_dict(credentials)],
                                gpu=gpu, cpu=4, memory=memory, timeout=timeout, retries=0,
                                serialized=True, include_source=False)(execute_analysis)
    with modal.enable_output(), worker_app.run(detach=os.environ.get("CS182_MODAL_DETACH") == "1"):
        worker.remote(options)
