"""Remove an analysis direction from Qwen residual writers and publish a new S3 model."""

import os

import modal

from cs182_project.activation_runs import activation_root
from cs182_project.analysis_directions import DEFAULT_RHAT
from cs182_project.model_artifacts import parse_s3_uri
from infra.modal._runner import run_task

app = modal.App("cs182-modal-orthogonalization-cli")


@app.local_entrypoint()
def main(model_s3_uri: str, analysis_s3_uri: str = "", analysis_root_s3_uri: str = "",
         rhat: str = DEFAULT_RHAT, device: str = "auto", gpu: str = "T4", timeout: int = 3600,
         project: str = "", region: str = ""):
    parse_s3_uri(model_s3_uri)
    if analysis_s3_uri and analysis_root_s3_uri:
        raise ValueError("Select an analysis run or an analysis root, not both.")
    for uri in (analysis_s3_uri, analysis_root_s3_uri):
        if uri:
            parse_s3_uri(uri)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    root = analysis_root_s3_uri or os.environ.get("ANALYSIS_URI", "")
    if not analysis_s3_uri and not root:
        import boto3

        standard = activation_root(boto3.Session(region_name=region or None),
                                   project=project or os.environ.get("CS182_PROJECT", "cs182-cddr"),
                                   region=region or None)
        bucket, _ = parse_s3_uri(standard)
        root = f"s3://{bucket.removesuffix('-activations')}-analyses/runs"
    print(run_task("orthogonalize_model", dict(model_s3_uri=model_s3_uri,
                                              analysis_s3_uri=analysis_s3_uri or None,
                                              analysis_root_s3_uri=root or None,
                                              rhat=rhat, device=device), gpu=gpu, timeout=timeout))
