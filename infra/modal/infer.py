"""Run archived-model inference on Modal using the existing ECR runtime."""

import modal

from cs182_project.model_artifacts import parse_s3_uri
from infra.modal._runner import run_task

app = modal.App("cs182-modal-infer-cli")


@app.local_entrypoint()
def main(
    model_s3_uri: str, prompt: str, max_new_tokens: int = 128,
    device: str = "auto", gpu: str = "T4", timeout: int = 3600,
):
    parse_s3_uri(model_s3_uri)
    if not prompt.strip():
        raise ValueError("--prompt must not be empty")
    if not 1 <= max_new_tokens <= 4096:
        raise ValueError("--max-new-tokens must be between 1 and 4096")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    print(run_task("infer", dict(
        model_s3_uri=model_s3_uri, prompt=prompt,
        max_new_tokens=max_new_tokens, device=device,
    ), gpu=gpu, timeout=timeout))
