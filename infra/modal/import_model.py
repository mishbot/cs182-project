"""Copy a mounted Modal checkpoint directly to the AWS models bucket without local weights."""

import os
from pathlib import Path, PurePosixPath

import modal

from cs182_project.model_import import model_bucket
from infra.modal._runner import aws_environment
from infra.modal._import_worker import execute_import

app = modal.App("cs182-modal-model-import")
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("boto3==1.38.23", "botocore==1.38.46", "s3transfer==0.13.1",
                         uv_version="0.10.4")
         .add_local_python_source("cs182_project", "infra.modal"))




@app.local_entrypoint()
def main(volume_name: str = "cs182-runs", volume_environment: str = "main",
         source_path: str = "/v1/merged", name: str = "Qwen-1_8B-Chat-merged",
         bucket: str = "", project: str = "", region: str = "", timeout: int = 3600):
    if not 1 <= timeout <= 86400:
        raise ValueError("--timeout must be between 1 and 86400 seconds")
    if not source_path or ".." in PurePosixPath(source_path).parts:
        raise ValueError("Expected a source path within the Volume without '..'.")
    import boto3

    session = boto3.Session(region_name=region or None)
    destination = bucket or model_bucket(session,
                                        project=project or os.environ.get("CS182_PROJECT", "cs182-cddr"),
                                        region=region or None)
    credentials = aws_environment(session)
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    volume = modal.Volume.from_name(volume_name, environment_name=volume_environment,
                                   create_if_missing=False)
    volume.hydrate()
    source = dict(kind="modal_volume", workspace=workspace.name, environment=volume_environment,
                  volume=volume_name, volume_id=volume.object_id,
                  path="/" + source_path.lstrip("/"))
    worker_app = modal.App("cs182-model-import")
    worker = worker_app.function(image=image, secrets=[modal.Secret.from_dict(credentials)],
                                volumes={"/source-volume": volume.with_mount_options(read_only=True)},
                                cpu=2, memory=2048, timeout=timeout, retries=0,
                                serialized=True, include_source=False)(execute_import)
    with modal.enable_output(), worker_app.run():
        print(worker.remote(source_path, destination, name, source))
