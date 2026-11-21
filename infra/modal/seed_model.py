"""Seed original Qwen from ModelScope directly to S3 on a CPU Modal worker."""

import os
import modal

from cs182_project.model_artifacts import MODEL_NAME, MODEL_REVISION
from cs182_project.model_import import model_bucket
from cs182_project.seeding import validate_model_destination
from infra.modal._runner import aws_environment
from infra.modal._seed_worker import execute_seed

app = modal.App('cs182-modal-model-seed')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('git', 'git-lfs')
         .uv_pip_install('boto3==1.38.23', 'botocore==1.38.46', 's3transfer==0.13.1', uv_version='0.10.4')
         .add_local_python_source('cs182_project', 'infra.modal'))


@app.local_entrypoint()
def main(bucket: str = '', revision: str = MODEL_REVISION, prefix: str = '',
         project: str = '', region: str = '', timeout: int = 3600):
    validate_model_destination(revision, prefix or f'models/{MODEL_NAME}/{revision}')
    if not 1 <= timeout <= 86400:
        raise ValueError('--timeout must be between 1 and 86400 seconds')
    import boto3

    session = boto3.Session(region_name=region or None)
    destination = bucket or model_bucket(session, project=project or os.environ.get('CS182_PROJECT', 'cs182-cddr'),
                                        region=region or None)
    worker_app = modal.App('cs182-original-qwen-seed')
    worker = worker_app.function(image=image, secrets=[modal.Secret.from_dict(aws_environment(session))],
                                cpu=2, memory=2048, ephemeral_disk=20000, timeout=timeout,
                                retries=0, serialized=True, include_source=False)(execute_seed)
    with modal.enable_output(), worker_app.run():
        print(worker.remote(destination, revision, prefix))
