"""Shared Modal image, credentials, execution, and cache configuration."""

import hashlib
import logging
import os
import re
import tempfile

import modal

CACHE_VOLUME = "cs182-project-cache"
CACHE_MOUNT = "/snapshot-cache"
logger = logging.getLogger(__name__)
# Bump when the import/adaptation recipe changes, so old prepared images cannot
# silently stand in for the new recipe. ECR references must be immutable.
IMAGE_ADAPTER_VERSION = "v1"


def aws_environment(session):
    """Freeze the current local credential chain for this invocation only."""
    credentials = session.get_credentials()
    if credentials is None:
        raise ValueError("No AWS credentials found. Source infra/scripts/aws-login.sh first.")
    frozen = credentials.get_frozen_credentials()
    region = session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
    values = {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
    }
    if frozen.token:
        values["AWS_SESSION_TOKEN"] = frozen.token
    return values


def _registry_match(reference):
    match = re.fullmatch(
        r"\d{12}\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/"
        r"[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[0-9a-f]{64})",
        reference,
    )
    if match is None:
        raise ValueError("Set CS182_MODAL_IMAGE to the full private ECR image URI with an immutable tag or digest.")
    return match


def modal_image_name(reference: str) -> str:
    """Map one immutable ECR reference to a stable, account-specific Modal name."""
    _registry_match(reference)
    digest = hashlib.sha256(reference.encode()).hexdigest()[:32]
    return f"cs182-runtime-{IMAGE_ADAPTER_VERSION}:{digest}"


def import_runtime_image(reference, credentials):
    """Import/adapt ECR only during explicit preparation, without local sources."""
    match = _registry_match(reference)
    # The registry can be in a different region from the S3 runtime configuration.
    registry_keys = {key: value for key, value in credentials.items() if key in {
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    }}
    registry_keys["AWS_REGION"] = match[1]
    return (
        modal.Image.from_registry(reference, secret=modal.Secret.from_dict(registry_keys))
        .entrypoint([])
        .run_commands("/app/.venv/bin/python -m ensurepip --upgrade")
        .env({"VIRTUAL_ENV": "/app/.venv"})
    )


def prepare_runtime_image(reference, credentials):
    """Publish only after the ECR import and all adaptation layers succeed."""
    name = modal_image_name(reference)
    logger.info("Preparing %s from %s", name, reference)
    with modal.enable_output():
        image = import_runtime_image(reference, credentials).build()
        image.publish(name)
    return name


def runtime_image(reference):
    """Use an already prepared image; execution never imports from a registry."""
    return modal.Image.from_name(modal_image_name(reference)).add_local_python_source("infra.modal")


def execute_analysis(options):
    """Use the image's isolated GPU environment; inherit stdout/stderr for Modal streaming."""
    # Keep the worker in the uploaded package: a CLI file loaded by basename
    # would collide with the image's unrelated /app/analyze_activations.py.
    import subprocess

    with tempfile.TemporaryDirectory(prefix="cs182-gpu-analysis-") as directory:
        command = ["/app/.analysis-venv/bin/python", "/app/analyze_activations.py",
                   "--backend", "cuml", "--output-dir", directory]
        for key, value in options.items():
            if value is None or value == "":
                continue
            if isinstance(value, bool):
                if value:
                    command.append("--" + key.replace("_", "-"))
            else:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        # Avoid leaking the parent's venv selection into the subprocess's libraries.
        environment = {**os.environ, "VIRTUAL_ENV": "/app/.analysis-venv"}
        subprocess.run(command, check=True, env=environment)


def execute_task(task, options):
    """Remote worker: use the image's library and persist only input snapshots."""
    import boto3
    from infra.modal._cache import SnapshotCache

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    snapshots = [("models", options["model_s3_uri"])]
    if task == "collect_activations":
        snapshots.append(("datasets", options["dataset_s3_uri"]))
    elif task not in {"infer", "orthogonalize_model"}:
        raise ValueError(f"Unknown task: {task}")
    volume = modal.Volume.from_name(CACHE_VOLUME)
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory(prefix="cs182-snapshots-") as working:
        cache = SnapshotCache(CACHE_MOUNT, working, snapshots)
        try:
            cache.restore()
        except OSError:
            logger.warning("Could not restore the snapshot cache; fetching from S3", exc_info=True)
        try:
            if task == "infer":
                from cs182_project.inference import generate
                from cs182_project.model_artifacts import download_model

                directory = download_model(s3, options["model_s3_uri"], f"{working}/models")
                return generate(directory, options["prompt"], options["max_new_tokens"], options["device"])
            if task == "orthogonalize_model":
                from cs182_project.weight_orthogonalization import orthogonalize_model

                return orthogonalize_model(s3, cache_dir=working, **options)
            from cs182_project.activations import collect

            return collect(s3, cache_dir=working, **options)
        finally:
            try:
                cache.publish()
                volume.commit()
            except Exception:
                # Caching is optional; don't hide a task failure or invalidate an
                # activation run whose completion manifest already reached S3.
                logger.warning("Could not persist the snapshot cache", exc_info=True)


def run_task(task, options, *, gpu="T4", timeout=3600):
    """Create an ephemeral GPU App after the local CLI has validated its inputs."""
    import boto3

    if not 1 <= timeout <= 86400:
        raise ValueError("--timeout must be between 1 and 86400 seconds")
    reference = os.environ.get("CS182_MODAL_IMAGE", "")
    if not reference:
        raise ValueError("Set CS182_MODAL_IMAGE to the ECR runtime image URI.")
    credentials = aws_environment(boto3.Session())
    image = runtime_image(reference)
    volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)
    worker_app = modal.App(f"cs182-project-{task.replace('_', '-')}")
    worker = worker_app.function(
        image=image,
        secrets=[modal.Secret.from_dict(credentials)],
        volumes={CACHE_MOUNT: volume},
        gpu=None if options["device"] == "cpu" else gpu,
        cpu=2,
        memory=16384,
        timeout=timeout,
        retries=0,
        serialized=True,
        include_source=False,
    )(execute_task)
    # The outer `modal run` entrypoint supplies output for this nested worker App.
    try:
        with modal.enable_output(), worker_app.run():
            return worker.remote(task, options)
    except modal.exception.NotFoundError as error:
        raise RuntimeError(
            f"Could not start with prepared image {modal_image_name(reference)}: {error}. "
            "Check the Modal environment and run "
            "`uv run --extra modal python -m infra.modal.prepare_image` before running GPU tasks."
        ) from error
