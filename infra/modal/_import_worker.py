"""Import worker with a stable package name for Modal serialization."""

from pathlib import Path


def execute_import(source_path, bucket, name, source):
    import logging
    import boto3
    from botocore.config import Config
    from cs182_project.model_import import import_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s3 = boto3.client("s3", config=Config(retries={"mode": "standard", "max_attempts": 5}))
    uri = import_model(s3, Path("/source-volume") / source_path.lstrip("/"), bucket,
                       name=name, source=source)
    logging.info("Archived checkpoint copied to %s; legacy Qwen execution requires separate support.", uri)
    return uri

