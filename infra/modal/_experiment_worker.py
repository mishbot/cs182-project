"""Durable stage worker with a stable import path for Modal serialization."""

import tempfile

import modal

from infra.modal._experiment import completion_record, put_json
from infra.modal._runner import execute_analysis, execute_task


def execute_stage(request, attempt):
    """Record remote progress independently of the workstation's lifetime."""
    import boto3

    s3 = boto3.client("s3")
    put_json(s3, attempt + "/started.json", {"call_id": modal.current_function_call_id()}, IfNoneMatch="*")
    task, options = request["task"], request["options"]
    try:
        if task == "analysis":
            result = execute_analysis(options, capture_result=True)
        elif task == "seed_model":
            from infra.modal._seed_worker import execute_seed

            result = execute_seed(**options)
        elif task == "partition_dataset":
            from cs182_project.dataset_partition import publish_partition

            with tempfile.TemporaryDirectory(prefix="cs182-partition-") as directory:
                result = publish_partition(s3, cache_dir=directory, **options)
        else:
            result = execute_task(task, options)
    except Exception as error:
        put_json(s3, attempt + "/failed.json", {"error": str(error)}, IfNoneMatch="*")
        raise
    # If recording fails after publication, the Modal return value is still a
    # recovery path. A hard crash in this window requires explicit adoption.
    try:
        put_json(s3, attempt + "/complete.json", completion_record(s3, request, result), IfNoneMatch="*")
    except Exception as error:
        print(f"Result published at {result}; could not save recovery receipt: {error}", flush=True)
    return result
