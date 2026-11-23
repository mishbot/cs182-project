"""Submit or reconnect to one durable stage of a local Qwen experiment."""

import json
from pathlib import Path

import modal

from infra.modal._experiment import (STAGES, atomic_save, completed_result, completion_record,
                                    document, prepare_request, put_json, read_json, resolve_stage,
                                    result_manifest, validate_result)
from infra.modal._experiment_worker import execute_stage
from infra.modal._runner import (CACHE_MOUNT, CACHE_VOLUME, aws_environment,
                                prepare_runtime_image, runtime_image)

app = modal.App("cs182-experiment-stage-cli")


def wait_for_call(s3, request, call_id, attempt):
    """Check durable results first; never interpret lost call history as failure."""
    call = modal.FunctionCall.from_id(call_id)
    while True:
        complete, _ = read_json(s3, attempt + "/complete.json", optional=True)
        if complete:
            return completed_result(s3, request, complete)
        failed, _ = read_json(s3, attempt + "/failed.json", optional=True)
        if failed:
            raise RuntimeError(f"Stage failed remotely: {failed['error']}. Resume to retry this failed attempt.")
        try:
            result = call.get(timeout=30)
        except modal.exception.FunctionTimeoutError:
            # A function timeout is a terminal remote failure, not a polling timeout.
            put_json(s3, attempt + "/failed.json", {"error": "Remote function timed out"})
            raise RuntimeError("The remote function timed out. Resume to retry this failed attempt.")
        except TimeoutError:
            print(f"Waiting for existing Modal call {call_id} ...", flush=True)
            continue
        except Exception:
            # A result may have been committed while the connection failed.
            complete, _ = read_json(s3, attempt + "/complete.json", optional=True)
            if complete:
                return completed_result(s3, request, complete)
            raise
        validate_result(s3, request, result)
        put_json(s3, attempt + "/complete.json", completion_record(s3, request, result))
        return result


def create_worker(request, credentials, *, prepare_image=True):
    task = request["task"]
    if task == "seed_model":
        from infra.modal.seed_model import image
    else:
        if prepare_image:
            prepare_runtime_image(request["runtime_image"], credentials)
        image = runtime_image(request["runtime_image"])
    worker_app = modal.App("cs182-experiment-" + task.replace("_", "-"))
    config = dict(image=image, secrets=[modal.Secret.from_dict(credentials)],
                  cpu=(request["options"].get("workers", 24) * request["options"].get("threads_per_worker", 1)) if task == "analysis" else 2,
                  memory=65536 if task == "analysis" else (2048 if task == "seed_model" else 16384),
                  gpu=None if task in {"analysis", "seed_model", "partition_dataset"} else "T4",
                  timeout=3600, retries=0, serialized=True, include_source=False)
    if task in {"collect_activations", "orthogonalize_model"}:
        config["volumes"] = {CACHE_MOUNT: modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)}
    return worker_app, worker_app.function(**config)(execute_stage)


def launch(s3, request, attempt, credentials, on_submitted, *, prepare_image=True, on_prepared=lambda: None):
    spawning = False
    try:
        worker_app, worker = create_worker(request, credentials, prepare_image=prepare_image)
        if request["task"] != "seed_model":
            on_prepared()
        with modal.enable_output(), worker_app.run(detach=True):
            # A spawn error can mean submission reached the server. Leave the
            # claim intact; a running worker records its own call ID too.
            spawning = True
            call = worker.spawn(request, attempt)
            on_submitted(call.object_id, attempt)
            put_json(s3, attempt + "/submitted.json", {"call_id": call.object_id}, IfNoneMatch="*")
            print(f"Submitted Modal call {call.object_id}", flush=True)
            return wait_for_call(s3, request, call.object_id, attempt)
    except Exception as error:
        if not spawning:
            put_json(s3, attempt + "/failed.json", {"error": str(error)}, IfNoneMatch="*")
        raise


@app.local_entrypoint()
def main(checkpoint: str, stage: str):
    if stage not in STAGES:
        raise ValueError("Unknown experiment stage")
    path = Path(checkpoint)
    state = json.loads(path.read_text())
    spec = state["stages"][stage]
    import boto3

    session = boto3.Session(region_name=state["region"])
    s3 = session.client("s3")
    credentials = aws_environment(session)
    request = prepare_request(s3, spec["task"], spec["options"],
                              "" if stage == "seed_model" else state["runtime_image"],
                              state.setdefault("pinned_manifests", {}))
    if spec.get("request") and spec["request"] != request:
        raise ValueError("Stage configuration differs from its saved request")
    spec["request"] = request
    if request["task"] == "collect_activations":
        state["partition"] = request["options"]["partition_s3_uri"]
    atomic_save(path, state)
    root = state["environment"]["ANALYSIS_URI"].rstrip("/") + "/experiments/" + state["experiment_id"] + "/" + stage
    spec["record_uri"] = root + "/stage.json"
    atomic_save(path, state)
    # Repair a submitted record if the workstation saved the ID just before
    # losing its S3 connection. Never apply an old ID to a newer attempt.
    head, _ = read_json(s3, root + "/stage.json", optional=True)
    if head and head.get("fingerprint") == request["fingerprint"]:
        attempt = root + "/" + head["attempt"]
        submitted, _ = read_json(s3, attempt + "/submitted.json", optional=True)
        if not submitted and spec.get("attempt_uri") == attempt and spec.get("call_id"):
            put_json(s3, attempt + "/submitted.json", {"call_id": spec["call_id"]}, IfNoneMatch="*")

    def on_submitted(call_id, attempt):
        progress = json.loads(path.read_text())
        progress["stages"][stage].update(status="running", call_id=call_id, attempt_uri=attempt)
        atomic_save(path, progress)

    def on_prepared():
        progress = json.loads(path.read_text())
        progress["prepared_for"] = progress["invocation_id"]
        atomic_save(path, progress)
    adopted = spec.get("adopted_uri") or state.get(stage)
    if adopted:
        document(s3, result_manifest(request, adopted), state["pinned_manifests"])
    # Setup artifacts have deterministic destinations and may predate this run.
    if not adopted and stage in {"seed_model", "partition"}:
        if stage == "seed_model":
            options = request["options"]
            candidate = f"s3://{options['bucket']}/{options['prefix']}"
            manifest = candidate + "/artifact-manifest.json"
        else:
            from cs182_project.dataset_partition import partition_uri

            source = request["options"]["dataset_s3_uri"]
            dataset, _ = document(s3, source + "/artifact-manifest.json")
            candidate = manifest = partition_uri(source, dataset)
        exists, _ = read_json(s3, manifest, optional=True)
        if exists is not None:
            adopted = candidate
    result = resolve_stage(s3, root, request,
                           lambda attempt: launch(s3, request, attempt, credentials, on_submitted,
                                                  prepare_image=state.get("prepared_for") != state["invocation_id"],
                                                  on_prepared=on_prepared),
                           lambda call_id, attempt: wait_for_call(s3, request, call_id, attempt),
                           adopted=adopted)
    validate_result(s3, request, result)
    state = json.loads(path.read_text())
    document(s3, result_manifest(request, result), state["pinned_manifests"])
    state[stage] = result
    state["stages"][stage]["status"] = "complete"
    atomic_save(path, state)
    print(result, flush=True)
