"""Durable experiment state and lightweight S3 artifact validation (no Modal SDK)."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import uuid

from cs182_project.model_artifacts import MANIFEST, parse_s3_uri, validate_manifest

STAGES = ("seed_model", "partition", "original_activations", "original_analysis",
          "orthogonalized_model", "orthogonalized_activations", "orthogonalized_analysis")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_save(path, value):
    """Keep the last valid checkpoint if a write is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
            temporary = output.name
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_json(s3, uri, *, optional=False):
    bucket, key = parse_s3_uri(uri)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        if optional:
            return None, None
        raise
    with response["Body"] as body:
        data = body.read()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {uri}")
    return value, response.get("ETag")


def put_json(s3, uri, value, **conditions):
    bucket, key = parse_s3_uri(uri)
    return s3.put_object(Bucket=bucket, Key=key,
                         Body=json.dumps(value, sort_keys=True, allow_nan=False).encode(),
                         ContentType="application/json", **conditions)


def document(s3, uri, pins=None):
    """Hash the actual manifest bytes, not a reserialized version."""
    bucket, key = parse_s3_uri(uri)
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as body:
        raw = body.read()
    digest = hashlib.sha256(raw).hexdigest()
    if pins is not None:
        if uri in pins and pins[uri] != digest:
            raise ValueError(f"Pinned input changed at {uri}; start a new experiment.")
        pins[uri] = digest
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a manifest object at {uri}")
    return value, digest


def check_sizes(s3, bucket, prefix, entries, *, relative=False):
    seen = set()
    for entry in entries:
        key = f"{prefix}/{entry['path']}" if relative else entry["key"]
        if (not key.startswith(prefix + "/") or str(PurePosixPath(key)) != key
                or ".." in PurePosixPath(key).parts or key in seen
                or type(entry.get("size")) is not int or entry["size"] < 0):
            raise ValueError("Invalid artifact location or size")
        seen.add(key)
        if s3.head_object(Bucket=bucket, Key=key)["ContentLength"] != entry["size"]:
            raise ValueError(f"Missing or changed artifact: s3://{bucket}/{key}")


def snapshot(s3, uri, *, dataset=False, pins=None):
    from cs182_project.dataset_artifacts import validate_dataset_manifest

    uri = uri.rstrip("/")
    value, digest = document(s3, f"{uri}/{MANIFEST}", pins)
    (validate_dataset_manifest if dataset else validate_manifest)(value)
    bucket, prefix = parse_s3_uri(uri)
    check_sizes(s3, bucket, prefix, value["files"], relative=True)
    return value, digest


def activation(s3, uri, pins=None):
    from cs182_project.activation_runs import _validate_manifest

    value, digest = document(s3, uri, pins)
    bucket, key = parse_s3_uri(uri)
    prefix = key.rsplit("/", 1)[0]
    _validate_manifest(value, prefix)
    if value["run_id"] != prefix.rsplit("/", 1)[-1]:
        raise ValueError("Activation run ID does not match its prefix")
    check_sizes(s3, bucket, prefix, value["shards"])
    return value, digest


def analysis(s3, uri, pins=None):
    from cs182_project.analysis_directions import _artifacts

    manifest_uri = uri.rsplit("/", 1)[0] + "/manifest.json"
    value, digest = document(s3, manifest_uri, pins)
    bucket, key = parse_s3_uri(manifest_uri)
    prefix = key.rsplit("/", 1)[0]
    artifacts = _artifacts(value, prefix)
    if "report.html" not in artifacts:
        raise ValueError("Analysis is missing its report")
    check_sizes(s3, bucket, prefix, artifacts.values())
    return value, digest


def prepare_request(s3, task, options, runtime_image, pins):
    """Pin metadata and resolve implicit partition selection before submission."""
    from cs182_project.dataset_partition import load_partition, partition_uri

    options = dict(options)
    inputs = {}
    if task in {"collect_activations", "orthogonalize_model"}:
        uri = options["model_s3_uri"]
        _, inputs[uri + "/" + MANIFEST] = snapshot(s3, uri, pins=pins)
    if task in {"collect_activations", "partition_dataset"}:
        uri = options["dataset_s3_uri"]
        dataset, digest = snapshot(s3, uri, dataset=True, pins=pins)
        inputs[uri + "/" + MANIFEST] = digest
        if task == "collect_activations":
            partition = options.get("partition_s3_uri") or partition_uri(uri, dataset)
            value, partition_sha = document(s3, partition, pins)
            load_partition(s3, partition, dataset, digest, value.get("row_count", 0))
            inputs[partition] = partition_sha
            options["partition_s3_uri"] = partition
    if task == "analysis":
        uri = options["run_s3_uri"]
        _, inputs[uri] = activation(s3, uri, pins)
    if task == "orthogonalize_model":
        uri = options["analysis_s3_uri"].rstrip("/") + "/report.html"
        _, digest = analysis(s3, uri, pins)
        inputs[uri.rsplit("/", 1)[0] + "/manifest.json"] = digest
    request = {"task": task, "options": options, "runtime_image": runtime_image, "inputs": inputs}
    request["fingerprint"] = fingerprint(request)
    return request


def validate_result(s3, request, uri):
    """Require completion, exact provenance, and all referenced object sizes."""
    task, options, inputs = request["task"], request["options"], request["inputs"]
    # Revalidate pinned metadata on every skip, including its upstream artifacts.
    for source, expected in inputs.items():
        _, digest = document(s3, source)
        if digest != expected:
            raise ValueError(f"Upstream manifest changed: {source}")
    if task == "collect_activations":
        value, _ = activation(s3, uri)
        for key in ("sample_size", "seed", "batch_size", "max_input_tokens", "device"):
            if value.get(key) != options[key]:
                raise ValueError(f"Activation {key} differs from this experiment")
        for kind in ("model", "dataset"):
            source = options[f"{kind}_s3_uri"]
            if (value[kind].get("s3_uri") != source
                    or value[kind].get("manifest_sha256") != inputs[source + "/" + MANIFEST]):
                raise ValueError(f"Activation {kind} provenance differs")
        if (value["dataset"].get("split") != options["split"]
                or value["dataset"].get("partition", {}).get("s3_uri") != options["partition_s3_uri"]
                or value["dataset"].get("partition", {}).get("sha256") != inputs[options["partition_s3_uri"]]):
            raise ValueError("Activation partition differs")
    elif task == "analysis":
        value, _ = analysis(s3, uri)
        if (value.get("source_manifest_uri") != options["run_s3_uri"]
                or value.get("source_manifest_sha256") != inputs[options["run_s3_uri"]]
                or any(value.get(key) != options[key] for key in ("folds", "seed", "include_adversarial"))):
            raise ValueError("Analysis source or settings differ")
    elif task in {"orthogonalize_model", "seed_model"}:
        value, _ = snapshot(s3, uri)
        if task == "orthogonalize_model":
            source = options["model_s3_uri"]
            analysis_uri = options["analysis_s3_uri"].rstrip("/") + "/manifest.json"
            if (value.get("artifact_kind") != "orthogonalized_model"
                    or value.get("source_model", {}).get("s3_uri") != source
                    or value["source_model"].get("manifest_sha256") != inputs[source + "/" + MANIFEST]
                    or value.get("direction", {}).get("analysis_manifest_uri") != analysis_uri
                    or value["direction"].get("analysis_manifest_sha256") != inputs[analysis_uri]
                    or value["direction"].get("selector") != options["rhat"]):
                raise ValueError("Orthogonalized model provenance differs")
        else:
            from cs182_project.model_artifacts import MODEL_ID

            if (uri != f"s3://{options['bucket']}/{options['prefix']}"
                    or value.get("revision") != options["revision"] or value.get("model_id") != MODEL_ID
                    or value.get("upstream", {}).get("provider") != "modelscope"):
                raise ValueError("Seeded model destination, identity, or revision differs")
    elif task == "partition_dataset":
        from cs182_project.dataset_partition import load_partition, partition_uri

        source = options["dataset_s3_uri"]
        dataset, digest = snapshot(s3, source, dataset=True)
        if uri != partition_uri(source, dataset):
            raise ValueError("Partition URI differs")
        value, _ = document(s3, uri)
        load_partition(s3, uri, dataset, digest, value.get("row_count", 0))
    else:
        raise ValueError(f"Unknown experiment task: {task}")
    return uri


def result_manifest(request, uri):
    task = request["task"]
    if task == "analysis":
        return uri.rsplit("/", 1)[0] + "/manifest.json"
    if task in {"seed_model", "orthogonalize_model"}:
        return uri.rstrip("/") + "/" + MANIFEST
    return uri


def completion_record(s3, request, uri):
    validate_result(s3, request, uri)
    manifest_uri = result_manifest(request, uri)
    _, digest = document(s3, manifest_uri)
    return {"result_uri": uri, "manifest_uri": manifest_uri, "manifest_sha256": digest}


def completed_result(s3, request, record):
    uri = record["result_uri"]
    manifest_uri = result_manifest(request, uri)
    _, digest = document(s3, manifest_uri)
    if record.get("manifest_uri") != manifest_uri or record.get("manifest_sha256") != digest:
        raise ValueError("Completed output manifest changed")
    return validate_result(s3, request, uri)


def resolve_stage(s3, root, request, run_new, wait_existing, *, adopted=None):
    """Claim once; recover receipts/calls; retry only a confirmed failed attempt."""
    head_uri = root + "/stage.json"
    head, etag = read_json(s3, head_uri, optional=True)
    if head and head.get("fingerprint") != request["fingerprint"]:
        raise ValueError("Stage inputs differ from its remote record")
    if head:
        attempt = root + "/" + head["attempt"]
        complete, _ = read_json(s3, attempt + "/complete.json", optional=True)
        if complete:
            if adopted and adopted != complete["result_uri"]:
                raise ValueError("Adopted URI differs from this stage's completed result")
            return completed_result(s3, request, complete)
        submitted, _ = read_json(s3, attempt + "/submitted.json", optional=True)
        started, _ = read_json(s3, attempt + "/started.json", optional=True)
        failed, _ = read_json(s3, attempt + "/failed.json", optional=True)
        if not failed and not adopted:
            call = submitted or started
            if not call:
                raise RuntimeError("Submission is uncertain; no call ID is recorded yet. Resume later or adopt its completed output. No duplicate was submitted.")
            if call:
                uri = wait_existing(call["call_id"], attempt)
                return validate_result(s3, request, uri)
    if adopted:
        uri = validate_result(s3, request, adopted)
        # Explicit adoption can heal expired call history. It never launches or
        # cancels work, and still requires exact artifact provenance.
        if not head:
            head = {"fingerprint": request["fingerprint"], "attempt": uuid.uuid4().hex}
            put_json(s3, head_uri, head, IfNoneMatch="*")
        put_json(s3, root + "/" + head["attempt"] + "/complete.json", completion_record(s3, request, uri), IfNoneMatch="*")
        return uri
    head = {"fingerprint": request["fingerprint"], "attempt": uuid.uuid4().hex, "request": request}
    conditions = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
    put_json(s3, head_uri, head, **conditions)
    attempt = root + "/" + head["attempt"]
    return validate_result(s3, request, run_new(attempt))
