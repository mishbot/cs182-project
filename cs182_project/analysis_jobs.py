"""Analysis orchestration, CPU execution, and completed-result publication."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .activation_runs import activation_root, latest_activation_run, load_activation_run
from .model_artifacts import parse_s3_uri, sha256

if TYPE_CHECKING:
    from boto3.session import Session
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)
ARTIFACT_TYPES = {
    "report.html": "text/html; charset=utf-8",
    "metrics.json": "application/json",
    "directions.npz": "application/octet-stream",
}


def execution_metadata(workers: int, threads_per_worker: int) -> JsonObject:
    """Describe CPU numerical execution independently of report rendering."""
    execution: JsonObject = {
        "backend": "cpu",
        "precision": "float64",
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "pyarrow",
                "plotly",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    logger.info("Analysis execution: %s", json.dumps(execution, sort_keys=True))
    return execution


def publish_analysis(
    s3: S3Client, output_s3_uri: str, directory: PathInput, source_run_id: str, metadata: JsonObject
) -> str:
    """Publish unique artifacts and write the completion marker only after all succeed."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_run_id):
        raise ValueError("Invalid source activation run ID.")
    bucket, root = parse_s3_uri(output_s3_uri)
    identifier = uuid.uuid4().hex
    prefix = f"{root}/{source_run_id}/{identifier}"
    artifacts: list[JsonObject] = []
    started = time.perf_counter()
    try:
        for filename, content_type in ARTIFACT_TYPES.items():
            path = Path(directory) / filename
            entry = {
                "key": f"{prefix}/{filename}",
                "sha256": sha256(path),
                "size": path.stat().st_size,
                "content_type": content_type,
            }
            s3.upload_file(str(path), bucket, entry["key"], ExtraArgs={"ContentType": content_type})
            artifacts.append(entry)
        timings = {**metadata.get("timings_seconds", {}), "upload": time.perf_counter() - started}
        manifest = {
            **metadata,
            "schema_version": 1,
            "status": "complete",
            "analysis_id": identifier,
            "activation_run_id": source_run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "timings_seconds": timings,
            "artifacts": artifacts,
        }
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/manifest.json",
            Body=json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as error:
        raise RuntimeError(
            f"Incomplete analysis at s3://{bucket}/{prefix}; no completion marker was published: {error}"
        ) from error
    logger.info("Analysis phase timings (seconds): %s", json.dumps(timings, sort_keys=True))
    return f"s3://{bucket}/{prefix}/report.html"


def run_analysis(
    session: Session,
    *,
    run_s3_uri: str | None = None,
    activation_root_s3_uri: str | None = None,
    output_s3_uri: str | None = None,
    project: str = "cs182-cddr",
    region: str | None = None,
    folds: int = 5,
    seed: int = 0,
    include_adversarial: bool = False,
    include_ridge: bool = False,
    workers: int = 1,
    threads_per_worker: int = 1,
    output_dir: PathInput | None = None,
) -> str:
    """Read S3 once, analyze enabled groups, render locally, and return the published report URI."""
    if workers < 1 or threads_per_worker < 1:
        raise ValueError("workers and threads_per_worker must be positive integers.")
    if folds < 2 or seed < 0:
        raise ValueError("--folds must be at least 2; --seed must be nonnegative.")
    if run_s3_uri and activation_root_s3_uri:
        raise ValueError("Select either an exact activation run or an activation root.")
    for uri in (run_s3_uri, activation_root_s3_uri, output_s3_uri):
        if uri:
            parse_s3_uri(uri)
    execution = execution_metadata(workers, threads_per_worker)
    from .activation_analysis import analyze_activations
    from .activation_report import write_analysis_report

    region = region or session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
    s3 = session.client("s3", region_name=region)
    root = activation_root_s3_uri or os.environ.get("OUTPUT_URI")
    output_s3_uri = output_s3_uri or os.environ.get("ANALYSIS_URI")
    if not output_s3_uri or (not run_s3_uri and not root):
        standard = activation_root(session, project=project, region=region)
        root = root or standard
        if not output_s3_uri:
            bucket, _ = parse_s3_uri(standard)
            output_s3_uri = f"s3://{bucket.removesuffix('-activations')}-analyses/runs"
    assert output_s3_uri is not None
    parse_s3_uri(output_s3_uri)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    uri = run_s3_uri or latest_activation_run(s3, root or "")
    run = load_activation_run(s3, uri)
    timings["download"] = time.perf_counter() - started
    started = time.perf_counter()
    result = analyze_activations(
        run,
        folds=folds,
        seed=seed,
        include_adversarial=include_adversarial,
        include_ridge=include_ridge,
        workers=workers,
        threads_per_worker=threads_per_worker,
    )
    timings["analysis"] = time.perf_counter() - started
    execution["layer_timings_seconds"] = {
        group.group: {str(layer.layer): layer.elapsed_seconds for layer in group.layers}
        for group in result.groups
    }
    metadata = {
        "source_manifest_uri": run.manifest_uri,
        "source_manifest_sha256": run.manifest_sha256,
        "folds": folds,
        "seed": seed,
        "include_adversarial": include_adversarial,
        "include_ridge": include_ridge,
        "refusal_threshold": 0,
        "refusal_comparison": ">",
        "execution": execution,
        "timings_seconds": timings,
    }
    directory = (
        Path(output_dir)
        if output_dir is not None
        else Path("reports/activations") / run.manifest["run_id"]
    )
    started = time.perf_counter()
    write_analysis_report(result, directory, execution=execution)
    timings["render"] = time.perf_counter() - started
    return publish_analysis(s3, output_s3_uri, directory, run.manifest["run_id"], metadata)
