"""Select a checksummed direction from a completed analysis of the exact source model."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from ._types import JsonObject
from .model_artifacts import parse_s3_uri

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np
    from mypy_boto3_s3 import S3Client
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
DEFAULT_RHAT = "diff_of_means_max_f1"
DIRECTION_ALIASES = {
    DEFAULT_RHAT: "non_adversarial_midpoint",
    "logistic_max_f1": "non_adversarial_logistic",
    "refusal_regression_max_r2": "non_adversarial_refusal",
}


@dataclass(frozen=True)
class AnalysisDirection:
    vector: NDArray[np.float64]
    provenance: JsonObject


def read_object(s3: S3Client, bucket: str, key: str) -> bytes:
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as body:
        return body.read()


def json_object(data: bytes) -> JsonObject:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")
    return value


def _artifacts(manifest: JsonObject, prefix: str) -> dict[str, JsonObject]:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Expected a completed analysis manifest.")
    if manifest.get("analysis_id") != prefix.rsplit("/", 1)[-1]:
        raise ValueError("Analysis ID does not match its S3 prefix.")
    result: dict[str, JsonObject] = {}
    for entry in manifest.get("artifacts", []):
        key = entry.get("key", "")
        name = key.removeprefix(f"{prefix}/")
        if (
            not key.startswith(f"{prefix}/")
            or name not in {"metrics.json", "directions.npz", "report.html"}
            or str(PurePosixPath(key)) != key
            or name in result
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
        ):
            raise ValueError("Invalid analysis artifact paths, sizes, or checksums.")
        result[name] = entry
    if not {"metrics.json", "directions.npz"} <= result.keys():
        raise ValueError("Analysis must contain metrics.json and directions.npz.")
    return result


def _verified_artifact(s3: S3Client, bucket: str, entry: JsonObject) -> bytes:
    data = read_object(s3, bucket, entry["key"])
    if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checksum/size mismatch for analysis artifact {entry['key']}.")
    return data


def _direction_from_analysis(
    s3: S3Client, uri: str, source_sha256: str, hidden_size: int, num_layers: int, selector: str
) -> AnalysisDirection | None:
    import numpy as np

    bucket, key = parse_s3_uri(uri)
    if not key.endswith("/manifest.json"):
        key += "/manifest.json"
    prefix = key.rsplit("/", 1)[0]
    manifest_bytes = read_object(s3, bucket, key)
    manifest = json_object(manifest_bytes)
    if manifest.get("status") != "complete":
        return None
    artifacts = _artifacts(manifest, prefix)
    metrics = json_object(_verified_artifact(s3, bucket, artifacts["metrics.json"]))
    if metrics.get("manifest_sha256") != manifest.get("source_manifest_sha256"):
        raise ValueError(
            "Analysis metrics and completion manifest disagree on the activation source."
        )
    source = metrics.get("source_manifest", {}).get("model", {})
    if source.get("manifest_sha256") != source_sha256:
        return None
    name = DIRECTION_ALIASES.get(selector, selector)
    data = _verified_artifact(s3, bucket, artifacts["directions.npz"])
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        names = arrays["selected_names"]
        layers, valid, directions = (
            arrays[key] for key in ("selected_layers", "selected_valid", "selected_directions")
        )
        if (
            names.ndim != 1
            or names.dtype.kind != "U"
            or len(set(names.tolist())) != names.size
            or directions.shape != (names.size, hidden_size)
            or directions.dtype.kind != "f"
            or valid.shape != names.shape
            or valid.dtype != np.bool_
            or layers.shape != names.shape
            or layers.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Malformed selected direction arrays or incompatible hidden dimension."
            )
        indices = np.flatnonzero(names == name)
        if indices.size != 1:
            raise ValueError(
                f"Direction {selector!r} ({name}) is absent from {uri}. Available: {names.tolist()}"
            )
        index = int(indices[0])
        vector = np.asarray(directions[index], dtype=np.float64).copy()
        layer = int(layers[index])
        norm = float(np.linalg.norm(vector))
        if not valid[index] or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Direction {name} is unavailable, zero, or nonfinite.")
        if not 0 <= layer < num_layers:
            raise ValueError("Selected direction layer is incompatible with the source model.")
        descriptor = next(
            (item for item in metrics.get("selected_directions", []) if item.get("name") == name),
            None,
        )
        if (
            descriptor is None
            or descriptor.get("layer") != layer
            or descriptor.get("valid") is not True
        ):
            raise ValueError("Selected direction metadata differs between metrics and NPZ.")
    vector /= norm
    return AnalysisDirection(
        vector,
        {
            "analysis_manifest_uri": f"s3://{bucket}/{key}",
            "analysis_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "analysis_id": manifest["analysis_id"],
            "activation_manifest_sha256": metrics["manifest_sha256"],
            "metrics_sha256": artifacts["metrics.json"]["sha256"],
            "directions_sha256": artifacts["directions.npz"]["sha256"],
            "selector": selector,
            "name": name,
            "layer": layer,
            "original_norm": norm,
            "normalized_sha256": hashlib.sha256(vector.astype("<f8").tobytes()).hexdigest(),
        },
    )


def load_analysis_direction(
    s3: S3Client,
    source_sha256: str,
    hidden_size: int,
    num_layers: int,
    *,
    analysis_s3_uri: str | None = None,
    analysis_root_s3_uri: str | None = None,
    rhat: str = DEFAULT_RHAT,
) -> AnalysisDirection:
    """Find latest by completion time, skipping other models but never hiding corrupt artifacts."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,95}", rhat):
        raise ValueError(
            "--rhat must be a direction alias or exported name using letters, digits, '_' or '-'."
        )
    if analysis_s3_uri:
        result = _direction_from_analysis(
            s3, analysis_s3_uri, source_sha256, hidden_size, num_layers, rhat
        )
        if result is None:
            raise ValueError(
                "The selected analysis is incomplete or its source model does not match."
            )
        return result
    if not analysis_root_s3_uri:
        raise ValueError("Provide an analysis root when selecting the latest matching analysis.")
    bucket, prefix = parse_s3_uri(analysis_root_s3_uri)
    candidates: list[tuple[datetime, str]] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            parts = key.removeprefix(f"{prefix}/").split("/")
            modified = item.get("LastModified")
            if len(parts) == 3 and parts[-1] == "manifest.json" and modified is not None:
                candidates.append((modified, key))
    for _, key in sorted(candidates, reverse=True):
        uri = f"s3://{bucket}/{key}"
        result = _direction_from_analysis(s3, uri, source_sha256, hidden_size, num_layers, rhat)
        if result is not None:
            logger.info(
                "Selected matching analysis %s and direction %s", uri, result.provenance["name"]
            )
            return result
    raise ValueError(f"No completed analyses matching this model under {analysis_root_s3_uri}.")
