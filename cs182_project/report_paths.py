"""Model provenance and naming for local report outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from ._types import JsonObject
from .model_artifacts import MODEL_NAME, parse_s3_uri

ModelRole = Literal["original", "defended"]


def model_role(uri: str, *, allow_legacy: bool = False) -> ModelRole:
    """Classify the model name after models/, independent of its revision."""
    _, key = parse_s3_uri(uri)
    root, _, remainder = key.partition("/")
    name = remainder.partition("/")[0]
    if root == "models":
        if name == MODEL_NAME or (allow_legacy and name == "Qwen1.5-1.8B-Chat"):
            return "original"
        prefix = MODEL_NAME + "-"
        if name.startswith(prefix) and len(name) > len(prefix):
            return "defended"
    raise ValueError(
        f"Cannot classify {uri}. Expected models/{MODEL_NAME} or models/{MODEL_NAME}-<suffix>."
    )


def analysis_model_role(model: JsonObject) -> ModelRole:
    """Classify an analysis by its model before weight orthogonalization."""
    manifest = model.get("manifest", {})
    uri = model["s3_uri"]
    if manifest.get("artifact_kind") == "orthogonalized_model":
        uri = manifest["source_model"]["s3_uri"]
    try:
        return model_role(uri, allow_legacy=True)
    except ValueError:
        # Local defense training lets callers choose the published snapshot name.
        if manifest.get("artifact_kind") == "trained_model":
            return "defended"
        raise


def is_report_path(path: Path) -> bool:
    return any(part in {"reports", "report_v1"} for part in path.resolve().parts)


def label_report_path(path: Path, label: str) -> Path:
    """Add a role to custom paths inside the report trees when they lack it."""
    parts = path.resolve().parts
    for index, part in enumerate(parts):
        if part in {"reports", "report_v1"}:
            if index == len(parts) - 1:
                return path / label
            pattern = rf"(?:^|[-_.]){re.escape(label)}(?:$|[-_.])"
            if not any(re.search(pattern, component) for component in parts[index + 1:]):
                return path.with_name(f"{label}-{path.name}")
            break
    return path
