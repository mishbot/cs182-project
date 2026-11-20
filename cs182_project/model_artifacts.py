"""Versioned S3 model snapshots shared by the seeding and inference CLIs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ._types import JsonObject, PathInput

if TYPE_CHECKING:
    from boto3.s3.transfer import TransferConfig
    from mypy_boto3_s3 import S3Client

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
MODEL_REVISION = "e482ee3f73c375a627a16fdf66fd0c8279743ca6"
MANIFEST = "artifact-manifest.json"
logger = logging.getLogger(__name__)


def sha256(path: PathInput) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ValueError("S3 location must be s3://BUCKET/PREFIX (a nonempty prefix).")
    return parsed.netloc, prefix


def validate_files(manifest: JsonObject) -> set[str]:
    """Validate the common file list used by model and dataset snapshots."""
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported or malformed artifact manifest.")
    files = manifest["files"]
    names: set[str] = set()
    for entry in files:
        name = entry["path"]
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or str(path) != name
            or any(part.startswith(".") for part in path.parts)
            or name == MANIFEST
            or name in names
        ):
            raise ValueError(f"Unsafe or duplicate artifact path: {name!r}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid artifact size: {name}")
        checksum = entry["sha256"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            raise ValueError(f"Invalid artifact checksum: {name}")
        names.add(name)
    return names


def validate_manifest(manifest: JsonObject) -> None:
    names = validate_files(manifest)
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if manifest.get("model_format") == "legacy_qwen":
        required = {
            "config.json",
            "tokenizer_config.json",
            "qwen.tiktoken",
            "configuration_qwen.py",
            "modeling_qwen.py",
            "tokenization_qwen.py",
        }
    if not required <= names:
        raise ValueError("Snapshot is missing model or tokenizer configuration.")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Snapshot contains no safetensors weights.")


def download_snapshot(
    s3: S3Client,
    uri: str,
    cache_dir: PathInput,
    validator: Callable[[JsonObject], None],
) -> Path:
    """Fetch only missing/corrupt files; publish each local file after verification."""
    bucket, prefix = parse_s3_uri(uri)
    cache = Path(cache_dir).expanduser() / hashlib.sha256(uri.rstrip("/").encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    # Serialize writers to the same cache, including processes in separate containers.
    with (cache / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        response = s3.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
        try:
            manifest_bytes = response["Body"].read()
        finally:
            response["Body"].close()
        manifest = json.loads(manifest_bytes)
        validator(manifest)
        # A published prefix is immutable. Reject a changed manifest rather than
        # mixing files from different model snapshots in the same local directory.
        previous = cache / MANIFEST
        if previous.exists() and previous.read_bytes() != manifest_bytes:
            raise ValueError("The S3 snapshot manifest changed; publish a new prefix instead.")
        for entry in manifest["files"]:
            target = cache / entry["path"]
            if (
                target.is_file()
                and target.stat().st_size == entry["size"]
                and sha256(target) == entry["sha256"]
            ):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".download-")
            os.close(fd)
            try:
                s3.download_file(bucket, f"{prefix}/{entry['path']}", temporary)
                if (
                    Path(temporary).stat().st_size != entry["size"]
                    or sha256(temporary) != entry["sha256"]
                ):
                    raise ValueError(f"Checksum/size mismatch for {entry['path']}")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        previous.write_bytes(manifest_bytes)
    return cache


def download_model(s3: S3Client, uri: str, cache_dir: PathInput) -> Path:
    return download_snapshot(s3, uri, cache_dir, validate_manifest)


def publish_model_snapshot(
    s3: S3Client,
    directory: PathInput,
    bucket: str,
    prefix: str,
    *,
    model_id: str,
    revision: str,
    artifact_kind: str,
    provenance: JsonObject | None = None,
    transfer_config: TransferConfig | None = None,
    verify_uploaded_sizes: bool = False,
) -> str:
    """Publish an immutable snapshot; upload completion only after file verification succeeds."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {root}")
    if (
        not prefix
        or str(PurePosixPath(prefix)) != prefix
        or PurePosixPath(prefix).is_absolute()
        or ".." in PurePosixPath(prefix).parts
    ):
        raise ValueError("Expected a nonempty relative snapshot prefix without '..'.")
    files: list[JsonObject] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f"Snapshot contains a symlink or unsupported filesystem entry: {path}")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST:
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        **(provenance or {}),
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "artifact_kind": artifact_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    validate_manifest(manifest)
    total = sum(entry["size"] for entry in files)
    logger.info(
        "Publishing %s files (%s bytes) from %s to s3://%s/%s",
        len(files),
        total,
        root,
        bucket,
        prefix,
    )
    try:
        for entry in files:
            key = f"{prefix}/{entry['path']}"
            logger.info("Uploading %s (%s bytes)", entry["path"], entry["size"])
            if transfer_config is None:
                s3.upload_file(str(root / entry["path"]), bucket, key)
            else:
                s3.upload_file(
                    str(root / entry["path"]),
                    bucket,
                    key,
                    Config=transfer_config,
                    ExtraArgs={"ChecksumAlgorithm": "SHA256"},
                )
            if verify_uploaded_sizes:
                remote = s3.head_object(Bucket=bucket, Key=key)
                if remote.get("ContentLength") != entry["size"]:
                    raise ValueError(f"Uploaded size mismatch for {key}.")
            logger.info("Uploaded %s", entry["path"])
        s3.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{MANIFEST}",
            Body=json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as error:
        raise RuntimeError(
            f"Incomplete model at s3://{bucket}/{prefix}; no completion manifest was published: {error}"
        ) from error
    logger.info("Completed %s files (%s bytes): s3://%s/%s", len(files), total, bucket, prefix)
    return f"s3://{bucket}/{prefix}"
