"""Versioned S3 model snapshots shared by the seeding and inference CLIs."""

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlparse

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
MODEL_REVISION = "e482ee3f73c375a627a16fdf66fd0c8279743ca6"
MANIFEST = "artifact-manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ValueError("S3 location must be s3://BUCKET/PREFIX (a nonempty prefix).")
    return parsed.netloc, prefix


def validate_files(manifest):
    """Validate the common file list used by model and dataset snapshots."""
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported or malformed artifact manifest.")
    files = manifest["files"]
    names = set()
    for entry in files:
        name = entry["path"]
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name or any(part.startswith(".") for part in path.parts)
                or name == MANIFEST or name in names):
            raise ValueError(f"Unsafe or duplicate artifact path: {name!r}")
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError(f"Invalid artifact size: {name}")
        checksum = entry["sha256"]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"Invalid artifact checksum: {name}")
        names.add(name)
    return names


def validate_manifest(manifest):
    names = validate_files(manifest)
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= names:
        raise ValueError("Snapshot is missing model or tokenizer configuration.")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Snapshot contains no safetensors weights.")


def download_snapshot(s3, uri, cache_dir, validator):
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
            if target.is_file() and target.stat().st_size == entry["size"] and sha256(target) == entry["sha256"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".download-")
            os.close(fd)
            try:
                s3.download_file(bucket, f"{prefix}/{entry['path']}", temporary)
                if Path(temporary).stat().st_size != entry["size"] or sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Checksum/size mismatch for {entry['path']}")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        previous.write_bytes(manifest_bytes)
    return cache


def download_model(s3, uri, cache_dir):
    return download_snapshot(s3, uri, cache_dir, validate_manifest)
