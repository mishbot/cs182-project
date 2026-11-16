"""Stage immutable snapshots locally so flock never runs on a Modal Volume."""

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from cs182_project.dataset_artifacts import validate_dataset_manifest
from cs182_project.model_artifacts import MANIFEST, sha256, validate_manifest

logger = logging.getLogger(__name__)


def _atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".cache-")
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]


class SnapshotCache:
    """Copy verified files between a disposable working directory and a Volume.

    The core downloader still fetches the authoritative manifest from S3 on
    every invocation. Only complete snapshots are published to the backing
    directory, with the manifest last. Concurrent writers publish identical
    immutable data via unique temporary files; no distributed lock is needed.
    """

    def __init__(self, backing_dir, working_dir, snapshots):
        self.backing_dir = Path(backing_dir)
        self.working_dir = Path(working_dir)
        self.snapshots = snapshots  # Pairs of ("models" or "datasets", S3 URI).

    def _locations(self):
        for kind, uri in self.snapshots:
            digest = hashlib.sha256(uri.rstrip("/").encode()).hexdigest()
            relative = Path(kind) / digest
            validator = validate_manifest if kind == "models" else validate_dataset_manifest
            yield self.backing_dir / relative, self.working_dir / relative, validator

    @staticmethod
    def _manifest(directory, validator):
        try:
            data = (directory / MANIFEST).read_bytes()
            manifest = json.loads(data)
            validator(manifest)
            return data, manifest
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Ignoring malformed cache manifest in %s", directory)
            return None

    def restore(self):
        for backing, working, validator in self._locations():
            result = self._manifest(backing, validator)
            if result is None:
                continue
            _, manifest = result
            restored = 0
            for entry in manifest["files"]:
                source = backing / entry["path"]
                if _matches(source, entry):
                    _atomic_copy(source, working / entry["path"])
                    restored += 1
            # Preserve the cached identity even if individual files need repair.
            # The downloader rejects a changed S3 manifest for this same URI.
            _atomic_copy(backing / MANIFEST, working / MANIFEST)
            logger.info("Restored %s verified cached files from %s", restored, backing)

    def publish(self):
        for backing, working, validator in self._locations():
            result = self._manifest(working, validator)
            if result is None:
                continue
            data, manifest = result
            if not all(_matches(working / entry["path"], entry) for entry in manifest["files"]):
                continue
            previous = self._manifest(backing, validator)
            if previous is not None and previous[0] != data:
                raise ValueError("The cached snapshot manifest changed; use a new S3 prefix.")
            for entry in manifest["files"]:
                destination = backing / entry["path"]
                if not _matches(destination, entry):
                    _atomic_copy(working / entry["path"], destination)
            _atomic_copy(working / MANIFEST, backing / MANIFEST)
