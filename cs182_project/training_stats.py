"""Durable, periodically published training statistics; failed uploads stay retryable."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .model_artifacts import parse_s3_uri

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: JsonObject) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TrainingStats:
    """Single writer per run. A local journal survives network and worker failures.

    Batch names encode journal line ranges. Retrying an ambiguous PUT writes the
    same bytes to the same key. A successful local marker avoids redundant PUTs.
    The persistent output directory, rather than RAM, is the upload queue.
    """

    def __init__(
        self,
        s3: S3Client,
        directory: PathInput,
        *,
        flush_seconds: float = 60,
        persist: Callable[[], None] | None = None,
    ) -> None:
        if flush_seconds <= 0:
            raise ValueError("Statistics flush interval must be positive.")
        self.s3 = s3
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "run.json").read_text())
        self.bucket, self.prefix = parse_s3_uri(self.metadata["stats_s3_uri"])
        self.batches = self.directory / "batches"
        self.batches.mkdir(exist_ok=True)
        self.journal = self.directory / "metrics.jsonl"
        self.journal.touch(exist_ok=True)
        self.interval = flush_seconds
        self.persist = persist or (lambda: None)
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def record(self, step: int, losses: JsonObject) -> None:
        record = {
            "schema_version": 1,
            "run_id": self.metadata["run_id"],
            "timestamp": utc_now(),
            "step": step,
            **losses,
        }
        line = json.dumps(record, allow_nan=False) + "\n"
        with self._lock, self.journal.open("a") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def _seal(self) -> None:
        with self._lock:
            existing = sorted(self.batches.glob("*.jsonl"))
            start = int(existing[-1].stem.split("-")[1]) if existing else 0
            lines = self.journal.read_text().splitlines(keepends=True)
            # A process killed mid-write can leave an incomplete last record.
            if lines and not lines[-1].endswith("\n"):
                lines.pop()
            if len(lines) > start:
                target = self.batches / f"{start:012d}-{len(lines):012d}.jsonl"
                temporary = target.with_suffix(".tmp")
                temporary.write_text("".join(lines[start:]))
                temporary.replace(target)

    def pending(self) -> list[str]:
        return [
            str(path)
            for path in sorted(self.batches.glob("*.jsonl"))
            if not path.with_suffix(".sent").exists()
        ]

    def flush(self) -> list[str]:
        """Seal on disk before networking; leave unsuccessful objects for retry."""
        with self._flush_lock:
            self._seal()
            self._persist()
            files = [self.directory / "run.json", self.directory / "summary.json"]
            files.extend(Path(path) for path in self.pending())
            failed_metadata: list[str] = []
            for path in files:
                if not path.exists():
                    continue
                try:
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=f"{self.prefix}/{path.relative_to(self.directory).as_posix()}",
                        Body=path.read_bytes(),
                        ContentType=(
                            "application/x-ndjson"
                            if path.suffix == ".jsonl"
                            else "application/json"
                        ),
                    )
                except Exception:
                    logger.warning("Statistics upload pending: %s", path, exc_info=True)
                    if path.suffix != ".jsonl":
                        failed_metadata.append(str(path))
                else:
                    if path.suffix == ".jsonl":
                        path.with_suffix(".sent").touch()
            self._persist()
            return self.pending() + failed_metadata

    def _persist(self) -> None:
        try:
            self.persist()
        except Exception:
            logger.exception("Could not commit training outputs; will retry on the next flush")

    def _upload_loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.flush()
            except Exception:
                # A volume commit failure must not terminate background retries.
                logger.exception("Could not persist statistics; will retry")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._upload_loop, daemon=True)
        self._thread.start()

    def close(self) -> list[str]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.flush()
