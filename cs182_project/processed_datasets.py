"""Normalized refusal-direction corpus, with explicitly attributed upstream definitions.

Dataset selection and preprocessing definitions are adapted from andyrdt/refusal_direction,
dataset/generate_datasets.ipynb at 9d852fae1a9121c78b29142de733cb1340770cc3 (Apache-2.0).
Adapted definitions: download_advbench, download_malicious_instruct, download_tdc2023,
and download_alpaca. Original dataset publishers retain their own dataset terms.

Project-specific changes: the three-column Parquet schema, inferred harm labels,
non-adversarial flags, immutable S3 publication, and 80/20 grouped membership. No
128-example cap or paper-specific train/val/test construction is copied here.
See third_party/refusal_direction/{LICENSE,NOTICE} and the generated corpus README.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .dataset_artifacts import (
    CORPUS_FILES,
    CORPUS_ID,
    CORPUS_NAME,
    prompt_schema,
    validate_dataset_manifest,
)
from .dataset_partition import publish_partition_rows
from .model_artifacts import MANIFEST, sha256

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)
DEFINITIONS_REVISION = "9d852fae1a9121c78b29142de733cb1340770cc3"
DEFINITIONS_URL = f"https://github.com/andyrdt/refusal_direction/blob/{DEFINITIONS_REVISION}/dataset/generate_datasets.ipynb"
ALPACA_ID = "tatsu-lab/alpaca"
GITHUB_SOURCES = [
    ("advbench", "llm-attacks/llm-attacks", ["data/advbench/harmful_behaviors.csv"]),
    ("maliciousinstruct", "Princeton-SysML/Jailbreak_LLM", ["data/MaliciousInstruct.txt"]),
    (
        "tdc2023",
        "centerforaisafety/tdc2023-starter-kit",
        ["red_teaming/data/dev/behaviors.json", "red_teaming/data/test/behaviors.json"],
    ),
]


@dataclass(frozen=True)
class DatasetSource:
    name: str
    repository: str
    revision: str
    files: dict[str, Path]
    license: str | list[str] | None = None


def _request(url: str) -> bytes:
    headers = {"User-Agent": "cs182-project dataset seeder"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as response:
        return response.read()


def _cached_file(url: str, path: Path) -> Path:
    checksum = path.with_name(path.name + ".sha256")
    if path.is_file() and checksum.is_file() and sha256(path) == checksum.read_text():
        return path
    data = _request(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    temporary.replace(path)
    checksum.write_text(hashlib.sha256(data).hexdigest())
    return path


def download_sources(cache_dir: PathInput) -> list[DatasetSource]:
    """Resolve each source once, then fetch files exclusively by its immutable revision."""
    import huggingface_hub as hub

    cache = Path(cache_dir).expanduser()
    result: list[DatasetSource] = []
    for name, repository, filenames in GITHUB_SOURCES:
        revision = json.loads(_request(f"https://api.github.com/repos/{repository}/commits/main"))[
            "sha"
        ]
        files: dict[str, Path] = {}
        for filename in [*filenames, "LICENSE"]:
            try:
                files[filename] = _cached_file(
                    f"https://raw.githubusercontent.com/{repository}/{revision}/{filename}",
                    cache / name / revision / filename,
                )
            except urllib.error.HTTPError as error:
                if filename != "LICENSE" or error.code != 404:
                    raise
        result.append(DatasetSource(name, f"https://github.com/{repository}", revision, files))
    api = hub.HfApi()
    info = api.dataset_info(ALPACA_ID)
    revision = info.sha
    if revision is None:
        raise ValueError("Alpaca did not resolve to an immutable revision.")
    names = api.list_repo_files(ALPACA_ID, repo_type="dataset", revision=revision)
    filenames = sorted(
        name for name in names if name.startswith("data/train-") and name.endswith(".parquet")
    )
    if not filenames:
        raise ValueError("Alpaca has no declared training parquet files.")
    if "README.md" in names:
        filenames.append("README.md")
    files = {
        name: Path(
            hub.hf_hub_download(
                ALPACA_ID,
                name,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache / "huggingface"),
            )
        )
        for name in filenames
    }
    card = info.card_data.to_dict() if info.card_data is not None else {}
    result.append(
        DatasetSource(
            "alpaca",
            f"https://huggingface.co/datasets/{ALPACA_ID}",
            revision,
            files,
            card.get("license"),
        )
    )
    return result


def process_source(source: DatasetSource) -> tuple[list[JsonObject], JsonObject]:
    """Apply the attributed source-to-instruction definitions; add only project input columns."""
    import pyarrow.parquet as pq

    if source.name == "advbench":
        path = source.files["data/advbench/harmful_behaviors.csv"]
        prompts = [
            row["goal"] for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
        ]
        input_count = len(prompts)
    elif source.name == "maliciousinstruct":
        prompts = (
            source.files["data/MaliciousInstruct.txt"].read_text(encoding="utf-8-sig").splitlines()
        )
        input_count = len(prompts)
    elif source.name == "tdc2023":
        prompts = []
        for filename in (
            "red_teaming/data/dev/behaviors.json",
            "red_teaming/data/test/behaviors.json",
        ):
            prompts.extend(json.loads(source.files[filename].read_text()))
        input_count = len(prompts)
    elif source.name == "alpaca":
        records = [
            row
            for name, path in source.files.items()
            if name.endswith(".parquet")
            for row in pq.read_table(path, columns=["instruction", "input"]).to_pylist()
        ]
        input_count = len(records)
        prompts = [row["instruction"] for row in records if row["input"].strip() == ""]
    else:
        raise ValueError(f"Unsupported corpus source: {source.name}")
    rows = [
        {
            "prompt": prompt.strip(),
            "prompt_harm_label": "unharmful" if source.name == "alpaca" else "harmful",
            "adversarial": False,
        }
        for prompt in prompts
        if prompt.strip()
    ]
    return rows, {
        "input_rows": input_count,
        "eligible_before_empty_filter": len(prompts),
        "output_rows": len(rows),
        "empty_prompts_omitted": len(prompts) - len(rows),
    }


def _readme(sources: list[JsonObject]) -> str:
    return (
        "# Refusal-direction corpus\n\n"
        f"Dataset source selection and preprocessing definitions adapted from {DEFINITIONS_URL}.\n"
        "The andyrdt code is Apache-2.0; its license and adaptation notice accompany this snapshot.\n"
        "Original dataset publishers retain their own data licenses/terms; see the per-source provenance.\n\n"
        "Upstream-derived definitions: AdvBench goal strings; MaliciousInstruct instruction lines; "
        "TDC2023 development then test behaviors; Alpaca instruction-only rows with empty input. "
        "Surrounding whitespace is stripped.\n\n"
        "Project-specific choices: only prompt, prompt_harm_label, adversarial columns; the three "
        "harmful sources are labeled harmful, Alpaca unharmful, and all rows non-adversarial. "
        "Alpaca's label is an experimental convention, not human annotation or moderation. "
        "All processed rows are stored, with no source cap and no seeding-time deduplication. "
        "80/20 grouped train/validation membership, seed 0, is published separately; sampling "
        "balances the pooled harmful/harmless classes. These are not andyrdt's paper splits.\n\n"
        "Source-row indices use concatenation in the listed file order; offsets are in the manifest.\n\n"
        + "\n".join(
            f"- {item['file']}: {item['row_count']} rows, offset {item['row_offset']}; "
            f"{item['repository']} at {item['revision']}"
            for item in sources
        )
        + "\n"
    )


def seed_corpus(s3: S3Client, bucket: str, cache_dir: PathInput = ".cache/datasets") -> str:
    """Publish four processed files and retryable default membership, with an immutable identity."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from botocore.exceptions import ClientError

    packages = download_sources(cache_dir)
    if [source.name + ".parquet" for source in packages] != CORPUS_FILES:
        raise ValueError("Expected the four corpus sources in canonical order.")
    all_rows: list[JsonObject] = []
    sources: list[JsonObject] = []
    with tempfile.TemporaryDirectory(prefix="cs182-corpus-") as temporary:
        directory = Path(temporary)
        for source, filename in zip(packages, CORPUS_FILES):
            rows, counts = process_source(source)
            pq.write_table(
                pa.Table.from_pylist(rows, schema=prompt_schema()),
                directory / filename,
                compression="zstd",
            )
            sources.append(
                {
                    "name": source.name,
                    "file": filename,
                    "row_count": len(rows),
                    "row_offset": len(all_rows),
                    "repository": source.repository,
                    "revision": source.revision,
                    "license": source.license,
                    "processing_counts": counts,
                    "raw_files": [
                        {"path": name, "sha256": sha256(path), "size": path.stat().st_size}
                        for name, path in source.files.items()
                    ],
                }
            )
            all_rows.extend(rows)
            for name, path in source.files.items():
                if name in {"LICENSE", "README.md"}:
                    target = directory / "attribution" / source.name / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        attribution = {
            "definitions_url": DEFINITIONS_URL,
            "definitions_revision": DEFINITIONS_REVISION,
            "code_license": "Apache-2.0",
            "scope": "source selection and instruction preprocessing only",
        }
        identity = {
            "source_files": CORPUS_FILES,
            "sources": sources,
            "preprocessing": {
                "version": 1,
                "schema": "prompt:string,prompt_harm_label:string,adversarial:bool",
                "writer": f"pyarrow-{pa.__version__};zstd",
                "attribution": attribution,
            },
        }
        revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        prefix = f"datasets/{CORPUS_NAME}/{revision}"
        uri = f"s3://{bucket}/{prefix}"
        (directory / "README.md").write_text(_readme(sources))
        notices = directory / "attribution" / "andyrdt-refusal-direction"
        notices.mkdir(parents=True, exist_ok=True)
        for filename in ("LICENSE", "NOTICE"):
            shutil.copyfile(
                Path(__file__).parent / "third_party/refusal_direction" / filename,
                notices / filename,
            )
        files = [
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "dataset_id": CORPUS_ID,
            "revision": revision,
            **identity,
            "row_count": len(all_rows),
            "files": files,
        }
        validate_dataset_manifest(manifest)
        data = json.dumps(manifest, sort_keys=True, indent=2).encode()
        try:
            response = s3.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")
        except s3.exceptions.NoSuchKey:
            existing = None
        else:
            with response["Body"] as body:
                existing = body.read()
        if existing is not None:
            if existing != data:
                raise ValueError(
                    f"Existing immutable corpus differs at {uri}; update the preprocessing version."
                )
            logger.info("Reusing completed corpus %s", uri)
        else:
            for entry in files:
                logger.info("Uploading %s (%s bytes)", entry["path"], entry["size"])
                s3.upload_file(str(directory / entry["path"]), bucket, f"{prefix}/{entry['path']}")
            try:
                s3.put_object(
                    Bucket=bucket,
                    Key=f"{prefix}/{MANIFEST}",
                    Body=data,
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") != "PreconditionFailed":
                    raise
                with s3.get_object(Bucket=bucket, Key=f"{prefix}/{MANIFEST}")["Body"] as body:
                    if body.read() != data:
                        raise ValueError(
                            f"Concurrent corpus publication differs at {uri}."
                        ) from error
        partition = publish_partition_rows(
            s3, uri, manifest, hashlib.sha256(data).hexdigest(), all_rows
        )
        logger.info(
            "Corpus complete: %s rows; train/validation membership: %s", len(all_rows), partition
        )
        return uri
