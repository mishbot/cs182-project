"""Sample held-out prompts and prepare a paired model response report."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .dataset_artifacts import download_dataset, read_prompt_rows
from .dataset_partition import load_partition, partition_uri
from .inference import generate_many
from .model_artifacts import MANIFEST, download_model, sha256

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

MAX_NEW_TOKENS = 128
CLASSES = (("harmful", "Harmful prompts"), ("unharmful", "Harmless prompts"))
TEX_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
    "\t": "    ",
})


def sample_validation_prompts(
    rows: Sequence[JsonObject], validation: set[int], count: int, seed: int
) -> list[tuple[str, str]]:
    """Return distinct (label, prompt) pairs from the held-out membership."""
    if count < 1 or seed < 0:
        raise ValueError("Samples per class must be positive and seed must be nonnegative.")
    labels: dict[str, set[str]] = {}
    training_prompts = set()
    for index, row in enumerate(rows):
        prompt, label = row.get("prompt"), row.get("prompt_harm_label")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if index not in validation:
            training_prompts.add(prompt)
        if label in ("harmful", "unharmful"):
            labels.setdefault(prompt, set()).add(label)

    rng = random.Random(seed)
    selected = []
    for label, title in CLASSES:
        eligible = [
            prompt for prompt, observed in labels.items()
            if observed == {label} and prompt not in training_prompts
        ]
        if len(eligible) < count:
            raise ValueError(
                f"Requested {count} {title.lower()}, but only {len(eligible)} are eligible."
            )
        selected.extend((label, prompt) for prompt in rng.sample(eligible, count))
    return selected


def compare_validation_outputs(
    s3: S3Client,
    *,
    model_s3_uri: str,
    orthogonalized_model_s3_uri: str,
    dataset_s3_uri: str,
    partition_s3_uri: str | None,
    pinned_manifests: dict[str, str],
    cache_dir: PathInput,
    samples_per_class: int,
    seed: int,
) -> list[tuple[str, str, str, str]]:
    cache = Path(cache_dir)

    def check_pin(uri: str, digest: str) -> None:
        expected = pinned_manifests.get(uri)
        if expected is not None and expected != digest:
            raise ValueError(f"Manifest differs from the experiment checkpoint at {uri}.")

    dataset = download_dataset(s3, dataset_s3_uri, cache / "datasets")
    manifest_path = dataset / MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    dataset_digest = sha256(manifest_path)
    check_pin(f"{dataset_s3_uri.rstrip('/')}/{MANIFEST}", dataset_digest)
    rows = read_prompt_rows(dataset, manifest, "validation")
    partition_s3_uri = partition_s3_uri or partition_uri(dataset_s3_uri, manifest)
    partition, digest = load_partition(s3, partition_s3_uri, manifest, dataset_digest, len(rows))
    check_pin(partition_s3_uri, digest)
    selected = sample_validation_prompts(
        rows, set(partition["splits"]["validation"]), samples_per_class, seed
    )

    model_paths = []
    for uri in (model_s3_uri, orthogonalized_model_s3_uri):
        directory = download_model(s3, uri, cache / "models")
        check_pin(f"{uri.rstrip('/')}/{MANIFEST}", sha256(directory / MANIFEST))
        model_paths.append(directory)
    prompts = [prompt for _, prompt in selected]
    original = generate_many(model_paths[0], prompts, MAX_NEW_TOKENS, "cuda")
    edited = generate_many(model_paths[1], prompts, MAX_NEW_TOKENS, "cuda")
    return [
        (label, prompt, before, after)
        for (label, prompt), before, after in zip(selected, original, edited, strict=True)
    ]


def _tex_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = "".join(char if char.isprintable() or char == "\t" else " " for char in line)
        lines.append(clean.translate(TEX_ESCAPES).strip())
    paragraphs = "\n".join(lines).strip().split("\n\n")
    return "\n\n".join(
        "\\newline\n".join(line for line in paragraph.splitlines() if line)
        for paragraph in paragraphs if paragraph.strip()
    ) or r"\emph{(empty response)}"


def validation_comparison_tex(
    experiment_id: str, seed: int, results: Sequence[tuple[str, str, str, str]]
) -> str:
    """Render untrusted prompt and response text as literal text in a TeX document."""
    parts = [
        r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\definecolor{originalgreen}{RGB}{0,104,56}
\definecolor{editedred}{RGB}{172,30,42}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\setlength{\emergencystretch}{3em}
\begin{document}
\section*{Validation response comparison}""",
        f"Experiment: {_tex_text(experiment_id)}. Sampling seed: {seed}. "
        f"Maximum new tokens per response: {MAX_NEW_TOKENS}.",
    ]
    for label, title in CLASSES:
        entries = [(prompt, before, after) for kind, prompt, before, after in results if kind == label]
        parts.extend((rf"\section{{{title}}}", f"Sampled prompts: {len(entries)}."))
        for number, (prompt, before, after) in enumerate(entries, start=1):
            parts.extend((
                rf"\subsection*{{Prompt {number}}}",
                _tex_text(prompt),
                rf"{{\color{{originalgreen}}\textbf{{Original model}}\par {_tex_text(before)}\par}}",
                rf"{{\color{{editedred}}\textbf{{Orthogonalized model}}\par {_tex_text(after)}\par}}",
            ))
    parts.append(r"\end{document}")
    return "\n\n".join(parts) + "\n"
