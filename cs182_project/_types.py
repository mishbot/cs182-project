"""Shared data shapes and function-boundary types for the library."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from huggingface_hub import HfApi
    from torch import Tensor
    from transformers.tokenization_utils_base import BatchEncoding

PathInput: TypeAlias = str | PathLike[str]
Device: TypeAlias = Literal["auto", "cpu", "cuda"]
HarmLabel: TypeAlias = Literal["harmful", "unharmful"]
# Manifests are untrusted JSON until checked by the snapshot validators.
JsonObject: TypeAlias = dict[str, Any]


class ArtifactFile(TypedDict):
    path: str
    size: int
    sha256: str


class PromptRow(TypedDict):
    source_row_index: int
    prompt: str
    prompt_harm_label: HarmLabel
    harmful: bool


class PromptGroup(TypedDict):
    source_row_index: int
    prompt: str
    labels: set[HarmLabel]


class PreparedPrompt(PromptRow):
    formatted_input_token_count: int


PreparedInput: TypeAlias = tuple[PreparedPrompt, "BatchEncoding"]


class TensorBatch(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor


class ActivationRow(PreparedPrompt):
    activations: list[list[float]]


class SnapshotSource(TypedDict):
    s3_uri: str
    manifest: JsonObject
    manifest_sha256: str


class DatasetSource(SnapshotSource):
    split: str


class _SampleSize(TypedDict):
    sample_size: int


class RunMetadata(_SampleSize, total=False):
    model: SnapshotSource
    dataset: DatasetSource
    seed: int
    batch_size: int
    max_input_tokens: int
    device: str
    compute_dtype: str
    activation_dtype: str
    num_layers: int
    hidden_size: int
    layer_index_base: int
    activation_semantics: str
    sampling: str
    selected_source_row_indices: list[int]
    class_counts: dict[HarmLabel, int]


class Shard(TypedDict):
    key: str
    rows: int
    size: int
    sha256: str


class RunManifest(RunMetadata):
    schema_version: int
    status: str
    run_id: str
    created_at: str
    row_count: int
    shards: list[Shard]


class DatasetHub(Protocol):
    """The Hugging Face module interface accepted by dataset seeding."""

    def HfApi(self) -> HfApi: ...

    def hf_hub_download(
        self,
        repo_id: str,
        *,
        repo_type: Literal["dataset"],
        revision: str,
        filename: str,
        cache_dir: str,
    ) -> str: ...
