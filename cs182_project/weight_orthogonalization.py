"""Project Qwen residual-stream writers and publish a separate immutable model snapshot."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._model_config import load_local_model_config
from ._types import Device, JsonObject, PathInput
from .analysis_directions import DEFAULT_RHAT, json_object, load_analysis_direction, read_object
from .model_artifacts import (
    MANIFEST,
    download_model,
    parse_s3_uri,
    publish_model_snapshot,
    sha256,
    validate_manifest,
)

if TYPE_CHECKING:
    import numpy as np
    from mypy_boto3_s3 import S3Client
    from numpy.typing import NDArray
    from torch import Tensor
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model

logger = logging.getLogger(__name__)


def project_weight(
    weight: Tensor, direction: Tensor, *, embedding: bool = False, chunk_size: int = 4096
) -> None:
    """Apply P on the residual axis in float32 chunks, preserving stored parameter dtype."""
    import torch

    if weight.ndim != 2 or direction.ndim != 1 or chunk_size < 1:
        raise ValueError("Expected a matrix, a direction vector, and a positive chunk size.")
    axis = 1 if embedding else 0
    if weight.shape[axis] != direction.numel() or not weight.is_floating_point():
        raise ValueError("Weight residual dimension/dtype is incompatible with the direction.")
    rhat = direction.to(device=weight.device, dtype=torch.float32)
    norm = rhat.norm()
    if not torch.isfinite(rhat).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Direction must be finite and nonzero.")
    rhat = rhat / norm
    with torch.no_grad():
        for start in range(0, weight.shape[1 - axis], chunk_size):
            view = (
                weight[start : start + chunk_size, :]
                if embedding
                else weight[:, start : start + chunk_size]
            )
            values = view.float()
            if embedding:
                update = (values @ rhat)[:, None] * rhat[None, :]
            else:
                update = rhat[:, None] * (rhat @ values)[None, :]
            view.copy_((values - update).to(dtype=weight.dtype))


def orthogonalize_weights(model: Qwen2ForCausalLM, direction: NDArray[np.float64]) -> JsonObject:
    """Change only embedding/attention-output/MLP-output writers; preserve any tied LM head."""
    import torch

    if model.config.model_type != "qwen2":
        raise ValueError(
            "Weight orthogonalization currently supports Qwen2 causal language models."
        )
    backbone = cast("Qwen2Model", model.model)
    embedding_module = cast(torch.nn.Embedding, backbone.embed_tokens)
    output_head = cast(torch.nn.Linear, model.get_output_embeddings())
    rhat = torch.as_tensor(direction, dtype=torch.float32, device=embedding_module.weight.device)
    if (
        rhat.shape != (model.config.hidden_size,)
        or not torch.isfinite(rhat).all()
        or rhat.norm() <= 0
    ):
        raise ValueError("Expected a finite nonzero direction matching the model's hidden size.")
    rhat = rhat / rhat.norm()
    embeddings = embedding_module.weight
    tied = output_head.weight.data_ptr() == embeddings.data_ptr()
    if tied:
        output_head.weight = torch.nn.Parameter(
            output_head.weight.detach().clone(), requires_grad=output_head.weight.requires_grad
        )
    # A checkpoint can request tying even if the two Parameter objects were untied externally.
    # Disable it during serialization too, so reload cannot replace the preserved LM head.
    model.config.tie_word_embeddings = False
    writers: list[tuple[str, Tensor, bool]] = [("model.embed_tokens.weight", embeddings, True)]
    biases: list[tuple[str, Tensor]] = []
    for layer_index, layer in enumerate(backbone.layers):
        for suffix in ("self_attn.o_proj", "mlp.down_proj"):
            module = cast(torch.nn.Linear, layer.get_submodule(suffix))
            prefix = f"model.layers.{layer_index}.{suffix}"
            writers.append((prefix + ".weight", module.weight, False))
            if module.bias is not None:
                biases.append((prefix + ".bias", module.bias))
    modified: list[JsonObject] = []
    for name, weight, embedding in writers:
        logger.info("Projecting %s %s (%s)", name, tuple(weight.shape), weight.dtype)
        project_weight(weight, rhat, embedding=embedding)
        modified.append(
            {
                "name": name,
                "shape": list(weight.shape),
                "storage_dtype": str(weight.dtype),
                "operation": "W P" if embedding else "P W",
            }
        )
    with torch.no_grad():
        for name, bias in biases:
            value = bias.float()
            bias.copy_((value - rhat * (rhat @ value)).to(dtype=bias.dtype))
            modified.append(
                {
                    "name": name,
                    "shape": list(bias.shape),
                    "storage_dtype": str(bias.dtype),
                    "operation": "P b",
                }
            )
    return {
        "method": "residual_writer_rank_one_orthogonalization",
        "compute_dtype": "float32",
        "modified_parameters": modified,
        "lm_head_untied": tied,
        "positional_embeddings": "No learned additive position matrix: Qwen2 uses RoPE; rotary parameters unchanged.",
    }


def publish_model(
    s3: S3Client,
    directory: PathInput,
    bucket: str,
    name: str,
    identifier: str,
    provenance: JsonObject,
) -> str:
    """Publish all model files before the standard snapshot completion marker."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid derived model name.")
    return publish_model_snapshot(
        s3,
        directory,
        bucket,
        f"models/{name}",
        model_id=name,
        revision=identifier,
        artifact_kind="orthogonalized_model",
        provenance=provenance,
    )


def orthogonalize_model(
    s3: S3Client,
    *,
    model_s3_uri: str,
    analysis_s3_uri: str | None = None,
    analysis_root_s3_uri: str | None = None,
    rhat: str = DEFAULT_RHAT,
    device: Device = "auto",
    cache_dir: PathInput = "/cache",
) -> str:
    """Derive a model using a matching analysis; return the new snapshot's S3 URI."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    bucket, prefix = parse_s3_uri(model_s3_uri)
    source_bytes = read_object(s3, bucket, f"{prefix}/{MANIFEST}")
    source = json_object(source_bytes)
    validate_manifest(source)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    config_file = next(entry for entry in source["files"] if entry["path"] == "config.json")
    config_bytes = read_object(s3, bucket, f"{prefix}/config.json")
    if (
        len(config_bytes) != config_file["size"]
        or hashlib.sha256(config_bytes).hexdigest() != config_file["sha256"]
    ):
        raise ValueError("Source config checksum/size mismatch.")
    config_data = json_object(config_bytes)
    if config_data.get("model_type") != "qwen2" or config_data.get("quantization_config"):
        raise ValueError("Expected an unquantized Qwen2 causal language model.")
    selected = load_analysis_direction(
        s3,
        source_sha,
        config_data["hidden_size"],
        config_data["num_hidden_layers"],
        analysis_s3_uri=analysis_s3_uri,
        analysis_root_s3_uri=analysis_root_s3_uri,
        rhat=rhat,
    )
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    directory = download_model(s3, model_s3_uri, Path(cache_dir) / "models")
    if sha256(directory / MANIFEST) != source_sha:
        raise ValueError("Source model manifest changed during direction selection.")
    config = load_local_model_config(directory)
    model = (
        AutoModelForCausalLM.from_pretrained(
            directory,
            config=config,
            torch_dtype="auto",
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .to(resolved_device)
        .eval()
    )
    transformation = orthogonalize_weights(model, selected.vector)
    model.to("cpu")
    identifier = uuid.uuid4().hex
    source_name = str(source.get("model_id", "Qwen1.5-1.8B-Chat")).rsplit("/", 1)[-1]
    source_name = re.sub(r"[^A-Za-z0-9_.-]", "_", source_name).lstrip("._-")
    name = f"{source_name}-{rhat}-{identifier}"
    with tempfile.TemporaryDirectory(prefix="cs182-orthogonalized-") as temporary:
        output = Path(temporary)
        model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
        del model
        # Preserve tokenizer and non-weight assets byte-for-byte. Generated config,
        # generation config, shards and indexes take precedence over source files.
        for entry in source["files"]:
            relative = entry["path"]
            if (
                relative.endswith((".safetensors", ".bin", ".pt", ".pth", ".index.json"))
                or (output / relative).exists()
            ):
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / relative, destination)
        return publish_model(
            s3,
            output,
            bucket,
            name,
            identifier,
            {
                "source_model": {
                    "s3_uri": model_s3_uri,
                    "manifest_sha256": source_sha,
                    "model_id": source.get("model_id"),
                    "revision": source.get("revision"),
                },
                "direction": selected.provenance,
                "transformation": transformation,
            },
        )
