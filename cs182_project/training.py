"""Step-based defender training using pinned project snapshots and streamed losses."""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._types import JsonObject, PathInput
from .model_artifacts import MANIFEST, download_model, parse_s3_uri, publish_model_snapshot, sha256
from .qwen_model import format_chat, load_model, load_tokenizer
from .training_data import base_target, collate, training_prompts
from .training_losses import training_step
from .training_stats import TrainingStats, utc_now, write_json

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)
DEFAULTS: JsonObject = {
    "n_harmless_train": 2000,
    "k": 4,
    "batch_per_class": 16,
    "steps": 1500,
    "lr": 1e-4,
    "alpha": 1.0,
    "lam_max": 1.0,
    "warmup_steps": 100,
    "ramp_steps": 300,
    "lora_r": 32,
    "lora_alpha": 64,
    "grad_checkpointing": True,
    "checkpoint_every": 250,
    "stats_every_steps": 20,
    "stats_flush_seconds": 60,
    "export_merged": True,
    "seed": 0,
}
DRY_RUN: JsonObject = {
    "steps": 20,
    "warmup_steps": 5,
    "ramp_steps": 5,
    "stats_every_steps": 2,
    "checkpoint_every": 10,
    "n_harmless_train": 64,
}


def training_config(overrides: JsonObject | None = None) -> JsonObject:
    overrides = overrides or {}
    unknown = overrides.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    for key in (
        "steps",
        "batch_per_class",
        "n_harmless_train",
        "ramp_steps",
        "lora_r",
        "lora_alpha",
        "checkpoint_every",
        "stats_every_steps",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "k", "warmup_steps"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    import math

    for key in ("lr", "alpha", "lam_max", "stats_flush_seconds"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["lr"] == 0 or cfg["stats_flush_seconds"] == 0:
        raise ValueError("lr and stats_flush_seconds must be positive")
    for key in ("grad_checkpointing", "export_merged"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    return cfg


def export_model(model: Any, tokenizer: Any, base_dir: Path, output: Path) -> float:
    """Merge in FP32, verify logits, and preserve standalone original-Qwen files."""
    import torch

    model.eval().float()
    ids = torch.tensor([format_chat(tokenizer, "Hi.")["input_ids"]], device=model.device)
    with torch.no_grad():
        before = model(input_ids=ids, use_cache=False).logits
        merged = model.merge_and_unload()
        after = merged(input_ids=ids, use_cache=False).logits
        difference = (before - after).abs().max().item()
        if not torch.allclose(before, after, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Merged model logits differ from adapter by {difference}")
    merged.half().save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)
    for source in base_dir.iterdir():
        if (
            source.is_file()
            and not source.name.startswith(".")
            and source.name != MANIFEST
            and not source.name.endswith((".bin", ".safetensors", ".index.json", ".pt"))
            and not (output / source.name).exists()
        ):
            shutil.copyfile(source, output / source.name)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config.update(bf16=False, fp16=True, fp32=False, use_flash_attn=False)
    write_json(config_path, config)
    # Check the actual serialized export, not only the in-memory merge.
    del before, after
    reloaded = load_model(output, "cpu")
    with torch.no_grad():
        reference = merged.cpu().float()(input_ids=ids.cpu(), use_cache=False).logits
        actual = reloaded(input_ids=ids.cpu(), use_cache=False).logits
        if not torch.allclose(reference, actual, rtol=1e-4, atol=1e-4):
            raise ValueError("Serialized merged checkpoint failed reload verification")
    return float(difference)


def train(
    s3: S3Client,
    *,
    model_s3_uri: str,
    dataset_s3_uri: str,
    stats_s3_uri: str,
    model_output_s3_uri: str,
    run_name: str,
    output_dir: PathInput = "reports/training",
    partition_s3_uri: str | None = None,
    device: str = "auto",
    overrides: JsonObject | None = None,
    persist: Callable[[], None] | None = None,
) -> JsonObject:
    cfg = training_config(overrides)
    for uri in (model_s3_uri, dataset_s3_uri, stats_s3_uri, model_output_s3_uri):
        parse_s3_uri(uri)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_name):
        raise ValueError("Run name must use letters, digits, '.', '_', or '-'.")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    run_id = f"{run_name}-{uuid.uuid4().hex}"
    directory = Path(output_dir) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "run_name": run_name,
        "created_at": utc_now(),
        "model_s3_uri": model_s3_uri,
        "dataset_s3_uri": dataset_s3_uri,
        "stats_s3_uri": f"{stats_s3_uri.rstrip('/')}/{run_id}",
        "configuration": cfg,
        "prompt_format": "original-qwen-chatml-v1",
        "refusal_token_ids": [40, 2121],
        "refusal_threshold": 0,
    }
    write_json(directory / "run.json", metadata)
    summary: JsonObject = {"run_id": run_id, "status": "running", "completed_steps": 0}
    write_json(directory / "summary.json", summary)
    stats = TrainingStats(s3, directory, flush_seconds=cfg["stats_flush_seconds"], persist=persist)
    stats.start()
    logger.info("Training outputs: %s; statistics: %s", directory, metadata["stats_s3_uri"])
    try:
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        base_dir = download_model(s3, model_s3_uri, directory / "cache/models")
        harmful, benign, provenance = training_prompts(
            s3,
            dataset_s3_uri,
            partition_s3_uri,
            directory / "cache/datasets",
            cfg["seed"],
            cfg["n_harmless_train"],
        )
        metadata.update(provenance)
        metadata["model_manifest_sha256"] = sha256(base_dir / MANIFEST)
        tokenizer = load_tokenizer(base_dir)
        if [tokenizer.decode([i]) for i in (40, 2121)] != ["I", "As"]:
            raise ValueError("Tokenizer does not match Qwen refusal token IDs")
        model = load_model(base_dir, device, for_training=True)
        cfg["layers"] = list(range(model.config.num_hidden_layers + 1 - cfg["k"]))
        cfg["n_eoi"] = len(
            tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", allowed_special="all")
        )
        if not cfg["layers"]:
            raise ValueError("k excludes all residual layers")
        write_json(directory / "run.json", metadata)
        examples = []
        for prompts, is_harmful, max_target in ((harmful, True, 64), (benign, False, 128)):
            selected = []
            for prompt in prompts:
                example = base_target(
                    model, tokenizer, prompt, harmful=is_harmful, max_target=max_target
                )
                if example is not None:
                    selected.append(example)
            if len(selected) < cfg["batch_per_class"]:
                raise ValueError(
                    f"Need {cfg['batch_per_class']} eligible examples per class after filtering; "
                    f"found {len(selected)} {'harmful' if is_harmful else 'benign'} examples"
                )
            examples.append(selected)
        summary.update(n_harmful_kept=len(examples[0]), n_harmless_train=len(examples[1]))
        from peft import LoraConfig, get_peft_model

        model.config.use_cache = False
        if cfg["grad_checkpointing"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=0.05,
                target_modules=["c_attn", "c_proj", "w1", "w2"],
                task_type="CAUSAL_LM",
            ),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
        )
        for step in range(cfg["steps"]):
            model.train()
            batches = [
                collate(tokenizer, random.sample(ex, cfg["batch_per_class"]), model.device)
                for ex in examples
            ]
            loss, logs = training_step(model, batches[0], batches[1], step, cfg)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at step {step}: {logs}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            summary["completed_steps"] = step + 1
            if step % cfg["stats_every_steps"] == 0 or step == cfg["steps"] - 1:
                stats.record(step, logs)
                logger.info("step %s: %s", step, logs)
            if step > 0 and step % cfg["checkpoint_every"] == 0:
                model.save_pretrained(directory / f"step{step}")
                write_json(directory / "summary.json", summary)
                if persist:
                    persist()
        model.save_pretrained(directory / "adapter")
        if persist:
            persist()
        if cfg["export_merged"]:
            summary["merge_max_logit_difference"] = export_model(
                model, tokenizer, base_dir, directory / "merged"
            )
            bucket, prefix = parse_s3_uri(model_output_s3_uri)
            summary["model_s3_uri"] = publish_model_snapshot(
                s3,
                directory / "merged",
                bucket,
                f"{prefix}/{run_id}",
                model_id=run_id,
                revision=run_id.rsplit("-", 1)[1],
                artifact_kind="trained_model",
                provenance={"model_format": "legacy_qwen", "training": metadata},
            )
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at"] = utc_now()
        write_json(directory / "summary.json", summary)
        pending = stats.close()
        summary["pending_batches"] = pending
        summary["statistics_delivery"] = "pending" if pending else "complete"
        write_json(directory / "summary.json", summary)
        stats.flush()
        if pending:
            logger.warning(
                "Unsent statistics in %s; retry with train_defense.py --retry-dir", directory
            )
    return summary
