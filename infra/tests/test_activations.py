"""Offline sampling, snapshot, Parquet, and real tiny-Qwen extraction tests."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cs182_project import activations as collect
from cs182_project import dataset_artifacts as dataset
from cs182_project import model_artifacts as artifacts
from cs182_project import seeding as seed_dataset

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ML_AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "transformers", "pyarrow"])


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.downloads = []
        self.fail_upload_after = None
        self.fail_put = False

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key))
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def upload_fileobj(self, body, bucket, key, ExtraArgs=None):
        if self.fail_upload_after is not None and len(self.uploads) >= self.fail_upload_after:
            raise OSError("upload interrupted")
        self.objects[(bucket, key)] = body.read()
        self.uploads.append((bucket, key))

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as body:
            self.upload_fileobj(body, bucket, key)

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.fail_put:
            raise OSError("manifest upload failed")
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise RuntimeError("PreconditionFailed")
        self.objects[(Bucket, Key)] = bytes(Body)
        self.uploads.append((Bucket, Key))

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def snapshot(self, bucket, prefix, files, **metadata):
        entries = []
        for name, body in files.items():
            self.objects[(bucket, f"{prefix}/{name}")] = body
            entries.append({"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        manifest = dict(schema_version=1, files=entries, **metadata)
        self.objects[(bucket, f"{prefix}/{artifacts.MANIFEST}")] = json.dumps(manifest).encode()


def dataset_snapshot(s3, train=b"train", test=b"test"):
    s3.snapshot("datasets", "dataset", {
        dataset.SPLITS["train"]: train, dataset.SPLITS["test"]: test, "README.md": b"Attribution",
    }, dataset_id=dataset.DATASET_ID, revision="a" * 40, splits=dataset.SPLITS)


class SamplingTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"prompt": f"{label}-{i}", "prompt_harm_label": label}
            for label in ["harmful", "unharmful"] for i in range(12)
        ]

    def test_seed_is_reproducible_and_classes_balanced(self):
        selected = collect.sample_prompts(self.rows, 10, 7)
        self.assertEqual(selected, collect.sample_prompts(self.rows, 10, 7))
        self.assertNotEqual(selected, collect.sample_prompts(self.rows, 10, 8))
        self.assertEqual(sum(row["harmful"] for row in selected), 5)
        self.assertEqual(len({row["prompt"] for row in selected}), 10)

    def test_odd_sample_extra_is_harmful_and_source_indices_preserved(self):
        selected = collect.sample_prompts(self.rows, 5, 0)
        self.assertEqual(sum(row["harmful"] for row in selected), 3)
        for row in selected:
            self.assertEqual(row["prompt"], self.rows[row["source_row_index"]]["prompt"])

    def test_duplicates_missing_empty_and_conflicting_labels(self):
        rows = [
            {"prompt": "duplicate", "prompt_harm_label": "harmful"},
            {"prompt": "duplicate", "prompt_harm_label": "harmful"},
            {"prompt": "conflict", "prompt_harm_label": "harmful"},
            {"prompt": "conflict", "prompt_harm_label": "unharmful"},
            {"prompt": "missing", "prompt_harm_label": None},
            {"prompt": " ", "prompt_harm_label": "unharmful"},
            {"prompt": None, "prompt_harm_label": "harmful"},
            {"prompt": "benign", "prompt_harm_label": "unharmful"},
        ]
        selected = collect.sample_prompts(rows, 2, 0)
        self.assertEqual({row["source_row_index"] for row in selected}, {0, 7})
        with self.assertRaisesRegex(ValueError, "only 1"):
            collect.sample_prompts(rows, 4, 0)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            collect.sample_prompts(rows, 1, 0)

    def test_new_clis_help_and_invalid_sample_without_dependencies(self):
        for name in ["seed_dataset.py", "collect_activations.py"]:
            result = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([
            sys.executable, str(SCRIPTS / "collect_activations.py"),
            "--model-s3-uri", "s3://a/model", "--dataset-s3-uri", "s3://a/data",
            "--output-s3-uri", "s3://b/runs", "--sample-size", "1",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


class DatasetCacheTest(unittest.TestCase):
    def test_cache_reuse_corruption_and_immutable_manifest(self):
        s3 = MemoryS3()
        dataset_snapshot(s3)
        with tempfile.TemporaryDirectory() as directory:
            local = dataset.download_dataset(s3, "s3://datasets/dataset", directory)
            dataset.download_dataset(s3, "s3://datasets/dataset", directory)
            self.assertEqual(len(s3.downloads), 3)
            (local / dataset.SPLITS["train"]).write_bytes(b"broken")
            dataset.download_dataset(s3, "s3://datasets/dataset", directory)
            self.assertEqual(len(s3.downloads), 4)
            self.assertEqual((local / dataset.SPLITS["train"]).read_bytes(), b"train")
            manifest_key = ("datasets", f"dataset/{artifacts.MANIFEST}")
            manifest = json.loads(s3.objects[manifest_key])
            manifest["revision"] = "b" * 40
            s3.objects[manifest_key] = json.dumps(manifest).encode()
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                dataset.download_dataset(s3, "s3://datasets/dataset", directory)

    def test_incomplete_snapshot_and_corrupt_remote_file_fail(self):
        s3 = MemoryS3()
        dataset_snapshot(s3)
        with tempfile.TemporaryDirectory() as directory:
            s3.objects[("datasets", f"dataset/{dataset.SPLITS['train']}")] = b"broken"
            with self.assertRaisesRegex(ValueError, "Checksum/size"):
                dataset.download_dataset(s3, "s3://datasets/dataset", directory)
            self.assertFalse(list(Path(directory).rglob(artifacts.MANIFEST)))
        manifest = json.loads(s3.objects[("datasets", f"dataset/{artifacts.MANIFEST}")])
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != "README.md"]
        with self.assertRaisesRegex(ValueError, "dataset card"):
            dataset.validate_dataset_manifest(manifest)


@unittest.skipUnless(importlib.util.find_spec("botocore"), "Requires container AWS SDK")
class DatasetSeedingTest(unittest.TestCase):
    def test_resolves_revision_and_publishes_last_without_overwrite(self):
        s3 = MemoryS3()
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "file"
            fixture.write_text("fixture source")
            hub = SimpleNamespace(HfApi=MagicMock(), hf_hub_download=MagicMock(return_value=str(fixture)))
            hub.HfApi.return_value.dataset_info.return_value.sha = "a" * 40
            uri = seed_dataset.seed_dataset(s3, "datasets", "main", directory, hub)
            self.assertEqual(uri, f"s3://datasets/datasets/wildguardmix/{'a' * 40}")
            self.assertEqual(len(s3.uploads), 4)
            self.assertTrue(s3.uploads[-1][1].endswith(artifacts.MANIFEST))
            for call in hub.hf_hub_download.call_args_list:
                self.assertEqual(call.kwargs["revision"], "a" * 40)
            manifest = json.loads(s3.objects[s3.uploads[-1]])
            self.assertEqual(manifest["attribution"]["source"], dataset.DATASET_URL)
            with self.assertRaisesRegex(ValueError, "already published"):
                seed_dataset.seed_dataset(s3, "datasets", "main", directory, hub)
            self.assertEqual(hub.hf_hub_download.call_count, 3)

    def test_failed_seed_never_publishes_completion(self):
        s3 = MemoryS3()
        s3.fail_upload_after = 1
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "file"
            fixture.write_text("fixture source")
            hub = SimpleNamespace(HfApi=MagicMock(), hf_hub_download=MagicMock(return_value=str(fixture)))
            hub.HfApi.return_value.dataset_info.return_value.sha = "a" * 40
            with self.assertRaisesRegex(OSError, "interrupted"):
                seed_dataset.seed_dataset(s3, "datasets", "main", directory, hub)
        self.assertFalse(any(key.endswith(artifacts.MANIFEST) for _, key in s3.objects))


def tiny_qwen():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3, "world": 4, "user": 5, "assistant": 6, ":": 7}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],
        chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}",
    )
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1, sliding_window=None,
    )).eval()
    return model, tokenizer


@unittest.skipUnless(ML_AVAILABLE, "Requires container ML and Parquet dependencies")
class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = tiny_qwen()
        self.rows = [
            {"source_row_index": 0, "prompt": "hello", "prompt_harm_label": "harmful", "harmful": True},
            {"source_row_index": 1, "prompt": "hello world world", "prompt_harm_label": "unharmful", "harmful": False},
        ]
        self.prepared = collect.prepare_inputs(self.tokenizer, self.rows, 64, 128)

    def test_batch_equivalence_final_real_token_and_layer_boundaries(self):
        import numpy as np
        import torch

        together = next(collect.activation_batches(self.model.model, self.tokenizer, self.prepared, 2, "cpu"))
        separately = [rows[0] for rows in collect.activation_batches(self.model.model, self.tokenizer, self.prepared, 1, "cpu")]
        np.testing.assert_allclose([row["activations"] for row in together], [row["activations"] for row in separately], atol=1e-6)
        batch = self.tokenizer.pad([encoded for _, encoded in self.prepared], padding=True, return_tensors="pt")
        reference = {}
        handles = [
            self.model.model.layers[1].register_forward_pre_hook(lambda _m, inputs: reference.update(first=inputs[0].detach().clone())),
            self.model.model.norm.register_forward_pre_hook(lambda _m, inputs: reference.update(last=inputs[0].detach().clone())),
        ]
        with torch.inference_mode():
            normalized = self.model.model(**batch, use_cache=False).last_hidden_state
        for handle in handles:
            handle.remove()
        for i, row in enumerate(together):
            final = row["formatted_input_token_count"] - 1
            np.testing.assert_allclose(row["activations"][0], reference["first"][i, final].numpy(), atol=1e-6)
            np.testing.assert_allclose(row["activations"][1], reference["last"][i, final].numpy(), atol=1e-6)
            self.assertFalse(np.allclose(row["activations"][1], normalized[i, final].numpy()))
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def test_forward_failure_cleans_hooks(self):
        batch = self.tokenizer.pad([encoded for _, encoded in self.prepared], padding=True, return_tensors="pt")
        with patch.object(self.model.model, "forward", side_effect=RuntimeError("forward failed")):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                collect.capture_batch(self.model.model, batch)
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def test_overlong_input_rejected_and_tokenizer_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "Source row 0.*never truncated"):
            collect.prepare_inputs(self.tokenizer, self.rows, 2, 128)
        with self.assertRaisesRegex(ValueError, "model context=2"):
            collect.prepare_inputs(self.tokenizer, self.rows, 64, 2)

    def test_parquet_shards_roundtrip_and_manifest_order(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        s3 = MemoryS3()
        row = dict(self.rows[0], formatted_input_token_count=5, activations=[[float(i)] * 3 for i in range(2)])
        uri = collect.upload_run(s3, "s3://activations/runs", [[row] * 129], {"sample_size": 129}, collect.activation_schema(2, 3))
        manifest = json.loads(s3.objects[("activations", uri.split("activations/", 1)[1])])
        self.assertEqual([shard["rows"] for shard in manifest["shards"]], [128, 1])
        self.assertEqual(manifest["row_count"], 129)
        self.assertTrue(s3.uploads[-1][1].endswith("manifest.json"))
        for shard in manifest["shards"]:
            data = s3.objects[("activations", shard["key"])]
            self.assertEqual(hashlib.sha256(data).hexdigest(), shard["sha256"])
            table = pq.read_table(pa.BufferReader(data))
            self.assertEqual(table.to_pylist()[0], row)
            self.assertEqual(table.schema.field("activations").type.value_type.value_type, pa.float32())

    def test_upload_failure_or_row_mismatch_cannot_publish_success(self):
        row = dict(self.rows[0], formatted_input_token_count=5, activations=[[0.] * 3] * 2)
        for failure in ["upload", "manifest", "row_count"]:
            with self.subTest(failure=failure):
                s3 = MemoryS3()
                s3.fail_upload_after = 1 if failure == "upload" else None
                s3.fail_put = failure == "manifest"
                count = 130 if failure == "row_count" else 129
                with self.assertRaisesRegex(RuntimeError, "Incomplete run"):
                    collect.upload_run(s3, "s3://activations/runs", [[row] * 129], {"sample_size": count}, collect.activation_schema(2, 3))
                self.assertFalse(any(key.endswith("manifest.json") for _, key in s3.objects))

    def test_full_collection_and_overlong_preflight_without_uploads(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        s3 = MemoryS3()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.model.save_pretrained(root / "source")
            self.tokenizer.save_pretrained(root / "source")
            files = {p.name: p.read_bytes() for p in (root / "source").iterdir() if p.is_file()}
            s3.snapshot("artifacts", "model", files, model_id=artifacts.MODEL_ID, revision=artifacts.MODEL_REVISION)
            records = [
                {"prompt": f"hello {'world ' * i}{label}", "prompt_harm_label": label}
                for label in ["harmful", "unharmful"] for i in range(3)
            ]
            buffer = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pylist(records), buffer)
            dataset_snapshot(s3, buffer.getvalue().to_pybytes(), buffer.getvalue().to_pybytes())
            args = SimpleNamespace(
                device="cpu", cache_dir=str(root / "cache"), model_s3_uri="s3://artifacts/model",
                dataset_s3_uri="s3://datasets/dataset", output_s3_uri="s3://activations/runs",
                split="train", sample_size=5, seed=0, batch_size=2, max_input_tokens=64,
            )
            uri = collect.collect(s3, **vars(args))
            manifest = json.loads(s3.objects[("activations", uri.split("activations/", 1)[1])])
            self.assertEqual(manifest["class_counts"], {"harmful": 3, "unharmful": 2})
            self.assertEqual(manifest["num_layers"], 2)
            self.assertEqual(manifest["hidden_size"], 16)
            self.assertEqual(manifest["dataset"]["manifest"]["revision"], "a" * 40)
            self.assertEqual(manifest["activation_semantics"], collect.ACTIVATION_SEMANTICS)
            # The only Parquet files on disk are the two input dataset files.
            self.assertEqual(len(list((root / "cache").rglob("*.parquet"))), 2)
            uploads = len(s3.uploads)
            args.max_input_tokens = 2
            with patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
                with self.assertRaisesRegex(ValueError, "Source row"):
                    collect.collect(s3, **vars(args))
                load.assert_not_called()
            self.assertEqual(len(s3.uploads), uploads)


if __name__ == "__main__":
    unittest.main()
