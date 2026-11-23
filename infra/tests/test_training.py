"""CPU training, loss math, input isolation, and standalone snapshot interoperability."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cs182_project import training, training_data

ML = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "pyarrow"))


class TrainingInputTest(unittest.TestCase):
    def test_configuration_rejects_unsafe_or_unknown_settings(self):
        for overrides in ({"steps": 0}, {"ramp_steps": 0}, {"lr": float("nan")},
                          {"stats_every_steps": -1}, {"typo": 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                training.training_config(overrides)

    def test_training_membership_deduplication_and_ambiguous_labels(self):
        rows = [dict(prompt=p, prompt_harm_label=l) for p, l in [
            ("harm", "harmful"), ("harm", "harmful"), ("benign", "unharmful"),
            ("held-out", "harmful"), ("ambiguous", "harmful"), ("ambiguous", "unharmful"),
        ]]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "artifact-manifest.json").write_text('{}')
            with patch.object(training_data, "download_dataset", return_value=Path(directory)), \
                 patch.object(training_data, "read_prompt_rows", return_value=rows), \
                 patch.object(training_data, "load_partition", return_value=({"splits": {"train": [0, 1, 2, 4, 5]}}, "digest")):
                harmful, benign, provenance = training_data.training_prompts(
                    MagicMock(), "s3://datasets/pinned", "s3://datasets/partition", directory, 0, 20,
                )
        self.assertEqual(harmful, ["harm"])
        self.assertEqual(benign, ["benign"])
        self.assertEqual(provenance["partition_sha256"], "digest")


@unittest.skipUnless(ML, "Install training extra")
class TinyTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_refusal_filter_is_strict_and_target_tokens_preserve_chat_format(self):
        import torch
        from types import SimpleNamespace
        model = MagicMock(device="cpu")
        tok = SimpleNamespace(im_end_id=9, eod_id=8, pad_token_id=0)
        model.generate.return_value = torch.tensor([[1, 2, 3, 7, 9]])
        with patch.object(training_data, "format_chat", return_value={"input_ids": [1, 2, 3]}), \
             patch.object(training_data, "next_token_statistics", return_value=[
                 {"refusal_log_probability": -.5, "nonrefusal_log_probability": -.5},
             ]):
            self.assertIsNone(training_data.base_target(model, tok, "p", harmful=True, max_target=4))
            model.generate.assert_not_called()
        with patch.object(training_data, "format_chat", return_value={"input_ids": [1, 2, 3]}), \
             patch.object(training_data, "next_token_statistics", return_value=[
                 {"refusal_log_probability": -.1, "nonrefusal_log_probability": -2},
             ]):
            self.assertEqual(training_data.base_target(model, tok, "p", harmful=True, max_target=4),
                             ([1, 2, 3], [7, 9]))

    def test_tiny_original_qwen_training_merge_reload_and_published_collection(self):
        from test_activations import MemoryS3, tiny_qwen
        from cs182_project.activations import activation_batches, prepare_inputs
        from cs182_project.model_artifacts import download_model
        from cs182_project.qwen_model import format_chat, load_model, load_tokenizer

        model, tokenizer = tiny_qwen()
        s3 = MemoryS3()
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "base"
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            s3.snapshot("models", "base", {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()},
                        model_format="legacy_qwen")
            def target(model, tok, prompt, **kwargs):
                return list(format_chat(tok, prompt)["input_ids"]), [40, tok.im_end_id]
            with patch.object(training, "training_prompts", return_value=(["harm"], ["benign"], {})), \
                 patch.object(training, "base_target", side_effect=target):
                summary = training.train(
                    s3, model_s3_uri="s3://models/base", dataset_s3_uri="s3://datasets/pinned",
                    stats_s3_uri="s3://stats/runs", model_output_s3_uri="s3://models/trained",
                    run_name="tiny", output_dir=Path(temporary) / "runs", device="cpu",
                    overrides=dict(steps=3, batch_per_class=1, k=1, lora_r=2, lora_alpha=4,
                                   warmup_steps=0, ramp_steps=1, grad_checkpointing=True, checkpoint_every=1),
                )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["statistics_delivery"], "complete")
            directory = Path(temporary, "runs", summary["run_id"])
            records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in records], [0, 2])
            for record in records:
                self.assertAlmostEqual(record["loss"], record["L_refuse"] + record["alpha"] * record["L_retain"] + record["lambda"] * record["L_obf"], places=4)
            exported = download_model(s3, summary["model_s3_uri"], Path(temporary) / "download")
            reloaded = load_model(exported, "cpu")
            tok = load_tokenizer(exported)
            prompt = dict(source_row_index=0, prompt="hello", prompt_harm_label="harmful", harmful=True, adversarial=False)
            prepared = prepare_inputs(tok, [prompt], 2048, reloaded.config.max_position_embeddings)
            batches = list(activation_batches(reloaded, tok, prepared, batch_size=1, device="cpu"))
            self.assertEqual(len(batches), 1)
            self.assertTrue((directory / "step1" / "adapter_config.json").exists())

            # Statistics failures must not turn completed optimization into a
            # failed run, and fresh credentials must recover the same queue.
            from cs182_project.training_stats import TrainingStats
            original_put = s3.put_object
            def offline_stats(**kwargs):
                if kwargs["Bucket"] == "stats":
                    raise PermissionError("expired credentials")
                return original_put(**kwargs)
            s3.put_object = offline_stats
            with patch.object(training, "training_prompts", return_value=(["harm"], ["benign"], {})), \
                 patch.object(training, "base_target", side_effect=target), \
                 self.assertLogs("cs182_project.training_stats", level="WARNING"):
                offline = training.train(
                    s3, model_s3_uri="s3://models/base", dataset_s3_uri="s3://datasets/pinned",
                    stats_s3_uri="s3://stats/runs", model_output_s3_uri="s3://models/trained",
                    run_name="tiny", output_dir=Path(temporary) / "runs", device="cpu",
                    overrides=dict(steps=1, batch_per_class=1, k=1, grad_checkpointing=False,
                                   export_merged=False),
                )
            self.assertNotEqual(offline["run_id"], summary["run_id"])
            self.assertEqual(offline["status"], "completed")
            self.assertEqual(offline["statistics_delivery"], "pending")
            s3.put_object = original_put
            offline_dir = Path(temporary, "runs", offline["run_id"])
            self.assertEqual(TrainingStats(s3, offline_dir).flush(), [])
            with patch.object(training, "training_prompts", return_value=([], [], {})), \
                 self.assertRaisesRegex(ValueError, "eligible examples per class"):
                training.train(
                    s3, model_s3_uri="s3://models/base", dataset_s3_uri="s3://datasets/pinned",
                    stats_s3_uri="s3://stats/runs", model_output_s3_uri="s3://models/trained",
                    run_name="empty", output_dir=Path(temporary) / "runs", device="cpu",
                    overrides=dict(steps=1, batch_per_class=1, k=1),
                )
            failed = next((Path(temporary) / "runs").glob("empty-*/summary.json"))
            self.assertEqual(json.loads(failed.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
