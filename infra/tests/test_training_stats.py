"""Offline durability, periodic publication and recovery checks."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from cs182_project.training_stats import TrainingStats, write_json


class StatisticsTest(unittest.TestCase):
    def make_stats(self, directory, s3=None, **kwargs):
        write_json(Path(directory) / "run.json", {
            "run_id": "test-id", "stats_s3_uri": "s3://stats/runs/test-id",
        })
        return TrainingStats(s3 or MagicMock(), directory, **kwargs)

    def test_loss_records_are_sealed_once_and_flush_includes_final_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = self.make_stats(directory)
            losses = dict(loss=2.5, L_refuse=1, L_retain=2, L_obf=1, alpha=.5, **{"lambda": .5})
            stats.record(0, losses)
            self.assertEqual(stats.flush(), [])
            stats.record(20, losses)
            stats.record(21, losses)
            self.assertEqual(stats.close(), [])
            batches = sorted(Path(directory, "batches").glob("*.jsonl"))
            self.assertEqual([p.name for p in batches], [
                "000000000000-000000000001.jsonl", "000000000001-000000000003.jsonl",
            ])
            records = [json.loads(line) for p in batches for line in p.read_text().splitlines()]
            self.assertEqual([r["step"] for r in records], [0, 20, 21])
            self.assertTrue(all(r["run_id"] == "test-id" and r["timestamp"] for r in records))
            for record in records:
                self.assertEqual(record["loss"], record["L_refuse"] + record["alpha"] * record["L_retain"] + record["lambda"] * record["L_obf"])
            stats.flush()
            uploads = [c for c in stats.s3.put_object.call_args_list if c.kwargs["Key"].endswith(".jsonl")]
            self.assertEqual(len(uploads), 2)

    def test_ambiguous_put_retries_same_bytes_after_reopening_queue(self):
        s3 = MagicMock()
        attempted = []
        fail = True

        def put(**kwargs):
            if kwargs["Key"].endswith(".jsonl"):
                attempted.append((kwargs["Key"], kwargs["Body"]))
                if fail:
                    raise OSError("response lost after upload")

        s3.put_object.side_effect = put
        with tempfile.TemporaryDirectory() as directory:
            stats = self.make_stats(directory, s3)
            stats.record(0, {"loss": 1})
            with self.assertLogs("cs182_project.training_stats", level="WARNING"):
                self.assertEqual(len(stats.flush()), 1)
            fail = False
            recovered = TrainingStats(s3, directory)
            self.assertEqual(recovered.flush(), [])
            self.assertEqual(attempted[0], attempted[1])
            self.assertEqual(len(list(Path(directory, "batches").glob("*.jsonl"))), 1)

    def test_timer_uploads_while_training_is_idle_and_close_flushes(self):
        ready = threading.Event()
        s3 = MagicMock()
        s3.put_object.side_effect = lambda **kw: ready.set() if kw["Key"].endswith(".jsonl") else None
        with tempfile.TemporaryDirectory() as directory:
            stats = self.make_stats(directory, s3, flush_seconds=.02)
            stats.record(0, {"loss": 1})
            stats.start()
            try:
                self.assertTrue(ready.wait(3))
            finally:
                stats.close()
            self.assertFalse(stats._thread.is_alive())

    def test_metadata_failures_are_not_reported_as_delivered(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = self.make_stats(directory)
            stats.s3.put_object.side_effect = PermissionError("AccessDenied")
            with self.assertLogs("cs182_project.training_stats", level="WARNING"):
                self.assertIn(str(Path(directory) / "run.json"), stats.flush())

    def test_nonfinite_records_are_rejected_without_poisoning_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = self.make_stats(directory)
            with self.assertRaises(ValueError):
                stats.record(0, {"loss": float("nan")})
            self.assertEqual(stats.journal.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
