"""Offline recovery, provenance, and interrupted-checkpoint regression tests."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

INFRA = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state = load("experiment_state_tests", INFRA / "modal/_experiment.py")
cli = load("experiment_resume_cli_tests", INFRA / "scripts/run_qwen_experiment.py")
MODAL_AVAILABLE = importlib.util.find_spec("modal") is not None
if MODAL_AVAILABLE:
    from infra.modal import experiment_stage as worker_cli
    from infra.modal import _experiment_worker as stage_worker


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.bodies = []
        self.writes = []
        self.exceptions = SimpleNamespace(NoSuchKey=type("NoSuchKey", (Exception,), {}))

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        data = self.objects[Bucket, Key]
        body = io.BytesIO(data)
        self.bodies.append(body)
        return {"Body": body, "ETag": hashlib.sha256(data).hexdigest()}

    def put_object(self, Bucket, Key, Body, **kwargs):
        old = self.objects.get((Bucket, Key))
        if kwargs.get("IfNoneMatch") == "*" and old is not None:
            raise RuntimeError("Conditional write conflict")
        if "IfMatch" in kwargs and (old is None or kwargs["IfMatch"] != hashlib.sha256(old).hexdigest()):
            raise RuntimeError("Conditional write conflict")
        self.objects[Bucket, Key] = Body
        self.writes.append((Bucket, Key, kwargs))
        return {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey(Key)
        return {"ContentLength": len(self.objects[Bucket, Key])}


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.s3 = MemoryS3()
        self.root = "s3://analyses/runs/experiments/experiment/stage"
        self.uri = "s3://activations/runs/run/manifest.json"
        state.put_json(self.s3, self.uri, {"example": "completed output"})
        self.request = {"fingerprint": "inputs", "task": "collect_activations"}
        self.run = Mock(return_value=self.uri)
        self.wait = Mock(return_value=self.uri)
        validate = patch.object(state, "validate_result", side_effect=lambda s3, request, uri: uri)
        validate.start()
        self.addCleanup(validate.stop)

    def resolve(self, **kwargs):
        return state.resolve_stage(self.s3, self.root, self.request, self.run, self.wait, **kwargs)

    def attempt(self):
        head, _ = state.read_json(self.s3, self.root + "/stage.json")
        return self.root + "/" + head["attempt"]

    def test_claim_is_written_before_submission_and_finished_result_is_reused(self):
        def run(attempt):
            self.assertEqual(self.attempt(), attempt)
            state.put_json(self.s3, attempt + "/complete.json", state.completion_record(self.s3, self.request, self.uri))
            return self.uri
        self.run.side_effect = run
        self.assertEqual(self.resolve(), self.uri)
        self.assertEqual(self.resolve(), self.uri)
        self.run.assert_called_once()
        self.wait.assert_not_called()
        self.assertTrue(all(body.closed for body in self.s3.bodies))

    def test_detached_completion_recovers_without_local_acknowledgement(self):
        self.run.side_effect = OSError("workstation disconnected")
        with self.assertRaises(OSError):
            self.resolve()
        state.put_json(self.s3, self.attempt() + "/complete.json", state.completion_record(self.s3, self.request, self.uri))
        self.assertEqual(self.resolve(), self.uri)
        self.run.assert_called_once()
        self.wait.assert_not_called()

    def test_running_job_is_awaited_instead_of_resubmitted(self):
        self.run.side_effect = OSError("lost connection")
        with self.assertRaises(OSError):
            self.resolve()
        state.put_json(self.s3, self.attempt() + "/submitted.json", {"call_id": "fc-existing"})
        self.assertEqual(self.resolve(), self.uri)
        self.wait.assert_called_once_with("fc-existing", self.attempt())
        self.run.assert_called_once()

    def test_remote_started_record_heals_lost_spawn_acknowledgement(self):
        self.run.side_effect = OSError("lost spawn acknowledgement")
        with self.assertRaises(OSError):
            self.resolve()
        state.put_json(self.s3, self.attempt() + "/started.json", {"call_id": "fc-started"})
        self.resolve()
        self.wait.assert_called_once_with("fc-started", self.attempt())
        self.run.assert_called_once()

    def test_uncertain_submission_is_never_automatically_duplicated(self):
        self.run.side_effect = OSError("lost spawn acknowledgement")
        with self.assertRaises(OSError):
            self.resolve()
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            self.resolve()
        self.run.assert_called_once()

    def test_confirmed_failure_is_retried_using_a_new_conditional_attempt(self):
        self.run.side_effect = OSError("failed")
        with self.assertRaises(OSError):
            self.resolve()
        old = self.attempt()
        state.put_json(self.s3, old + "/failed.json", {"error": "job failed"})
        self.run.side_effect = None
        self.assertEqual(self.resolve(), self.uri)
        self.assertNotEqual(self.attempt(), old)
        self.assertTrue(any("IfMatch" in conditions for _, _, conditions in self.s3.writes))

    def test_adoption_can_heal_an_uncertain_attempt_without_starting_work(self):
        self.run.side_effect = OSError("lost acknowledgement")
        with self.assertRaises(OSError):
            self.resolve()
        self.assertEqual(self.resolve(adopted=self.uri), self.uri)
        self.assertEqual(self.resolve(), self.uri)
        self.run.assert_called_once()

    def test_changed_request_or_completed_manifest_is_rejected(self):
        self.resolve(adopted=self.uri)
        self.request = {**self.request, "fingerprint": "changed image"}
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            self.resolve()
        self.request["fingerprint"] = "inputs"
        state.put_json(self.s3, self.uri, {"example": "mutated"})
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.resolve()

    def test_permission_errors_and_conditional_conflicts_do_not_launch_work(self):
        with patch.object(self.s3, "get_object", side_effect=PermissionError("AccessDenied")):
            with self.assertRaises(PermissionError):
                self.resolve()
        with patch.object(self.s3, "put_object", side_effect=RuntimeError("Conditional write conflict")):
            with self.assertRaisesRegex(RuntimeError, "conflict"):
                self.resolve()
        self.run.assert_not_called()


class ArtifactValidationTest(unittest.TestCase):
    def setUp(self):
        self.s3 = MemoryS3()
        self.uri = "s3://activations/runs/run/manifest.json"
        self.s3.objects["activations", "runs/run/part-00000.parquet"] = b"shard"
        self.options = dict(model_s3_uri="s3://models/model", dataset_s3_uri="s3://datasets/data",
                            sample_size=10, seed=0, batch_size=4, max_input_tokens=2048,
                            device="auto", split="train", partition_s3_uri="s3://datasets/partition.json")
        self.inputs = {}
        for uri in ("s3://models/model/artifact-manifest.json", "s3://datasets/data/artifact-manifest.json",
                    "s3://datasets/partition.json"):
            state.put_json(self.s3, uri, {"input": uri})
            self.inputs[uri] = state.document(self.s3, uri)[1]
        self.manifest = dict(schema_version=2, status="complete", run_id="run", row_count=10,
                             num_layers=2, hidden_size=8, sample_size=10, seed=0, batch_size=4,
                             max_input_tokens=2048, device="auto",
                             model=dict(s3_uri=self.options["model_s3_uri"],
                                        manifest_sha256=self.inputs["s3://models/model/artifact-manifest.json"]),
                             dataset=dict(s3_uri=self.options["dataset_s3_uri"], split="train",
                                          manifest_sha256=self.inputs["s3://datasets/data/artifact-manifest.json"],
                                          partition=dict(s3_uri="s3://datasets/partition.json",
                                                         sha256=self.inputs["s3://datasets/partition.json"])),
                             shards=[dict(key="runs/run/part-00000.parquet", size=5, rows=10, sha256="a" * 64)])
        self.request = dict(task="collect_activations", options=self.options, inputs=self.inputs)

    def validate(self):
        state.put_json(self.s3, self.uri, self.manifest)
        return state.validate_result(self.s3, self.request, self.uri)

    def test_completed_activation_is_bound_to_exact_inputs_and_settings(self):
        self.assertEqual(self.validate(), self.uri)
        for key, wrong in [("seed", 1), ("sample_size", 11), ("batch_size", 8), ("max_input_tokens", 1024)]:
            with self.subTest(key=key), patch.dict(self.manifest, {key: wrong}):
                with self.assertRaises(ValueError):
                    self.validate()
        self.manifest["dataset"]["partition"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "partition differs"):
            self.validate()

    def test_incomplete_or_missing_artifacts_are_never_reused(self):
        self.manifest["status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "completed"):
            self.validate()
        self.manifest["status"] = "complete"
        self.s3.objects.pop(("activations", "runs/run/part-00000.parquet"))
        with self.assertRaises(self.s3.exceptions.NoSuchKey):
            self.validate()

    def test_wrong_size_and_changed_upstream_manifests_are_rejected(self):
        self.s3.objects["activations", "runs/run/part-00000.parquet"] = b"broken shard"
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            self.validate()
        state.put_json(self.s3, "s3://datasets/data/artifact-manifest.json", {"input": "changed"})
        with self.assertRaisesRegex(ValueError, "Upstream manifest changed"):
            self.validate()

    def test_pinned_document_cannot_silently_change(self):
        pins = {}
        uri = "s3://datasets/partition.json"
        state.document(self.s3, uri, pins)
        state.put_json(self.s3, uri, {"new": "partition"})
        with self.assertRaisesRegex(ValueError, "Pinned input changed"):
            state.document(self.s3, uri, pins)


class CheckpointTest(unittest.TestCase):
    def test_interrupted_atomic_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            state.atomic_save(path, {"old": "valid"})
            with patch.object(state.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    state.atomic_save(path, {"new": "incomplete"})
            self.assertEqual(json.loads(path.read_text()), {"old": "valid"})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def invoke(self, args, execute):
        with patch("sys.argv", ["experiment", *args]), patch.object(cli, "execute", side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return cli.main()

    def fake_execute(self, commands, fail_stage=None):
        def execute(command, *, env, **kwargs):
            commands.append((command, dict(env)))
            output = ""
            if "get-caller-identity" in command:
                output = '{"Account": "123456789012"}'
            elif "infra/scripts/discover_modal.py" in command:
                output = "image:pinned\ts3://models/original\ts3://datasets/original\ts3://activations/runs\ts3://analyses/runs\ts3://stats/runs\n"
            elif "--stage" in command:
                name = command[command.index("--stage") + 1]
                if name == fail_stage:
                    raise subprocess.CalledProcessError(1, command)
                if "activations" in name:
                    output = f"s3://activations/runs/{name}/manifest.json\n"
                elif "analysis" in name:
                    output = f"s3://analyses/runs/{name}/analysis/report.html\n"
                else:
                    output = "s3://models/edited\n"
            return subprocess.CompletedProcess(command, 0, stdout=output)
        return execute

    def test_resume_restores_pinned_environment_and_never_discovers_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            commands = []
            self.assertEqual(self.invoke(["--results-file", str(path)], self.fake_execute(commands, "original_analysis")), 1)
            checkpoint = json.loads(path.read_text())
            self.assertIn("original_activations", checkpoint)
            self.assertNotIn("original_analysis", checkpoint)
            commands.clear()
            self.assertEqual(self.invoke(["--resume", str(path)], self.fake_execute(commands)), 0)
            self.assertFalse(any("infra/scripts/discover_modal.py" in command for command, _ in commands))
            self.assertTrue(all(env["CS182_MODAL_IMAGE"] == "image:pinned"
                                for command, env in commands if "--stage" in command))
            self.assertEqual(json.loads(path.read_text())["experiment_id"], checkpoint["experiment_id"])

    def test_conflicting_resume_options_stop_before_commands_or_checkpoint_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            self.invoke(["--results-file", str(path)], self.fake_execute([]))
            original = path.read_bytes()
            execute = Mock()
            with self.assertRaises(SystemExit):
                self.invoke(["--resume", str(path), "--sample-size", "2000"], execute)
            execute.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_resume_dry_run_does_not_execute_commands_or_write_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            self.invoke(["--results-file", str(path)], self.fake_execute([]))
            original = path.read_bytes()
            with patch("sys.argv", ["experiment", "--resume", str(path), "--dry-run"]), \
                    patch.object(cli.subprocess, "Popen") as popen, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
            popen.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_legacy_summary_is_imported_and_adoption_is_saved_for_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            state.atomic_save(path, dict(original_model="s3://models/original", dataset="s3://datasets/original",
                                         runtime_image="image:legacy", sample_size=1000, sampling_seed=0,
                                         original_activations="s3://activations/runs/original/manifest.json"))
            commands = []
            self.assertEqual(self.invoke(["--resume", str(path), "--adopt-stage",
                                          "original_analysis=s3://analyses/runs/original/analysis/report.html"],
                                         self.fake_execute(commands)), 0)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["schema_version"], 2)
            self.assertEqual(saved["stages"]["original_analysis"]["adopted_uri"],
                             "s3://analyses/runs/original/analysis/report.html")
            self.assertFalse(any("infra/scripts/discover_modal.py" in command for command, _ in commands))


@unittest.skipUnless(MODAL_AVAILABLE, "Install the modal extra")
class StageWorkerTest(unittest.TestCase):
    def setUp(self):
        self.s3 = MemoryS3()
        self.attempt = "s3://analyses/runs/experiments/example/stage/attempt"
        self.uri = "s3://activations/runs/example/manifest.json"
        self.request = dict(task="collect_activations", options={})
        self.record = dict(result_uri=self.uri, manifest_uri=self.uri, manifest_sha256="digest")

    def execute(self, *, error=None, receipt_error=False):
        original_put = self.s3.put_object

        def put(**kwargs):
            if receipt_error and kwargs["Key"].endswith("/complete.json"):
                raise OSError("receipt connection lost")
            return original_put(**kwargs)

        with patch("boto3.client", return_value=self.s3), \
                patch.object(worker_cli.modal, "current_function_call_id", return_value="fc-worker"), \
                patch.object(stage_worker, "execute_task", return_value=self.uri, side_effect=error), \
                patch.object(stage_worker, "completion_record", return_value=self.record), \
                patch.object(self.s3, "put_object", side_effect=put):
            return worker_cli.execute_stage(self.request, self.attempt)

    def test_worker_records_its_own_call_id_and_result_without_local_process(self):
        self.assertEqual(self.execute(), self.uri)
        started, _ = state.read_json(self.s3, self.attempt + "/started.json")
        complete, _ = state.read_json(self.s3, self.attempt + "/complete.json")
        self.assertEqual(started["call_id"], "fc-worker")
        self.assertEqual(complete, self.record)

    def test_failed_worker_records_failure_without_completion(self):
        with self.assertRaisesRegex(ValueError, "compute failed"):
            self.execute(error=ValueError("compute failed"))
        failure, _ = state.read_json(self.s3, self.attempt + "/failed.json")
        self.assertEqual(failure["error"], "compute failed")
        self.assertIsNone(state.read_json(self.s3, self.attempt + "/complete.json", optional=True)[0])

    def test_published_result_still_returns_when_receipt_write_fails(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.execute(receipt_error=True), self.uri)

    def test_reconnection_polls_same_call_and_saves_result(self):
        call = Mock()
        call.get.side_effect = [TimeoutError("still running"), self.uri]
        with patch.object(worker_cli.modal.FunctionCall, "from_id", return_value=call) as from_id, \
                patch.object(worker_cli, "validate_result", return_value=self.uri), \
                patch.object(worker_cli, "completion_record", return_value=self.record), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker_cli.wait_for_call(self.s3, self.request, "fc-existing", self.attempt), self.uri)
        from_id.assert_called_once_with("fc-existing")
        self.assertEqual(call.get.call_count, 2)
        self.assertEqual(state.read_json(self.s3, self.attempt + "/complete.json")[0], self.record)

    def test_remote_timeout_is_confirmed_failure_and_not_infinite_polling(self):
        call = Mock()
        call.get.side_effect = worker_cli.modal.exception.FunctionTimeoutError("timed out")
        with patch.object(worker_cli.modal.FunctionCall, "from_id", return_value=call):
            with self.assertRaisesRegex(RuntimeError, "Resume to retry"):
                worker_cli.wait_for_call(self.s3, self.request, "fc-existing", self.attempt)
        self.assertIn("timed out", state.read_json(self.s3, self.attempt + "/failed.json")[0]["error"])
        call.get.assert_called_once()

    def test_pre_submission_failure_can_retry_but_ambiguous_spawn_cannot(self):
        with patch.object(worker_cli, "create_worker", side_effect=ValueError("image unavailable")):
            with self.assertRaises(ValueError):
                worker_cli.launch(self.s3, self.request, self.attempt, {}, Mock())
        self.assertIsNotNone(state.read_json(self.s3, self.attempt + "/failed.json")[0])
        app, worker = Mock(), Mock()
        app.run.return_value = contextlib.nullcontext()
        worker.spawn.side_effect = OSError("lost acknowledgement")
        other = self.attempt + "-other"
        with patch.object(worker_cli, "create_worker", return_value=(app, worker)), \
                patch.object(worker_cli.modal, "enable_output", return_value=contextlib.nullcontext()):
            with self.assertRaises(OSError):
                worker_cli.launch(self.s3, self.request, other, {}, Mock())
        app.run.assert_called_once_with(detach=True)
        self.assertIsNone(state.read_json(self.s3, other + "/failed.json", optional=True)[0])

    def test_call_id_is_saved_before_waiting(self):
        app, worker, call = Mock(), Mock(), Mock(object_id="fc-submitted")
        app.run.return_value = contextlib.nullcontext()
        worker.spawn.return_value = call
        saved = Mock()

        def wait(*args):
            saved.assert_called_once_with("fc-submitted", self.attempt)
            self.assertEqual(state.read_json(self.s3, self.attempt + "/submitted.json")[0]["call_id"], "fc-submitted")
            return self.uri

        with patch.object(worker_cli, "create_worker", return_value=(app, worker)), \
                patch.object(worker_cli.modal, "enable_output", return_value=contextlib.nullcontext()), \
                patch.object(worker_cli, "wait_for_call", side_effect=wait), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker_cli.launch(self.s3, self.request, self.attempt, {}, saved), self.uri)

    def test_analysis_streams_logs_and_returns_exact_report_uri(self):
        from infra.modal import _runner

        process = Mock(returncode=0)
        process.stdout = io.StringIO("GPU analysis progress\ns3://analyses/run/analysis/report.html\n")
        process.wait.return_value = 0
        with patch.object(subprocess, "Popen", return_value=process), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = _runner.execute_analysis({"run_s3_uri": self.uri}, capture_result=True)
        self.assertEqual(result, "s3://analyses/run/analysis/report.html")
        self.assertIn("GPU analysis progress", output.getvalue())

    def test_worker_serializes_through_stable_package_when_launcher_is_loaded_by_basename(self):
        from modal._serialization import deserialize, serialize
        from types import ModuleType
        import sys

        launcher = load("experiment_stage", INFRA / "modal/experiment_stage.py")
        with patch.dict(sys.modules, {"experiment_stage": launcher}):
            payload = serialize(launcher.execute_stage)
            sys.modules["experiment_stage"] = ModuleType("experiment_stage")
            self.assertIs(deserialize(payload, None), stage_worker.execute_stage)

    def test_completed_stage_entrypoint_skips_compute_and_image_preparation(self):
        fixture = ArtifactValidationTest()
        fixture.setUp()
        uri = fixture.validate()
        request = {**fixture.request, "runtime_image": "image:pinned", "fingerprint": "same-inputs"}
        name = "original_activations"
        root = "s3://analyses/runs/experiments/example/" + name
        state.put_json(fixture.s3, root + "/stage.json", {"fingerprint": "same-inputs", "attempt": "existing"})
        state.put_json(fixture.s3, root + "/existing/complete.json", state.completion_record(fixture.s3, request, uri))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            state.atomic_save(path, dict(experiment_id="example", invocation_id="invocation", region="us-east-1",
                                         runtime_image="image:pinned", environment={"ANALYSIS_URI": "s3://analyses/runs"},
                                         stages={name: {"task": "collect_activations", "options": fixture.options}}))
            with patch("boto3.Session", return_value=SimpleNamespace(client=lambda name: fixture.s3)), \
                    patch.object(worker_cli, "aws_environment", return_value={}), \
                    patch.object(worker_cli, "prepare_request", return_value=request), \
                    patch.object(worker_cli, "launch") as launch, \
                    patch.object(worker_cli, "prepare_runtime_image") as prepare, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                worker_cli.main(str(path), name)
            saved = json.loads(path.read_text())
            self.assertEqual(saved[name], uri)
            self.assertEqual(saved["stages"][name]["status"], "complete")
            self.assertIn(uri, saved["pinned_manifests"])
            self.assertIn(uri, output.getvalue())
            launch.assert_not_called()
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
