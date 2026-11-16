"""Offline checks for Modal adapters, credentials, and the persistent cache."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cs182_project import model_artifacts as artifacts
from cs182_project import dataset_artifacts as datasets
from test_activations import MemoryS3, dataset_snapshot
from test_model_scripts import FakeS3

MODAL_DIR = Path(__file__).resolve().parents[1] / "modal"
# Load the cache adapter by file so these tests also work in the existing ML
# image with only infra mounted at /test-infra and no Modal SDK installed.
spec = importlib.util.spec_from_file_location("modal_cache_tests", MODAL_DIR / "_cache.py")
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
MODAL_AVAILABLE = importlib.util.find_spec("modal") is not None
if MODAL_AVAILABLE:
    from infra.modal import _runner as runner, collect_activations, infer


class SnapshotCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backing = self.root / "volume"
        self.uri = "s3://models/snapshot"
        self.s3 = FakeS3()
        self.working = self.root / "first"
        self.directory = artifacts.download_model(self.s3, self.uri, self.working / "models")
        self.relative = self.directory.relative_to(self.working)

    def adapter(self, working):
        return cache_module.SnapshotCache(self.backing, working, [("models", self.uri)])

    def test_publish_restore_reuses_downloads_and_does_not_copy_locks(self):
        writes = []
        original = cache_module._atomic_copy

        def record(source, destination):
            writes.append(destination.name)
            return original(source, destination)

        with patch.object(cache_module, "_atomic_copy", side_effect=record):
            self.adapter(self.working).publish()
        self.assertEqual(writes[-1], artifacts.MANIFEST)
        self.assertFalse(list(self.backing.rglob(".download.lock")))
        second = self.root / "second"
        self.adapter(second).restore()
        directory = artifacts.download_model(self.s3, self.uri, second / "models")
        self.assertEqual(len(self.s3.downloads), 4)
        self.assertEqual((directory / "model.safetensors").read_bytes(), b"fixture weights")

    def test_corrupt_or_missing_cached_files_are_repaired_from_s3(self):
        self.adapter(self.working).publish()
        backing = self.backing / self.relative
        (backing / "model.safetensors").write_bytes(b"broken")
        (backing / "tokenizer.json").unlink()
        second = self.root / "second"
        self.adapter(second).restore()
        artifacts.download_model(self.s3, self.uri, second / "models")
        self.assertEqual(len(self.s3.downloads), 6)
        self.adapter(second).publish()
        self.assertEqual((backing / "model.safetensors").read_bytes(), b"fixture weights")

    def test_partial_snapshot_is_not_published_and_orphans_are_ignored(self):
        (self.directory / "model.safetensors").write_bytes(b"broken")
        self.adapter(self.working).publish()
        self.assertFalse((self.backing / self.relative / artifacts.MANIFEST).exists())
        orphan = self.backing / self.relative / "config.json"
        orphan.parent.mkdir(parents=True)
        orphan.write_text("orphan")
        second = self.root / "second"
        self.adapter(second).restore()
        artifacts.download_model(self.s3, self.uri, second / "models")
        self.assertEqual(len(self.s3.downloads), 8)

    def test_interrupted_publication_cannot_publish_manifest(self):
        original = cache_module._atomic_copy
        calls = 0

        def interrupt(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("interrupted copy")
            return original(source, destination)

        with patch.object(cache_module, "_atomic_copy", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.adapter(self.working).publish()
        self.assertFalse((self.backing / self.relative / artifacts.MANIFEST).exists())

    def test_concurrent_publication_of_immutable_snapshot(self):
        roots = [self.root / str(index) for index in range(3)]
        for working in roots:
            artifacts.download_model(self.s3, self.uri, working / "models")
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda working: self.adapter(working).publish(), roots))
        restored = self.root / "restored"
        self.adapter(restored).restore()
        count = len(self.s3.downloads)
        artifacts.download_model(self.s3, self.uri, restored / "models")
        self.assertEqual(len(self.s3.downloads), count)
        self.assertFalse(list(self.backing.rglob(".cache-*")))

    def test_s3_manifest_remains_authoritative(self):
        self.adapter(self.working).publish()
        second = self.root / "second"
        self.adapter(second).restore()
        self.s3.manifest["revision"] = "changed"
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            artifacts.download_model(self.s3, self.uri, second / "models")

    def test_model_and_dataset_caches_are_separate_and_both_reusable(self):
        s3 = MemoryS3()
        dataset_snapshot(s3)
        dataset_uri = "s3://datasets/dataset"
        datasets.download_dataset(s3, dataset_uri, self.working / "datasets")
        snapshots = [("models", self.uri), ("datasets", dataset_uri)]
        cache_module.SnapshotCache(self.backing, self.working, snapshots).publish()
        second = self.root / "second"
        cache_module.SnapshotCache(self.backing, second, snapshots).restore()
        datasets.download_dataset(s3, dataset_uri, second / "datasets")
        artifacts.download_model(self.s3, self.uri, second / "models")
        self.assertEqual(len(s3.downloads), 3)
        self.assertEqual(len(self.s3.downloads), 4)

    def test_invalid_cached_manifest_is_ignored_without_following_unsafe_paths(self):
        self.adapter(self.working).publish()
        marker = self.backing / self.relative / artifacts.MANIFEST
        manifest = json.loads(marker.read_bytes())
        manifest["files"][0]["path"] = "../../outside"
        marker.write_text(json.dumps(manifest))
        with self.assertLogs(cache_module.logger, level="WARNING"):
            self.adapter(self.root / "second").restore()
        self.assertFalse((self.root / "second").exists())


@unittest.skipUnless(MODAL_AVAILABLE, "Install the modal extra to test Modal orchestration")
class ModalAdapterTest(unittest.TestCase):
    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/runtime:dev-test"

    def session(self, token="temporary-token"):
        session = MagicMock()
        session.region_name = "us-west-2"
        session.get_credentials.return_value.get_frozen_credentials.return_value = SimpleNamespace(
            access_key="test-access", secret_key="test-secret", token=token,
        )
        return session

    def test_forwards_frozen_session_credentials_without_profile_or_extra_environment(self):
        session = self.session()
        values = runner.aws_environment(session)
        self.assertEqual(values, dict(
            AWS_ACCESS_KEY_ID="test-access", AWS_SECRET_ACCESS_KEY="test-secret",
            AWS_SESSION_TOKEN="temporary-token", AWS_REGION="us-west-2", AWS_DEFAULT_REGION="us-west-2",
        ))
        session.get_credentials.return_value.get_frozen_credentials.assert_called_once_with()
        self.assertNotIn("AWS_SESSION_TOKEN", runner.aws_environment(self.session(token=None)))
        session.get_credentials.return_value = None
        with self.assertRaisesRegex(ValueError, "No AWS credentials"):
            runner.aws_environment(session)

    def test_registry_uses_its_own_region_and_only_modal_wrapper_source(self):
        values = runner.aws_environment(self.session())
        image = MagicMock()
        image.entrypoint.return_value = image
        image.run_commands.return_value = image
        image.env.return_value = image
        image.add_local_python_source.return_value = image
        with patch.object(runner.modal.Image, "from_registry", return_value=image) as registry, \
                patch.object(runner.modal.Secret, "from_dict", return_value="secret") as secret:
            self.assertIs(runner.runtime_image(self.image, values), image)
        self.assertEqual(secret.call_args.args[0]["AWS_REGION"], "us-east-1")
        self.assertEqual(secret.call_args.args[0]["AWS_SESSION_TOKEN"], "temporary-token")
        registry.assert_called_once_with(self.image, secret="secret")
        image.entrypoint.assert_called_once_with([])
        image.run_commands.assert_called_once_with("/app/.venv/bin/python -m ensurepip --upgrade")
        image.add_local_python_source.assert_called_once_with("infra.modal")
        for reference in ["", "runtime:latest", self.image.split(":")[0], "https://example.com/image"]:
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "CS182_MODAL_IMAGE"):
                runner.runtime_image(reference, values)

    def test_sdk_constructs_image_and_function_without_resolving_cloud_objects(self):
        class StopBeforeCloud(Exception):
            pass

        with patch.dict(os.environ, {"CS182_MODAL_IMAGE": self.image}), \
                patch("boto3.Session", return_value=self.session()), \
                patch.object(runner.modal.App, "run", side_effect=StopBeforeCloud):
            with self.assertRaises(StopBeforeCloud):
                runner.run_task("infer", {"device": "auto"})

    def test_worker_configuration_and_failures(self):
        for device in ["cpu", "cuda", "auto"]:
            with self.subTest(device=device):
                app, worker = MagicMock(), MagicMock()
                app.function.return_value.return_value = worker
                app.run.return_value = nullcontext()
                worker.remote.return_value = "response"
                options = {"device": device}
                with patch.dict(os.environ, {"CS182_MODAL_IMAGE": self.image}), \
                        patch("boto3.Session", return_value=self.session()), \
                        patch.object(runner, "runtime_image", return_value="image"), \
                        patch.object(runner.modal, "App", return_value=app), \
                        patch.object(runner.modal.Volume, "from_name", return_value="volume"), \
                        patch.object(runner.modal, "enable_output", return_value=nullcontext()):
                    self.assertEqual(runner.run_task("infer", options, gpu="L4", timeout=123), "response")
                config = app.function.call_args.kwargs
                self.assertEqual(config["gpu"], None if device == "cpu" else "L4")
                self.assertEqual(config["timeout"], 123)
                self.assertEqual(config["memory"], 16384)
                self.assertEqual(config["cpu"], 2)
                self.assertEqual(config["retries"], 0)
                self.assertFalse(config["include_source"])
                worker.remote.assert_called_once_with("infer", options)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CS182_MODAL_IMAGE"):
                runner.run_task("infer", {"device": "auto"})
        with self.assertRaisesRegex(ValueError, "timeout"):
            runner.run_task("infer", {"device": "auto"}, timeout=0)

    def test_entrypoints_forward_options_and_print_results_once(self):
        with patch.object(infer, "run_task", return_value="answer") as run, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            infer.main("s3://models/snapshot", "hello", gpu="L4")
        self.assertEqual(stdout.getvalue(), "answer\n")
        self.assertEqual(run.call_args.args, ("infer", dict(
            model_s3_uri="s3://models/snapshot", prompt="hello", max_new_tokens=128, device="auto",
        )))
        self.assertEqual(run.call_args.kwargs, {"gpu": "L4", "timeout": 3600})
        with patch.object(collect_activations, "run_task", return_value="s3://results/run/manifest.json") as run, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            collect_activations.main("s3://models/snapshot", "s3://datasets/snapshot", "s3://results/runs", 5,
                                     split="test", seed=7, batch_size=2, max_input_tokens=512, device="cuda")
        self.assertEqual(stdout.getvalue(), "s3://results/run/manifest.json\n")
        self.assertEqual(run.call_args.args, ("collect_activations", dict(
            model_s3_uri="s3://models/snapshot", dataset_s3_uri="s3://datasets/snapshot",
            output_s3_uri="s3://results/runs", sample_size=5, split="test", seed=7,
            batch_size=2, max_input_tokens=512, device="cuda",
        )))

    def test_invalid_inputs_do_not_start_remote_work_and_remote_errors_propagate(self):
        with patch.object(infer, "run_task") as run:
            for kwargs in [{"prompt": " "}, {"max_new_tokens": 0}, {"device": "invalid"}]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    infer.main(**(dict(model_s3_uri="s3://models/snapshot", prompt="hello") | kwargs))
            run.assert_not_called()
        options = dict(model_s3_uri="s3://models/snapshot", dataset_s3_uri="s3://datasets/snapshot",
                       output_s3_uri="s3://results/runs", sample_size=2)
        with patch.object(collect_activations, "run_task") as run:
            for kwargs in [{"sample_size": 1}, {"split": "invalid"}, {"output_s3_uri": "invalid"},
                           {"batch_size": 0}, {"max_input_tokens": 0}, {"device": "invalid"}]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    collect_activations.main(**(options | kwargs))
            run.assert_not_called()
        with patch.object(infer, "run_task", side_effect=RuntimeError("remote failure")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaisesRegex(RuntimeError, "remote failure"):
                infer.main("s3://models/snapshot", "hello")
            self.assertEqual(stdout.getvalue(), "")

    def test_cli_help_loads_without_image_or_credentials(self):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("AWS_", "MODAL_TOKEN_", "CS182_MODAL_"))}
        environment["AWS_EC2_METADATA_DISABLED"] = "true"
        for script in ["infer.py", "collect_activations.py"]:
            result = subprocess.run([sys.executable, "-m", "modal", "run", str(MODAL_DIR / script), "--help"],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--model-s3-uri", result.stdout)
            self.assertIn("--gpu", result.stdout)

    def test_remote_worker_commits_cache_on_success_and_failure_without_printing_result(self):
        options = dict(model_s3_uri="s3://models/snapshot", prompt="hello", max_new_tokens=10, device="cuda")
        for failure in [False, True]:
            with self.subTest(failure=failure), \
                    patch("boto3.client", return_value="s3"), \
                    patch("infra.modal._cache.SnapshotCache") as cache, \
                    patch.object(runner.modal.Volume, "from_name") as volume, \
                    patch("cs182_project.model_artifacts.download_model", return_value="/model"), \
                    patch("cs182_project.inference.generate", return_value="answer") as generate, \
                    patch.object(runner.logging, "basicConfig"), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout:
                if failure:
                    generate.side_effect = RuntimeError("GPU failed")
                    with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                        runner.execute_task("infer", options)
                else:
                    self.assertEqual(runner.execute_task("infer", options), "answer")
                cache.return_value.restore.assert_called_once_with()
                cache.return_value.publish.assert_called_once_with()
                volume.return_value.commit.assert_called_once_with()
                generate.assert_called_once_with("/model", "hello", 10, "cuda")
                self.assertEqual(stdout.getvalue(), "")

    def test_remote_collection_preserves_result_when_cache_publication_fails(self):
        options = dict(model_s3_uri="s3://models/snapshot", dataset_s3_uri="s3://datasets/snapshot",
                       output_s3_uri="s3://results/runs", sample_size=4, device="auto")
        with patch("boto3.client", return_value="s3"), \
                patch("infra.modal._cache.SnapshotCache") as cache, \
                patch.object(runner.modal.Volume, "from_name") as volume, \
                patch.object(runner.logging, "basicConfig"), \
                patch("cs182_project.activations.collect", return_value="s3://results/run/manifest.json") as collect:
            cache.return_value.publish.side_effect = OSError("Volume unavailable")
            with self.assertLogs(runner.logger, level="WARNING"):
                self.assertEqual(runner.execute_task("collect_activations", options),
                                 "s3://results/run/manifest.json")
            self.assertEqual(collect.call_args.args, ("s3",))
            self.assertEqual({key: value for key, value in collect.call_args.kwargs.items() if key != "cache_dir"},
                             options)
            self.assertEqual(cache.call_args.args[2], [("models", options["model_s3_uri"]),
                                                      ("datasets", options["dataset_s3_uri"])])
            volume.return_value.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
