"""Offline tests: no AWS account, downloaded weights, or ML libraries required."""

from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
from cs182_project import inference
from cs182_project import activations
from cs182_project import model_artifacts as artifacts
from cs182_project import seeding as seed_model

# Load the CLI only for interface tests; core tests import the installed package.
spec = importlib.util.spec_from_file_location("infer_cli", SCRIPTS / "infer.py")
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


class FakeS3:
    def __init__(self):
        self.files = {
            "config.json": b'{"max_position_embeddings": 32768}',
            "tokenizer_config.json": b"{}",
            "tokenizer.json": b"{}",
            "model.safetensors": b"fixture weights",
        }
        self.manifest = {
            "schema_version": 1,
            "revision": artifacts.MODEL_REVISION,
            "files": [
                {"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
                for name, body in self.files.items()
            ],
        }
        self.downloads = []
        self.corrupt = False
        self.interrupt = False

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(json.dumps(self.manifest).encode())}

    def download_file(self, bucket, key, target):
        name = key.removeprefix("snapshot/")
        self.downloads.append(name)
        Path(target).write_bytes(b"corrupt" if self.corrupt else self.files[name])
        if self.interrupt:
            raise OSError("interrupted transfer")


class ArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.s3 = FakeS3()

    def download(self):
        return artifacts.download_model(self.s3, "s3://models/snapshot", self.temporary.name)

    def test_cache_reuse_and_corruption_repair(self):
        directory = self.download()
        self.assertEqual(len(self.s3.downloads), 4)
        self.assertEqual(self.download(), directory)
        self.assertEqual(len(self.s3.downloads), 4)
        (directory / "model.safetensors").write_bytes(b"broken")
        self.download()
        self.assertEqual(len(self.s3.downloads), 5)
        self.assertEqual((directory / "model.safetensors").read_bytes(), b"fixture weights")

    def test_checksum_failure_does_not_publish_snapshot(self):
        self.s3.corrupt = True
        with self.assertRaisesRegex(ValueError, "Checksum/size mismatch"):
            self.download()
        self.assertEqual(list(Path(self.temporary.name).rglob(".download-*")), [])
        self.assertEqual(list(Path(self.temporary.name).rglob(artifacts.MANIFEST)), [])

    def test_interrupted_download_can_be_retried(self):
        self.s3.interrupt = True
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.download()
        self.s3.interrupt = False
        self.assertTrue((self.download() / artifacts.MANIFEST).exists())

    def test_changed_manifest_is_rejected(self):
        self.download()
        self.s3.manifest["revision"] = "different"
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.download()

    def test_unsafe_paths_and_missing_weights_are_rejected(self):
        for name in ["../outside", "/absolute", "nested/../../outside", "./config.json", "nested\\file"]:
            with self.subTest(name=name):
                self.s3.manifest["files"][0]["path"] = name
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    self.download()
        self.s3.manifest["files"][0]["path"] = "config.json"
        self.s3.manifest["files"].pop()
        with self.assertRaisesRegex(ValueError, "safetensors"):
            self.download()

    def test_s3_error_is_not_hidden(self):
        self.s3.get_object = MagicMock(side_effect=PermissionError("AccessDenied"))
        with self.assertRaisesRegex(PermissionError, "AccessDenied"):
            self.download()

    def test_uri_validation(self):
        self.assertEqual(artifacts.parse_s3_uri("s3://bucket/path/to/model/"), ("bucket", "path/to/model"))
        for uri in ["https://example.com/model", "s3://bucket", "s3:///model", "s3://bucket/model?revision=x"]:
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                artifacts.parse_s3_uri(uri)


class CliTest(unittest.TestCase):
    def test_help_does_not_require_ml_or_aws_dependencies(self):
        for script in ["infer.py", "seed_model.py"]:
            result = subprocess.run([sys.executable, str(SCRIPTS / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_invalid_arguments_fail_before_loading_dependencies(self):
        base = [sys.executable, str(SCRIPTS / "infer.py"), "--model-s3-uri", "s3://bucket/snapshot"]
        for arguments in [["--prompt", " "], ["--prompt", "hello", "--max-new-tokens", "0"]]:
            result = subprocess.run(base + arguments, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("error:", result.stderr)

    def test_runtime_error_has_nonzero_exit_and_message(self):
        with patch.object(sys, "argv", ["infer", "--model-s3-uri", "s3://bucket/snapshot", "--prompt", "hello"]), \
                patch.dict(sys.modules, {"boto3": types.SimpleNamespace(client=lambda _: object())}), \
                patch.object(infer, "download_model", side_effect=PermissionError("AccessDenied")), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(infer.main(), 1)
            self.assertIn("Inference failed: AccessDenied", stderr.getvalue())


class LibraryBoundaryTest(unittest.TestCase):
    def test_installed_imports_and_help_without_optional_dependencies(self):
        # An isolated interpreter outside the checkout must use the installation.
        # Block optional imports even when this test runs in the full ML image.
        code = '''
import importlib.abc
import runpy
import sys

class NoOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "pyarrow", "boto3", "botocore", "huggingface_hub", "numpy", "sklearn", "plotly"}:
            raise ImportError("optional dependency imported: " + fullname)

sys.meta_path.insert(0, NoOptionalDependencies())
from cs182_project import activations, dataset_artifacts, inference, model_artifacts, seeding
from cs182_project import activation_runs, activation_analysis, activation_report
if len(sys.argv) > 1:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
'''
        with tempfile.TemporaryDirectory() as directory:
            for script in [None, "infer.py", "seed_model.py", "seed_dataset.py", "collect_activations.py", "analyze_activations.py"]:
                with self.subTest(script=script):
                    command = [sys.executable, "-I", "-c", code]
                    if script:
                        command += [str(SCRIPTS / script), "--help"]
                    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_library_calls_validate_before_loading_dependencies(self):
        with self.assertRaisesRegex(ValueError, "prompt must not be empty"):
            inference.generate("/unused", " ", 1, "cpu")
        with self.assertRaisesRegex(ValueError, "between 1 and 4096"):
            inference.generate("/unused", "hello", 0, "cpu")
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            seed_model.seed_model(object(), "bucket", revision="main")
        options = dict(model_s3_uri="s3://models/snapshot", dataset_s3_uri="s3://datasets/snapshot",
                       output_s3_uri="s3://activations/runs", sample_size=2)
        for override, message in [({"sample_size": 1}, "sample-size"),
                                  ({"batch_size": 0}, "batch-size"),
                                  ({"split": "invalid"}, "split must"),
                                  ({"device": "invalid"}, "device must"),
                                  ({"output_s3_uri": "https://invalid"}, "S3 location")]:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                activations.collect(object(), **(options | override))


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = False
        self.torch.inference_mode.return_value = nullcontext()
        self.model = MagicMock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.config.max_position_embeddings = 32768
        self.tokenizer = MagicMock()
        self.inputs = MagicMock()
        self.inputs.to.return_value = self.inputs
        self.inputs.keys.return_value = ["input_ids"]
        self.inputs.__getitem__.return_value.shape = [1, 3]
        self.tokenizer.return_value = self.inputs
        self.transformers = types.SimpleNamespace(
            AutoTokenizer=MagicMock(), AutoModelForCausalLM=MagicMock(),
        )
        self.transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        self.transformers.AutoModelForCausalLM.from_pretrained.return_value = self.model
        self.modules = patch.dict(sys.modules, {
            "torch": self.torch,
            "transformers": self.transformers,
            "transformers.models.auto.modeling_auto": self.transformers,
            "transformers.models.auto.tokenization_auto": self.transformers,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_chat_template_response_slicing_and_cpu_fallback(self):
        inference.generate("/snapshot", "hello", 10, "auto")
        self.model.to.assert_called_once_with("cpu")
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True,
        )
        self.model.generate.assert_called_once()
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 10)
        self.model.generate.return_value.__getitem__.assert_called_once_with((0, slice(3, None)))
        self.assertTrue(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["local_files_only"])

    def test_auto_uses_gpu_and_explicit_cuda_fails_without_one(self):
        with self.assertRaisesRegex(ValueError, "CUDA requested"):
            inference.generate("/snapshot", "hello", 10, "cuda")
        self.torch.cuda.is_available.return_value = True
        inference.generate("/snapshot", "hello", 10, "auto")
        self.model.to.assert_called_once_with("cuda")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], self.torch.float16)

    def test_context_overflow_fails_before_generation(self):
        self.model.config.max_position_embeddings = 5
        with self.assertRaisesRegex(ValueError, "context window"):
            inference.generate("/snapshot", "hello", 10, "cpu")
        self.model.generate.assert_not_called()


class SeedingTest(unittest.TestCase):
    def test_model_download_orchestration_is_in_library(self):
        download = MagicMock(return_value="/snapshot")
        client = object()
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=download)}), \
                patch.object(seed_model, "seed", return_value="s3://models/snapshot") as publish:
            result = seed_model.seed_model(client, "models", prefix="/snapshot/", cache_dir="/hf-cache")
        self.assertEqual(result, "s3://models/snapshot")
        download.assert_called_once_with(repo_id=artifacts.MODEL_ID, revision=artifacts.MODEL_REVISION,
                                         cache_dir="/hf-cache")
        publish.assert_called_once_with(client, "/snapshot", "models", "snapshot", artifacts.MODEL_REVISION)

    def test_uploads_all_artifacts_and_publishes_manifest_last(self):
        class MissingObject(Exception):
            response = {"Error": {"Code": "404"}}

        fake_exceptions = types.ModuleType("botocore.exceptions")
        fake_exceptions.ClientError = MissingObject
        s3 = MagicMock()
        s3.head_object.side_effect = MissingObject()
        fixture = FakeS3()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"botocore": types.ModuleType("botocore"), "botocore.exceptions": fake_exceptions}):
            for name, body in fixture.files.items():
                (Path(directory) / name).write_bytes(body)
            (Path(directory) / "LICENSE").write_text("fixture license")
            result = seed_model.seed(s3, directory, "models", "snapshot", artifacts.MODEL_REVISION)
            self.assertEqual(result, "s3://models/snapshot")
            self.assertEqual(s3.upload_file.call_count, 5)
            self.assertEqual(s3.mock_calls[-1][0], "put_object")
            self.assertEqual(s3.put_object.call_args.kwargs["IfNoneMatch"], "*")
            manifest = json.loads(s3.put_object.call_args.kwargs["Body"])
            self.assertEqual(manifest["revision"], artifacts.MODEL_REVISION)
            s3.head_object.side_effect = None
            with self.assertRaisesRegex(ValueError, "already published"):
                seed_model.seed(s3, directory, "models", "snapshot", artifacts.MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
