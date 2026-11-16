"""Offline publication discovery and sourceable shell environment checks."""

from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cs182_project.dataset_artifacts import DATASET_ID, SPLITS
from cs182_project.environment import discover_modal_environment
from cs182_project.model_artifacts import MANIFEST, MODEL_ID, MODEL_REVISION
from test_model_scripts import FakeS3

INFRA = Path(__file__).resolve().parents[1]
HELPER = INFRA / "modal" / "env.sh"
spec = importlib.util.spec_from_file_location("discover_modal_cli", INFRA / "scripts" / "discover_modal.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.region_name = "us-east-1"
        self.sts, self.s3, self.ecr = MagicMock(), MagicMock(), MagicMock()
        clients = {"sts": self.sts, "s3": self.s3, "ecr": self.ecr}
        self.session.client.side_effect = lambda name, **kwargs: clients[name]
        self.sts.get_caller_identity.return_value = {"Account": "123456789012"}
        old, new, incomplete = (datetime(2026, 11, day, tzinfo=timezone.utc) for day in [1, 2, 3])
        self.old_sha, self.new_sha = "f" * 40, "a" * 40
        self.dataset = {
            "schema_version": 1, "dataset_id": DATASET_ID, "revision": self.new_sha, "splits": SPLITS,
            "files": [{"path": name, "size": 1, "sha256": "0" * 64}
                      for name in [*SPLITS.values(), "README.md"]],
        }
        self.model = dict(FakeS3().manifest, model_id=MODEL_ID)
        self.bodies = []

        def get_object(**kwargs):
            manifest = self.dataset if kwargs["Key"].startswith("datasets/") else self.model
            body = io.BytesIO(json.dumps(manifest).encode())
            self.bodies.append(body)
            return {"Body": body}

        self.s3.get_object.side_effect = get_object
        self.s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": f"datasets/wildguardmix/{self.old_sha}/{MANIFEST}", "LastModified": old}]},
            {"Contents": [
                {"Key": f"datasets/wildguardmix/{self.new_sha}/{MANIFEST}", "LastModified": new},
                {"Key": f"datasets/wildguardmix/{'b' * 40}/train/wildguard_train.parquet", "LastModified": incomplete},
                {"Key": f"datasets/wildguardmix/main/{MANIFEST}", "LastModified": incomplete},
            ]},
        ]
        self.ecr.get_paginator.return_value.paginate.return_value = [
            {"imageDetails": [{"imageTags": ["dev-fffffff"], "imagePushedAt": old, "imageDigest": "sha256:old"}]},
            {"imageDetails": [
                {"imageTags": ["z-alias", "dev-0000001"], "imagePushedAt": new, "imageDigest": "sha256:new"},
                {"imageTags": [], "imagePushedAt": incomplete, "imageDigest": "sha256:untagged"},
            ]},
        ]
        self.ecr.describe_repositories.return_value = {
            "repositories": [{"repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/cs182-cddr-runtime"}],
        }

    def test_selects_latest_publications_across_pages_and_ignores_incomplete_uploads(self):
        values = discover_modal_environment(self.session)
        self.assertEqual(values, {
            "CS182_MODAL_IMAGE": "123456789012.dkr.ecr.us-east-1.amazonaws.com/cs182-cddr-runtime:dev-0000001",
            "MODEL_URI": f"s3://cs182-cddr-123456789012-us-east-1-models/models/Qwen1.5-1.8B-Chat/{MODEL_REVISION}",
            "DATASET_URI": f"s3://cs182-cddr-123456789012-us-east-1-datasets/datasets/wildguardmix/{self.new_sha}",
            "OUTPUT_URI": "s3://cs182-cddr-123456789012-us-east-1-activations/runs",
        })
        self.s3.get_paginator.assert_called_once_with("list_objects_v2")
        self.ecr.get_paginator.assert_called_once_with("describe_images")
        self.ecr.get_paginator.return_value.paginate.assert_called_once_with(
            repositoryName="cs182-cddr-runtime", filter={"tagStatus": "TAGGED"},
        )
        self.assertTrue(all(body.closed for body in self.bodies))
        self.assertEqual(self.s3.get_object.call_count, 2)

    def test_project_and_region_overrides_use_current_account(self):
        values = discover_modal_environment(self.session, project="another-project", region="us-west-2")
        self.assertIn("another-project-123456789012-us-west-2-models", values["MODEL_URI"])
        self.assertEqual(values["OUTPUT_URI"], "s3://another-project-123456789012-us-west-2-activations/runs")
        self.assertTrue(all(call.kwargs["region_name"] == "us-west-2" for call in self.session.client.call_args_list))
        self.ecr.describe_repositories.assert_called_once_with(repositoryNames=["another-project-runtime"])

    def test_no_tagged_images_fails(self):
        self.ecr.get_paginator.return_value.paginate.return_value = [{"imageDetails": []}]
        with self.assertRaisesRegex(ValueError, "No tagged runtime images"):
            discover_modal_environment(self.session)

    def test_no_completed_datasets_fails(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{}]
        with self.assertRaisesRegex(ValueError, "No completed WildGuardMix"):
            discover_modal_environment(self.session)

    def test_invalid_latest_manifest_and_identity_mismatch_do_not_fall_back(self):
        self.dataset["revision"] = self.old_sha
        with self.assertRaisesRegex(ValueError, "revision does not match"):
            discover_modal_environment(self.session)
        self.dataset["revision"] = self.new_sha
        self.dataset["splits"] = {}
        with self.assertRaisesRegex(ValueError, "both official splits"):
            discover_modal_environment(self.session)
        self.model["revision"] = self.old_sha
        with self.assertRaisesRegex(ValueError, "pinned model and revision"):
            discover_modal_environment(self.session)

    def test_aws_errors_propagate_and_project_validation_precedes_requests(self):
        self.s3.get_object.side_effect = PermissionError("AccessDenied")
        with self.assertRaisesRegex(PermissionError, "AccessDenied"):
            discover_modal_environment(self.session)
        self.session.client.reset_mock()
        with self.assertRaisesRegex(ValueError, "Project must"):
            discover_modal_environment(self.session, project="../invalid")
        self.session.client.assert_not_called()

    def test_cli_formats_values_and_does_not_print_partial_configuration_on_error(self):
        expected = dict(CS182_MODAL_IMAGE="image:tag", MODEL_URI="model", DATASET_URI="dataset", OUTPUT_URI="output")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover", "--project", "custom"]), \
                patch.object(cli, "discover_modal_environment", return_value=expected) as discover, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(stdout.getvalue(), "image:tag\tmodel\tdataset\toutput\n")
        discover.assert_called_once_with(self.session, project="custom", region=None)
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover"]), \
                patch.object(cli, "discover_modal_environment", side_effect=PermissionError("AccessDenied")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cli.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("AccessDenied", stderr.getvalue())


class ShellEnvironmentTest(unittest.TestCase):
    def source(self, directory, response, exit_code=0):
        root = Path(directory)
        uv = root / "uv"
        uv.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$FAKE_UV_ARGS\"\n"
                      + "printf '%s\\n' " + shlex.quote(response) + f"\nexit {exit_code}\n")
        uv.chmod(0o755)
        environment = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}", FAKE_UV_ARGS=str(root / "args"))
        code = '''
set -euo pipefail
export CS182_MODAL_IMAGE=old-image MODEL_URI=old-model DATASET_URI=old-dataset OUTPUT_URI=old-output
before_flags=$-
before_options=$(set +o)
before_directory=$PWD
if source "$1" --project custom; then status=0; else status=$?; fi
[[ $- == "$before_flags" && $(set +o) == "$before_options" && $PWD == "$before_directory" ]] || exit 90
[[ $(type -t _cs182_modal_env || true) == "" ]] || exit 91
printf 'RESULT:%s\\n' "$status"
bash -c 'printf "%s\\n" "$CS182_MODAL_IMAGE" "$MODEL_URI" "$DATASET_URI" "$OUTPUT_URI"'
'''
        return subprocess.run(["bash", "-c", code, "test", str(HELPER)], cwd=directory,
                              env=environment, capture_output=True, text=True)

    def test_source_exports_all_values_from_any_directory_and_preserves_shell_state(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.source(directory, "image:tag\tmodel-uri\tdataset-uri\toutput-uri")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[-5:], ["RESULT:0", "image:tag", "model-uri", "dataset-uri", "output-uri"])
            arguments = (Path(directory) / "args").read_text().splitlines()
            self.assertEqual(arguments[:2], ["run", "--project"])
            self.assertIn(str(INFRA / "scripts" / "discover_modal.py"), arguments)
            self.assertEqual(arguments[-2:], ["--project", "custom"])

    def test_discovery_failure_or_malformed_output_leaves_all_existing_values_unchanged(self):
        for response, exit_code in [("partial", 1), ("image\tmodel", 0), ("a\tb\tc\td\textra", 0), ("a\nb\tc\td", 0)]:
            with self.subTest(response=response, exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                result = self.source(directory, response, exit_code)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-5:], ["RESULT:1", "old-image", "old-model", "old-dataset", "old-output"])

    def test_discovered_values_are_not_evaluated_as_shell_code(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            literal = f"image:$(touch {marker})"
            result = self.source(directory, f"{literal}\tmodel\tdataset\toutput")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(literal, result.stdout.splitlines())
            self.assertFalse(marker.exists())

    def test_execution_instead_of_sourcing_returns_usage_error(self):
        result = subprocess.run(["bash", str(HELPER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Use: source", result.stderr)


if __name__ == "__main__":
    unittest.main()
