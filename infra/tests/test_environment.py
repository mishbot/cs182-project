"""Offline checks for published image and dataset discovery."""

from datetime import datetime, timezone
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

from cs182_project.dataset_artifacts import CORPUS_FILES, CORPUS_ID, DATASET_ID, SPLITS
from cs182_project.environment import discover_modal_environment
from cs182_project.model_artifacts import MANIFEST, MODEL_ID, MODEL_REVISION
from test_model_scripts import FakeS3

INFRA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("discover_modal_cli", INFRA / "scripts" / "discover_modal.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
spec = importlib.util.spec_from_file_location("qwen_experiment_cli", INFRA / "scripts" / "run_qwen_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock()
        self.session.region_name = "us-east-1"
        self.sts, self.s3, self.ecr = MagicMock(), MagicMock(), MagicMock()
        self.s3.exceptions = SimpleNamespace(NoSuchKey=type("NoSuchKey", (Exception,), {}))
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
        self.model = dict(FakeS3().manifest, model_id=MODEL_ID, upstream={"provider": "modelscope"})
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

    def test_default_selects_new_corpus_and_legacy_is_explicit(self):
        revision='c'*64
        self.dataset=dict(schema_version=1,dataset_id=CORPUS_ID,revision=revision,
                          source_files=CORPUS_FILES,preprocessing={'version':1},row_count=4,
                          sources=[dict(file=name,row_count=1,row_offset=i) for i,name in enumerate(CORPUS_FILES)],
                          files=[dict(path=name,size=1,sha256='0'*64) for name in [*CORPUS_FILES,'README.md']])
        self.s3.get_paginator.return_value.paginate.return_value=[dict(Contents=[
            dict(Key=f'datasets/refusal-direction/{revision}/{MANIFEST}',LastModified=datetime(2026,11,2,tzinfo=timezone.utc)),
            dict(Key=f'datasets/wildguardmix/{self.new_sha}/{MANIFEST}',LastModified=datetime(2026,11,3,tzinfo=timezone.utc))])]
        values=discover_modal_environment(self.session)
        self.assertEqual(values['DATASET_URI'],f's3://cs182-cddr-123456789012-us-east-1-datasets/datasets/refusal-direction/{revision}')

    def test_selects_latest_publications_across_pages_and_ignores_incomplete_uploads(self):
        values = discover_modal_environment(self.session, dataset="wildguardmix")
        self.assertEqual(values, {
            "CS182_MODAL_IMAGE": "123456789012.dkr.ecr.us-east-1.amazonaws.com/cs182-cddr-runtime:dev-0000001",
            "MODEL_URI": f"s3://cs182-cddr-123456789012-us-east-1-models/models/Qwen-1_8B-Chat/{MODEL_REVISION}",
            "DATASET_URI": f"s3://cs182-cddr-123456789012-us-east-1-datasets/datasets/wildguardmix/{self.new_sha}",
            "OUTPUT_URI": "s3://cs182-cddr-123456789012-us-east-1-activations/runs",
            "ANALYSIS_URI": "s3://cs182-cddr-123456789012-us-east-1-analyses/runs",
            "TRAINING_STATS_URI": "s3://cs182-cddr-123456789012-us-east-1-training-stats/runs",
        })
        self.s3.get_paginator.assert_called_once_with("list_objects_v2")
        self.ecr.get_paginator.assert_called_once_with("describe_images")
        self.ecr.get_paginator.return_value.paginate.assert_called_once_with(
            repositoryName="cs182-cddr-runtime", filter={"tagStatus": "TAGGED"},
        )
        self.assertTrue(all(body.closed for body in self.bodies))
        self.assertEqual(self.s3.get_object.call_count, 2)

    def test_project_and_region_overrides_use_current_account(self):
        values = discover_modal_environment(self.session, project="another-project", region="us-west-2", dataset="wildguardmix")
        self.assertIn("another-project-123456789012-us-west-2-models", values["MODEL_URI"])
        self.assertEqual(values["OUTPUT_URI"], "s3://another-project-123456789012-us-west-2-activations/runs")
        self.assertTrue(all(call.kwargs["region_name"] == "us-west-2" for call in self.session.client.call_args_list))
        self.ecr.describe_repositories.assert_called_once_with(repositoryNames=["another-project-runtime"])

    def test_explicit_model_does_not_require_the_original_model_to_be_seeded(self):
        imported = "s3://imported-models/models/finetuned"
        self.model.update(model_id="finetuned", revision="import-revision", artifact_kind="modal_volume_import")
        self.model.pop("upstream")
        read = self.s3.get_object.side_effect

        def only_imported_model(**kwargs):
            if kwargs["Bucket"].endswith("-models") and kwargs["Bucket"] != "imported-models":
                raise self.s3.exceptions.NoSuchKey("Original model is not seeded")
            return read(**kwargs)

        self.s3.get_object.side_effect = only_imported_model
        values = discover_modal_environment(self.session, dataset="wildguardmix", model_s3_uri=imported + "/")
        self.assertEqual(values["MODEL_URI"], imported)
        self.s3.get_object.assert_any_call(Bucket="imported-models", Key="models/finetuned/" + MANIFEST)

    def test_invalid_explicit_model_uri_fails_before_aws_requests(self):
        with self.assertRaises(ValueError):
            discover_modal_environment(self.session, model_s3_uri="https://example.com/model")
        self.session.client.assert_not_called()

    def test_missing_explicit_model_manifest_fails_without_fallback(self):
        self.s3.get_object.side_effect = self.s3.exceptions.NoSuchKey("missing")
        with self.assertRaisesRegex(ValueError, "Selected model has no completion manifest"):
            discover_modal_environment(self.session, model_s3_uri="s3://models/imported")
        self.s3.get_object.assert_called_once_with(Bucket="models", Key="imported/" + MANIFEST)

    def test_no_tagged_images_fails(self):
        self.ecr.get_paginator.return_value.paginate.return_value = [{"imageDetails": []}]
        with self.assertRaisesRegex(ValueError, "No tagged runtime images"):
            discover_modal_environment(self.session, dataset="wildguardmix")

    def test_no_completed_datasets_fails(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{}]
        with self.assertRaisesRegex(ValueError, "No completed wildguardmix"):
            discover_modal_environment(self.session, dataset="wildguardmix")

    def test_invalid_latest_manifest_and_identity_mismatch_do_not_fall_back(self):
        self.dataset["revision"] = self.old_sha
        with self.assertRaisesRegex(ValueError, "revision does not match"):
            discover_modal_environment(self.session, dataset="wildguardmix")
        self.dataset["revision"] = self.new_sha
        self.dataset["splits"] = {}
        with self.assertRaisesRegex(ValueError, "both official splits"):
            discover_modal_environment(self.session, dataset="wildguardmix")
        self.model["revision"] = self.old_sha
        with self.assertRaisesRegex(ValueError, "pinned model and revision"):
            discover_modal_environment(self.session, dataset="wildguardmix")

    def test_aws_errors_propagate_and_project_validation_precedes_requests(self):
        self.s3.get_object.side_effect = PermissionError("AccessDenied")
        with self.assertRaisesRegex(PermissionError, "AccessDenied"):
            discover_modal_environment(self.session, dataset="wildguardmix")
        self.session.client.reset_mock()
        with self.assertRaisesRegex(ValueError, "Project must"):
            discover_modal_environment(self.session, project="../invalid")
        self.session.client.assert_not_called()

    def test_cli_formats_values_and_does_not_print_partial_configuration_on_error(self):
        expected = dict(CS182_MODAL_IMAGE="image:tag", MODEL_URI="model", DATASET_URI="dataset", OUTPUT_URI="output", ANALYSIS_URI="analyses", TRAINING_STATS_URI="stats")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover", "--project", "custom"]), \
                patch.object(cli, "discover_modal_environment", return_value=expected) as discover, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(stdout.getvalue(), "image:tag\tmodel\tdataset\toutput\tanalyses\tstats\n")
        discover.assert_called_once_with(self.session, project="custom", region=None, dataset="refusal-direction")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover"]), \
                patch.object(cli, "discover_modal_environment", side_effect=PermissionError("AccessDenied")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cli.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("AccessDenied", stderr.getvalue())

    def test_experiment_accepts_all_six_discovery_fields(self):
        commands = []

        def execute(command, *, env, **kwargs):
            commands.append((command, dict(env)))
            output = "image:tag\tmodel\tdataset\toutput\tanalyses\tstats\n" if "infra/scripts/discover_modal.py" in command else ""
            if "get-caller-identity" in command:
                output = '{"Account": "123456789012"}'
            return subprocess.CompletedProcess(command, 0, stdout=output)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["experiment", "--seed-model", "--results-file", str(Path(directory) / "results.json")]), \
                patch.object(experiment, "execute", side_effect=execute), \
                patch.object(experiment, "result_uri", side_effect=lambda result, **kwargs: kwargs["placeholder"]), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(experiment.main(), 0)
        collection = next(env for command, env in commands if "original_activations" in command)
        self.assertEqual(collection["CS182_MODAL_IMAGE"], "image:tag")
        self.assertEqual(collection["TRAINING_STATS_URI"], "stats")
        modal_runs = [(command, env) for command, env in commands
                      if any(command[i:i + 2] == ["modal", "run"] for i in range(len(command) - 1))]
        self.assertEqual(len(modal_runs), 6)
        for command, env in modal_runs:
            self.assertIn(["modal", "run", "--detach"], [command[i:i + 3] for i in range(len(command) - 2)])
            self.assertEqual(env["CS182_MODAL_DETACH"], "1")


if __name__ == "__main__":
    unittest.main()
