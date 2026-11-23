"""Offline checks for published image and dataset discovery."""

from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
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
        expected = dict(CS182_MODAL_IMAGE="image:tag", MODEL_URI="model", DATASET_URI="dataset", OUTPUT_URI="output", ANALYSIS_URI="analyses")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover", "--project", "custom"]), \
                patch.object(cli, "discover_modal_environment", return_value=expected) as discover, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(stdout.getvalue(), "image:tag\tmodel\tdataset\toutput\tanalyses\n")
        discover.assert_called_once_with(self.session, project="custom", region=None, dataset="refusal-direction")
        with patch.dict(sys.modules, {"boto3": SimpleNamespace(Session=lambda: self.session)}), \
                patch.object(sys, "argv", ["discover"]), \
                patch.object(cli, "discover_modal_environment", side_effect=PermissionError("AccessDenied")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cli.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("AccessDenied", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
