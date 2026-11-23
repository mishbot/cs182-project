"""Offline publication checks using scientific fixtures."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cs182_project import analysis_jobs as jobs
from test_activation_analysis import AVAILABLE, fixture_run
from test_activations import MemoryS3


class AnalysisS3(MemoryS3):
    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        with open(filename, 'rb') as body:
            self.upload_fileobj(body, bucket, key, ExtraArgs)


@unittest.skipUnless(AVAILABLE, 'Install the analysis extra')
class AnalysisPublicationTest(unittest.TestCase):
    def test_workflow_exports_only_enabled_groups_and_publishes_manifest_last(self):
        s3 = AnalysisS3()
        session = MagicMock(region_name='us-east-1')
        session.client.return_value = s3
        run = fixture_run()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(jobs, 'load_activation_run', return_value=run) as load:
            uri = jobs.run_analysis(session, run_s3_uri=run.manifest_uri,
                                    output_s3_uri='s3://analyses/runs', output_dir=directory)
            load.assert_called_once_with(s3, run.manifest_uri)
            self.assertTrue(uri.startswith('s3://analyses/runs/fixture/'))
            self.assertTrue(uri.endswith('/report.html'))
            self.assertTrue((Path(directory)/'report.html').exists())
            self.assertEqual(len(s3.uploads), 4)
            self.assertTrue(s3.uploads[-1][1].endswith('/manifest.json'))
            manifest = json.loads(s3.objects[s3.uploads[-1]])
            self.assertFalse(manifest['include_adversarial'])
            self.assertEqual(manifest['refusal_threshold'], 0)
            self.assertEqual(set(manifest['timings_seconds']), {'download', 'analysis', 'render', 'upload'})
            self.assertEqual(manifest['execution']['backend'], 'cpu')
            for artifact in manifest['artifacts']:
                data = s3.objects['analyses', artifact['key']]
                self.assertEqual(artifact['sha256'], hashlib.sha256(data).hexdigest())
                self.assertEqual(artifact['size'], len(data))
            metrics = json.loads((Path(directory)/'metrics.json').read_text())
            self.assertEqual(list(metrics['groups']), ['non_adversarial'])
            self.assertEqual(metrics['execution']['backend'], 'cpu')

    def test_upload_failure_never_publishes_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in jobs.ARTIFACT_TYPES:
                (Path(directory)/name).write_text('fixture')
            s3 = AnalysisS3()
            s3.fail_upload_after = 1
            with self.assertRaisesRegex(RuntimeError, 'Incomplete analysis'):
                jobs.publish_analysis(s3, 's3://analyses/runs', directory, 'fixture', {})
            self.assertFalse(any(key.endswith('manifest.json') for _, key in s3.objects))
            with self.assertRaisesRegex(ValueError, 'run ID'):
                jobs.publish_analysis(s3, 's3://analyses/runs', directory, '../escape', {})

    def test_latest_and_default_output_resolve_without_model_or_ecr_requests(self):
        s3, sts = AnalysisS3(), MagicMock()
        sts.get_caller_identity.return_value = {'Account': '123456789012'}
        session = MagicMock(region_name='us-east-1')
        session.client.side_effect = lambda service, **kwargs: {'s3': s3, 'sts': sts}[service]
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True), \
                patch.object(jobs, 'latest_activation_run', return_value=fixture_run().manifest_uri) as latest, \
                patch.object(jobs, 'load_activation_run', return_value=fixture_run()):
            uri = jobs.run_analysis(session, output_dir=directory)
        self.assertTrue(uri.startswith('s3://cs182-cddr-123456789012-us-east-1-analyses/runs/'))
        latest.assert_called_once_with(s3, 's3://cs182-cddr-123456789012-us-east-1-activations/runs')


if __name__ == '__main__':
    unittest.main()
