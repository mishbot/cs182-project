"""Small deterministic scientific fixtures; no AWS, models, or git state."""

from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs182_project import activation_analysis as analysis
from cs182_project import activation_report as report
from cs182_project import activation_runs as runs

AVAILABLE = all(importlib.util.find_spec(name) for name in ["numpy", "pyarrow", "sklearn", "plotly"])
if AVAILABLE:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.preprocessing import StandardScaler


def fixture_run():
    labels = np.tile([False, True], 10)
    features = np.column_stack([np.where(labels, 4., -4.), np.tile([-2., -1., 0., 1., 2.], 4), np.zeros(20)])
    values = np.stack([features, np.zeros_like(features), features], axis=1).astype(np.float32)
    manifest = dict(schema_version=1, status="complete", run_id="fixture", row_count=20, sample_size=20,
                    num_layers=3, hidden_size=3, layer_index_base=0,
                    selected_source_row_indices=list(range(100, 120)),
                    class_counts={"harmful": 10, "unharmful": 10})
    return runs.ActivationRun("s3://activations/runs/fixture/manifest.json", "0" * 64,
                              manifest, np.arange(100, 120, dtype=np.int64),
                              ['prompt </script><script>window.bad=1</script> & text'] * 20, labels, values)


class MemoryRunS3:
    def __init__(self):
        self.objects = {}
        self.pages = []
        self.bodies = []

    def get_object(self, Bucket, Key):
        body = io.BytesIO(self.objects[Key])
        self.bodies.append(body)
        return {"Body": body}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        return iter(self.pages)


@unittest.skipUnless(AVAILABLE, "Install the analysis extra")
class ActivationLoadingTest(unittest.TestCase):
    def setUp(self):
        from cs182_project.activations import activation_schema

        self.run = fixture_run()
        self.s3 = MemoryRunS3()
        self.key = "runs/fixture/manifest.json"
        self.rows = [dict(source_row_index=int(index), prompt=prompt, harmful=bool(flag),
                          prompt_harm_label="harmful" if flag else "unharmful",
                          formatted_input_token_count=3, activations=vector.tolist())
                     for index, prompt, flag, vector in zip(self.run.source_row_indices, self.run.prompts,
                                                           self.run.harmful, self.run.activations)]
        self.schema = activation_schema(3, 3)
        self.write_shard()

    def write_shard(self):
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(self.rows, schema=self.schema), buffer)
        data = buffer.getvalue().to_pybytes()
        self.s3.objects["runs/fixture/part-00000.parquet"] = data
        self.manifest = dict(self.run.manifest, shards=[dict(key="runs/fixture/part-00000.parquet", rows=20,
                                                           size=len(data), sha256=hashlib.sha256(data).hexdigest())])
        self.write_manifest()

    def write_manifest(self):
        self.s3.objects[self.key] = json.dumps(self.manifest).encode()

    def test_loads_verified_parquet_arrays_and_closes_streams(self):
        loaded = runs.load_activation_run(self.s3, "s3://activations/runs/fixture")
        np.testing.assert_array_equal(loaded.activations, self.run.activations)
        np.testing.assert_array_equal(loaded.source_row_indices, self.run.source_row_indices)
        self.assertEqual(loaded.activations.dtype, np.float32)
        self.assertEqual(loaded.prompts, self.run.prompts)
        self.assertEqual(loaded.manifest_sha256, hashlib.sha256(self.s3.objects[self.key]).hexdigest())
        self.assertTrue(all(body.closed for body in self.s3.bodies))

    def test_latest_selection_uses_all_pages_and_completion_markers(self):
        timestamp = lambda day: datetime(2026, 11, day, tzinfo=timezone.utc)
        self.s3.objects["runs/old/manifest.json"] = self.s3.objects[self.key]
        self.s3.objects["runs/incomplete/manifest.json"] = b'{"status":"incomplete"}'
        self.s3.pages = [dict(Contents=[dict(Key="runs/old/manifest.json", LastModified=timestamp(1))]),
                         dict(Contents=[dict(Key=self.key, LastModified=timestamp(2)),
                                        dict(Key="runs/orphan/part.parquet", LastModified=timestamp(4)),
                                        dict(Key="runs/incomplete/manifest.json", LastModified=timestamp(3))])]
        self.assertEqual(runs.latest_activation_run(self.s3, "s3://activations/runs"), self.run.manifest_uri)
        self.s3.pages = [dict(Contents=[])]
        with self.assertRaisesRegex(ValueError, "No completed"):
            runs.latest_activation_run(self.s3, "s3://activations/runs")

    def test_checksum_dimensions_counts_and_unsafe_keys_fail_clearly(self):
        self.s3.objects["runs/fixture/part-00000.parquet"] += b"corrupt"
        with self.assertRaisesRegex(ValueError, "Checksum/size"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.write_shard()
        self.manifest["hidden_size"] = 4
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "hidden dimensions"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.write_shard()
        self.manifest["class_counts"] = {"harmful": 9, "unharmful": 11}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Class counts"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.write_shard()
        self.manifest["shards"][0]["key"] = "runs/another-run/part.parquet"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "within the selected run"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)

    def test_nonfinite_activations_and_inconsistent_labels_are_rejected(self):
        self.rows[0]["activations"][0][0] = float("nan")
        self.write_shard()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.rows[0]["activations"][0][0] = -4.
        self.rows[0]["prompt_harm_label"] = "harmful"
        self.write_shard()
        with self.assertRaisesRegex(ValueError, "invalid indices, labels"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)


@unittest.skipUnless(AVAILABLE, "Install the analysis extra")
class ActivationMathTest(unittest.TestCase):
    def test_projection_has_expected_means_geometry_and_orthogonal_pc(self):
        run = fixture_run()
        x = run.activations[:, 0].astype(np.float64)
        projection = analysis.project_layer(x, run.harmful)
        np.testing.assert_allclose(projection.difference, [8, 0, 0])
        np.testing.assert_allclose(projection.direction, [1, 0, 0])
        np.testing.assert_allclose(projection.orthogonal_pc1, [0, 1, 0])
        self.assertAlmostEqual(projection.direction @ projection.orthogonal_pc1, 0)
        np.testing.assert_allclose(projection.coordinates, x[:, :2])
        self.assertAlmostEqual(projection.residual_explained_variance_ratio, 1)

    def test_midpoint_boundary_and_metrics_have_known_values(self):
        training = np.array([[0., 0.], [2., 0.], [4., 0.], [6., 0.]])
        predictions = analysis.midpoint_predict(training, np.array([False, False, True, True]),
                                                 np.array([[2.9, 50.], [3., 0.], [3.1, -50.]]))
        np.testing.assert_array_equal(predictions, [False, True, True])
        metrics = analysis.classification_metrics(np.array([True]*5+[False]*3),
                                                  np.array([True]*3+[False]*2+[True]+[False]*2))
        self.assertEqual((metrics.tp, metrics.fp, metrics.tn, metrics.fn), (3, 1, 2, 2))
        self.assertAlmostEqual(metrics.precision, .75)
        self.assertAlmostEqual(metrics.recall, .6)
        self.assertAlmostEqual(metrics.f1, 2/3)

    def test_cv_fits_only_training_rows_and_highlights_all_f1_ties(self):
        run = fixture_run()
        fits = []
        original = StandardScaler.fit

        def record_fit(scaler, x, y=None, **kwargs):
            fits.append(x.copy())
            return original(scaler, x, y, **kwargs)

        with patch.object(StandardScaler, "fit", new=record_fit), \
                patch.object(analysis, "midpoint_predict", wraps=analysis.midpoint_predict) as midpoint:
            result = analysis.analyze_activations(run, folds=5, seed=7)
        self.assertEqual(len(fits), 15)
        for layer in range(3):
            for fold in range(5):
                expected = run.activations[result.fold_ids != fold, layer, :]
                np.testing.assert_array_equal(fits[layer*5+fold], expected)
                np.testing.assert_array_equal(midpoint.call_args_list[layer*5+fold].args[0], expected)
        self.assertEqual(analysis.best_layers(result, "logistic"), [0, 2])
        self.assertEqual(analysis.best_layers(result, "midpoint"), [0, 2])
        self.assertEqual(result.layers[0].logistic.f1, 1)
        self.assertIsNone(result.layers[1].midpoint)
        self.assertFalse(result.layers[1].midpoint_valid.any())
        np.testing.assert_array_equal(np.bincount(result.fold_ids), [4]*5)

    def test_degenerate_axes_small_runs_and_nonconvergence(self):
        run = fixture_run()
        projection = analysis.project_layer(run.activations[:, 1].astype(np.float64), run.harmful)
        np.testing.assert_array_equal(projection.coordinates, np.zeros((20, 2)))
        self.assertTrue(projection.notes)
        with self.assertRaisesRegex(ValueError, "class count"):
            analysis.analyze_activations(replace(run, harmful=np.array([True]*19+[False])), folds=5)
        from sklearn.exceptions import ConvergenceWarning
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=ConvergenceWarning("not converged")):
            with self.assertRaisesRegex(ValueError, "layer 0, fold 0"):
                analysis.analyze_activations(run)

    def test_report_is_offline_and_exports_directions_predictions_and_provenance(self):
        result = analysis.analyze_activations(fixture_run())
        with tempfile.TemporaryDirectory() as directory:
            path = report.write_analysis_report(result, directory)
            text = path.read_text()
            self.assertIn('class="best"', text)
            self.assertIn('Best logistic F1 layer(s): [0, 2]', text)
            self.assertNotIn('prompt </script><script>window.bad=1</script>', text)
            scripts = []

            class ScriptParser(HTMLParser):
                def handle_starttag(self, tag, attrs):
                    if tag == "script":
                        scripts.append(dict(attrs))

            ScriptParser().feed(text)
            self.assertTrue(scripts)
            self.assertTrue(all('src' not in attributes for attributes in scripts))
            summary = json.loads((Path(directory)/"metrics.json").read_text())
            self.assertEqual(summary['best_layers'], {'logistic': [0, 2], 'midpoint': [0, 2]})
            self.assertEqual(summary['evaluation']['folds'], 5)
            self.assertEqual(summary['manifest_uri'], result.manifest_uri)
            with np.load(Path(directory)/"directions.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays['differences'].shape, (3, 3))
                np.testing.assert_allclose(arrays['differences'][0], [8, 0, 0])
                self.assertEqual(arrays['logistic_predictions'].shape, (3, 20))
                np.testing.assert_array_equal(arrays['fold_ids'], result.fold_ids)


if __name__ == "__main__":
    unittest.main()
