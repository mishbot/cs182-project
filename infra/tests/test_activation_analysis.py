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
    labels = np.tile([False, True], 20)
    features = np.column_stack([np.where(labels, 4., -4.), np.tile([-2., -1., 0., 1., 2.], 8), np.zeros(40)])
    values = np.stack([features, np.zeros_like(features), features], axis=1).astype(np.float32)
    manifest = dict(schema_version=2, status="complete", run_id="fixture", row_count=40, sample_size=40,
                    num_layers=3, hidden_size=3, layer_index_base=0,
                    selected_source_row_indices=list(range(100, 140)),
                    class_counts={"harmful": 20, "unharmful": 20})
    return runs.ActivationRun("s3://activations/runs/fixture/manifest.json", "0" * 64,
                              manifest, np.arange(100, 140, dtype=np.int64),
                              ['prompt </script><script>window.bad=1</script> & text'] * 40, labels, values,
                              np.repeat([False, True], 20),
                              -np.logaddexp(0, -(features[:, 0] * .3 + features[:, 1] * 2 - 5)),
                              -np.logaddexp(0, features[:, 0] * .3 + features[:, 1] * 2 - 5),
                              np.tile(np.arange(100, dtype=np.int32), (40, 1)), np.full((40, 100), .001))


@unittest.skipUnless(AVAILABLE, "Install the analysis extra")
class ParallelAnalysisTest(unittest.TestCase):
    def test_serial_and_process_workers_agree(self):
        run = fixture_run()
        serial = analysis.analyze_activations(run, include_adversarial=True, include_ridge=True, workers=1)
        parallel = analysis.analyze_activations(run, include_adversarial=True, include_ridge=True, workers=2)
        for expected, actual in zip(serial.groups, parallel.groups):
            self.assertEqual([layer.layer for layer in actual.layers], [0, 1, 2])
            np.testing.assert_array_equal(expected.fold_ids, actual.fold_ids)
            for left, right in zip(expected.layers, actual.layers):
                self.assertEqual(left.logistic, right.logistic)
                self.assertEqual(left.midpoint, right.midpoint)
                self.assertEqual(left.refusal_regression.r2 is None, right.refusal_regression.r2 is None)
                np.testing.assert_allclose(
                    [left.refusal_regression.rmse, left.refusal_regression.mae],
                    [right.refusal_regression.rmse, right.refusal_regression.mae], atol=1e-10)
                if left.refusal_regression.r2 is not None:
                    np.testing.assert_allclose(left.refusal_regression.r2,
                                               right.refusal_regression.r2, atol=1e-10)
                for name in ("logistic_predictions", "midpoint_predictions", "refusal_predictions"):
                    np.testing.assert_allclose(getattr(left, name), getattr(right, name), atol=1e-10)
                for name in ("logistic_fit", "refusal_fit"):
                    np.testing.assert_allclose(getattr(left, name).direction,
                                               getattr(right, name).direction, atol=1e-10)
                np.testing.assert_allclose(left.projection.coordinates, right.projection.coordinates,
                                           atol=1e-10)


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
                          formatted_input_token_count=3, activations=vector.tolist(),
                          adversarial=bool(self.run.adversarial[i]),
                          refusal_log_probability=float(self.run.refusal_log_probability[i]),
                          nonrefusal_log_probability=float(self.run.nonrefusal_log_probability[i]),
                          top_token_ids=self.run.top_token_ids[i].tolist(),
                          top_token_probabilities=self.run.top_token_probabilities[i].tolist())
                     for i, (index, prompt, flag, vector) in enumerate(zip(self.run.source_row_indices, self.run.prompts,
                                                           self.run.harmful, self.run.activations))]
        self.schema = activation_schema(3, 3)
        self.write_shard()

    def write_shard(self):
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(self.rows, schema=self.schema), buffer)
        data = buffer.getvalue().to_pybytes()
        self.s3.objects["runs/fixture/part-00000.parquet"] = data
        self.manifest = dict(self.run.manifest, shards=[dict(key="runs/fixture/part-00000.parquet", rows=40,
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
        self.manifest["class_counts"] = {"harmful": 19, "unharmful": 21}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Class counts"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.write_shard()
        self.manifest["shards"][0]["key"] = "runs/another-run/part.parquet"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "within the selected run"):
            runs.load_activation_run(self.s3, self.run.manifest_uri)

    def test_legacy_runs_and_invalid_new_columns_are_rejected(self):
        self.manifest['schema_version'] = 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Recollect'):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.write_shard()
        self.rows[0]['nonrefusal_log_probability'] = -20.
        self.write_shard()
        with self.assertRaisesRegex(ValueError, 'log-probability'):
            runs.load_activation_run(self.s3, self.run.manifest_uri)
        self.rows[0]['nonrefusal_log_probability'] = float(self.run.nonrefusal_log_probability[0])
        self.rows[0]['top_token_probabilities'][1] = 0.5
        self.write_shard()
        with self.assertRaisesRegex(ValueError, 'top-100'):
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
            result = analysis.analyze_activations(run, folds=5, seed=7, include_adversarial=True)
        self.assertEqual(len(fits), 36)
        for group_index, group in enumerate(result.groups):
            subset = run.activations[run.adversarial == bool(group_index)]
            for layer in range(3):
                for fold in range(5):
                    expected = subset[group.fold_ids != fold, layer, :]
                    np.testing.assert_array_equal(fits[group_index*18+layer*6+fold], expected)
                    np.testing.assert_array_equal(midpoint.call_args_list[group_index*15+layer*5+fold].args[0], expected)
            self.assertEqual(analysis.best_layers(group, "logistic"), [0, 2])
            self.assertEqual(analysis.best_layers(group, "midpoint"), [0])
            self.assertEqual(group.layers[0].logistic.f1, 1)
            self.assertIsNone(group.layers[1].midpoint)
            self.assertFalse(group.layers[1].midpoint_valid.any())
            np.testing.assert_array_equal(np.bincount(group.fold_ids), [4]*5)

    def test_midpoint_cutoff_excludes_late_winners(self):
        result = analysis.analyze_activations(fixture_run())
        group = result.groups[0]
        template = group.layers[0]
        layers = [
            replace(template, layer=i, midpoint=replace(template.midpoint, f1=score))
            for i, score in enumerate([.5] * 16 + [.75, .75, .9] + [.5] * 4 + [1.0])
        ]
        group = replace(group, layers=layers)
        result = replace(result, groups=(group,))
        self.assertEqual(analysis.best_layers(group, "midpoint"), [16, 17])
        midpoint = next(item for item in analysis.selected_directions(result)
                        if item.name == "non_adversarial_midpoint")
        self.assertEqual(midpoint.layer, 16)
        self.assertTrue(midpoint.valid)
        group = replace(group, layers=[replace(layer, midpoint=None) if layer.layer < 18
                                      else layer for layer in layers])
        midpoint = analysis.selected_directions(replace(result, groups=(group,)))[1]
        self.assertEqual(midpoint.layer, -1)
        self.assertFalse(midpoint.valid)

    def test_degenerate_axes_small_runs_and_nonconvergence(self):
        run = fixture_run()
        projection = analysis.project_layer(run.activations[:, 1].astype(np.float64), run.harmful)
        np.testing.assert_array_equal(projection.coordinates, np.zeros((40, 2)))
        self.assertTrue(projection.notes)
        with self.assertRaisesRegex(ValueError, "class count"):
            analysis.analyze_activations(replace(run, harmful=np.array([True]*39+[False])), folds=5)
        from sklearn.exceptions import ConvergenceWarning
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=ConvergenceWarning("not converged")):
            with self.assertRaisesRegex(ValueError, "layer 0, fold 0"):
                analysis.analyze_activations(run, include_adversarial=True)

    def test_report_is_offline_and_exports_directions_predictions_and_provenance(self):
        result = analysis.analyze_activations(fixture_run(), include_adversarial=True)
        with tempfile.TemporaryDirectory() as directory:
            path = report.write_analysis_report(result, directory)
            text = path.read_text()
            self.assertIn('class="best"', text)
            self.assertIn('Best logistic F1 layer(s): [0, 2]', text)
            self.assertIn('Best eligible midpoint F1 layer(s): [0]', text)
            self.assertNotIn('Refusal Ridge regression</th>', text)
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
            self.assertEqual(summary['groups']['non_adversarial']['best_layers']['logistic'], [0, 2])
            self.assertEqual(summary['groups']['non_adversarial']['evaluation']['folds'], 5)
            self.assertEqual(summary['manifest_uri'], result.run.manifest_uri)
            self.assertFalse(summary['include_ridge'])
            self.assertEqual(summary['groups']['non_adversarial']['evaluation']
                             ['midpoint_layer_selection']['block_input_limit_exclusive'], 2)
            with np.load(Path(directory)/"directions.npz", allow_pickle=False) as arrays:
                self.assertNotIn('non_adversarial__refusal_predictions', arrays.files)
                exported = dict(zip(arrays['selected_names'], arrays['selected_layers']))
                self.assertEqual(exported['non_adversarial_midpoint'], 0)
                self.assertEqual(arrays['non_adversarial__differences'].shape, (3, 3))
                np.testing.assert_allclose(arrays['non_adversarial__differences'][0], [8, 0, 0])
                self.assertEqual(arrays['non_adversarial__logistic_predictions'].shape, (3, 20))
                np.testing.assert_array_equal(arrays['non_adversarial__fold_ids'], result.groups[0].fold_ids)


@unittest.skipUnless(AVAILABLE, "Install the analysis extra")
class RefusalAnalysisTest(unittest.TestCase):
    def test_refusal_metric_and_strict_threshold_have_known_confusion_counts(self):
        values = np.array([-10., 0., 1., 5.])
        log_refusal = -np.logaddexp(0, -values)
        log_nonrefusal = -np.logaddexp(0, values)
        np.testing.assert_allclose(analysis.refusal_metric(log_refusal, log_nonrefusal), values)
        metrics = analysis.classification_metrics(np.array([False, True, False, True]), values > 0)
        self.assertEqual((metrics.tp, metrics.fp, metrics.tn, metrics.fn), (1, 1, 1, 1))
        self.assertEqual((metrics.precision, metrics.recall, metrics.f1), (.5, .5, .5))
        with self.assertRaisesRegex(ValueError, "normalized"):
            analysis.refusal_metric(np.zeros(4), np.zeros(4))

    def test_regression_and_exported_raw_coefficients_match_independent_fit(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        run = fixture_run()
        run = replace(run, activations=run.activations * np.array([3., 100., 1.], dtype=np.float32))
        result = analysis.analyze_activations(run, include_adversarial=True, include_ridge=True)
        selected = analysis.selected_directions(result)
        self.assertEqual(len(selected), 6)
        for group in result.groups:
            subset = run.activations[run.adversarial == (group.group == 'adversarial')]
            features = subset[:, 0].astype(np.float64)
            scaler = StandardScaler().fit(features)
            for fitted, expected in [
                (group.layers[0].logistic_fit,
                 LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=0).fit(scaler.transform(features), group.harmful)),
                (group.layers[0].refusal_fit,
                 Ridge(alpha=1, solver='lsqr', tol=1e-8).fit(scaler.transform(features), group.refusal_metric))]:
                coefficients = np.asarray(expected.coef_).reshape(-1) / scaler.scale_
                np.testing.assert_allclose(fitted.coefficient, coefficients)
                self.assertAlmostEqual(fitted.intercept, float(np.asarray(expected.intercept_).reshape(-1)[0] - coefficients @ scaler.mean_))
                self.assertAlmostEqual(np.linalg.norm(fitted.direction), 1)
            self.assertGreater(group.layers[0].refusal_regression.r2, .98)
            self.assertEqual(analysis.best_regression_layers(group), [0, 2])
        with tempfile.TemporaryDirectory() as directory:
            report.write_analysis_report(result, directory)
            with np.load(Path(directory)/'directions.npz', allow_pickle=False) as arrays:
                directions = arrays['selected_directions']
                np.testing.assert_allclose(arrays['inner_products'], directions @ directions.T)
                np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
                self.assertEqual(directions.shape, (6, 3))
                self.assertEqual(arrays['selected_layers'].tolist(), [0]*6)
            summary = json.loads((Path(directory)/'metrics.json').read_text())
            figure = report._histogram(result, result.groups[0])
            self.assertEqual(len(figure.data), 2)
            self.assertEqual(len({trace.xbins.size for trace in figure.data}), 1)
            self.assertEqual(summary['groups']['adversarial']['refusal_detector']['threshold'], 0)
            other = report._histogram(result, result.groups[1])
            self.assertEqual(figure.data[0].xbins.to_plotly_json(), other.data[0].xbins.to_plotly_json())
            self.assertEqual(float(figure.layout.shapes[0].x0), 0)

    def test_default_skips_adversarial_and_ridge_fits(self):
        run = fixture_run()
        # A one-class adversarial subset must not block non-adversarial evaluation.
        harmful = run.harmful.copy()
        harmful[run.adversarial] = True
        with patch("sklearn.linear_model.Ridge.fit", side_effect=AssertionError("Unexpected Ridge fit")):
            result = analysis.analyze_activations(replace(run, harmful=harmful))
        self.assertEqual([group.group for group in result.groups], ['non_adversarial'])
        self.assertEqual(len(analysis.selected_directions(result)), 2)
        summary = report.analysis_summary(result)
        self.assertFalse(summary['include_adversarial'])
        self.assertEqual(np.shape(summary['inner_products']), (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            text = report.write_analysis_report(result, directory).read_text()
            self.assertIn('id="non_adversarial-refusal-histogram"', text)
            self.assertNotIn('id="adversarial-refusal-histogram"', text)
        with self.assertRaisesRegex(ValueError, 'adversarial:.*class count'):
            analysis.analyze_activations(replace(run, harmful=harmful), include_adversarial=True)

    def test_subgroups_are_independent_and_constant_targets_are_flagged(self):
        run = fixture_run()
        original = analysis.analyze_activations(run, include_adversarial=True, include_ridge=True)
        features = run.activations.copy()
        features[run.adversarial] += 100
        changed = analysis.analyze_activations(replace(run, activations=features), include_adversarial=True, include_ridge=True)
        for old, new in zip(original.groups[0].layers, changed.groups[0].layers):
            np.testing.assert_array_equal(old.logistic_fit.coefficient, new.logistic_fit.coefficient)
            np.testing.assert_array_equal(old.refusal_fit.coefficient, new.refusal_fit.coefficient)
            np.testing.assert_array_equal(old.refusal_predictions, new.refusal_predictions)
        targets = np.full(40, -5.)
        constant = analysis.analyze_activations(replace(run, refusal_log_probability=-np.logaddexp(0, -targets),
                                                       nonrefusal_log_probability=-np.logaddexp(0, targets)), include_adversarial=True, include_ridge=True)
        self.assertEqual(analysis.best_regression_layers(constant.groups[0]), [])
        self.assertFalse(analysis.selected_directions(constant)[2].valid)
        self.assertIsNone(constant.groups[0].layers[0].refusal_regression.r2)


if __name__ == "__main__":
    unittest.main()
