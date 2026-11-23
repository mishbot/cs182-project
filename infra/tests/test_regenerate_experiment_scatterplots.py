"""Offline checks for local PNG regeneration from exported analysis arrays."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_activation_analysis import AVAILABLE

if AVAILABLE:
    import numpy as np
    import plotly.graph_objects as go

    from cs182_project.activation_report import render_scatterplot_pngs

from cs182_project.activation_report import download_analysis_plot_artifacts
from infra.scripts.regenerate_experiment_scatterplots import analysis_stages


class ArtifactS3:
    def __init__(self, objects):
        self.objects = objects

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


class DownloadAnalysisArtifactsTest(unittest.TestCase):
    def test_downloads_only_verified_numerical_artifacts(self):
        metrics = json.dumps({"manifest_sha256": "a" * 64}).encode()
        directions = b"npz fixture"
        prefix = "runs/run-id/analysis-id"
        entries = [
            {
                "key": f"{prefix}/{name}",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in (("metrics.json", metrics), ("directions.npz", directions))
        ]
        manifest = json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "analysis_id": "analysis-id",
                "source_manifest_sha256": "a" * 64,
                "artifacts": entries,
            }
        ).encode()
        s3 = ArtifactS3(
            {
                f"{prefix}/manifest.json": manifest,
                f"{prefix}/metrics.json": metrics,
                f"{prefix}/directions.npz": directions,
            }
        )
        loaded_metrics, loaded_directions = download_analysis_plot_artifacts(
            s3, f"s3://analysis-bucket/{prefix}/report.html"
        )
        self.assertEqual(loaded_metrics["manifest_sha256"], "a" * 64)
        self.assertEqual(loaded_directions, directions)


class ExperimentStageSelectionTest(unittest.TestCase):
    def test_returns_both_analysis_stages(self):
        experiment = {
            "stages": {
                "original_analysis": {"status": "complete"},
                "orthogonalized_analysis": {"status": "complete"},
            },
            "original_analysis": "s3://analysis/original/report.html",
            "orthogonalized_analysis": "s3://analysis/orthogonalized/report.html",
        }
        self.assertEqual(
            analysis_stages(experiment),
            [
                ("original_analysis", "originalqwen"),
                ("orthogonalized_analysis", "originalqwen_orthogonalized"),
            ],
        )

    def test_requires_both_stages_to_be_complete(self):
        experiment = {
            "stages": {"original_analysis": {"status": "complete"}},
            "original_analysis": "s3://analysis/original/report.html",
        }
        with self.assertRaisesRegex(ValueError, "orthogonalized_analysis is not complete"):
            analysis_stages(experiment)

@unittest.skipUnless(AVAILABLE, "Install the analysis extra")
class ScatterplotRegenerationTest(unittest.TestCase):
    def test_renders_both_plot_variants_to_collision_safe_experiment_directory(self):
        arrays = io.BytesIO()
        np.savez_compressed(
            arrays,
            non_adversarial__harmful=np.array([True, False]),
            non_adversarial__refusal_metric=np.array([2.0, -1.0]),
            non_adversarial__coordinates=np.array([[[1.0, 0.0], [-1.0, 0.0]], [[2.0, 1.0], [-2.0, -1.0]]]),
        )
        metrics = {
            "groups": {
                "non_adversarial": {"best_layers": {"midpoint": [1, 0]}},
            }
        }
        written = []

        def fake_write_image(figure, file, **kwargs):
            Path(file).write_bytes(b"png")
            written.append((Path(file).name, figure.data[0].x.tolist()))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment-a1b2"
            with patch.object(go.Figure, "write_image", fake_write_image):
                paths = render_scatterplot_pngs(
                    metrics, arrays.getvalue(), output, name_prefix="originalqwen"
                )
            self.assertEqual(
                [path.name for path in paths],
                [
                    "originalqwen_midpointF1_scatter_by_class.png",
                    "originalqwen_midpointF1_scatter_by_refusal_metric.png",
                ],
            )
            self.assertEqual([item[1] for item in written], [[1.0], [1.0]])
            self.assertTrue(all(path.read_bytes() == b"png" for path in paths))


if __name__ == "__main__":
    unittest.main()
