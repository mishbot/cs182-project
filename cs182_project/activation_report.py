"""Self-contained Plotly reports and portable numerical analysis artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import JsonObject, PathInput
from .activation_analysis import ActivationAnalysis, BinaryMetrics, Classifier, best_layers

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def analysis_summary(analysis: ActivationAnalysis) -> JsonObject:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_uri": analysis.manifest_uri,
        "manifest_sha256": analysis.manifest_sha256,
        "source_manifest": analysis.manifest,
        "rows": int(analysis.harmful.size),
        "class_counts": {
            "harmful": int(analysis.harmful.sum()),
            "harmless": int((~analysis.harmful).sum()),
        },
        "evaluation": {
            "method": "shuffled stratified K-fold; pooled out-of-fold predictions",
            "folds": analysis.folds,
            "seed": analysis.seed,
            "positive_class": "harmful",
            "logistic": {
                "scaler": "training-fold StandardScaler",
                "regularization": "L2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "prediction": "sklearn.predict",
            },
            "midpoint": "training-fold means; (x - (mu_h + mu_u)/2) dot (mu_h - mu_u) >= 0",
            "zero_division": 0,
        },
        "projection": "Full-run descriptive means; normalized mean difference and PC1 of centered orthogonal residuals (full SVD). Not used for evaluation.",
        "versions": {
            name: version(name) for name in ["numpy", "pyarrow", "scikit-learn", "plotly"]
        },
        "best_layers": {name: best_layers(analysis, name) for name in ("logistic", "midpoint")},
        "layers": [
            {
                "layer": layer.layer,
                "logistic": asdict(layer.logistic),
                "midpoint": asdict(layer.midpoint) if layer.midpoint is not None else None,
                "difference_norm": float(
                    (layer.projection.difference @ layer.projection.difference) ** 0.5
                ),
                "residual_pc1_explained_variance_ratio": layer.projection.residual_explained_variance_ratio,
                "notes": [*layer.projection.notes, *layer.notes],
            }
            for layer in analysis.layers
        ],
    }


def _metric_figure(analysis: ActivationAnalysis) -> Figure:
    import plotly.graph_objects as go

    figure = go.Figure()
    styles: list[tuple[Classifier, str, str]] = [
        ("logistic", "Logistic", "#7c3aed"),
        ("midpoint", "Midpoint", "#0284c7"),
    ]
    for classifier, label, color in styles:
        winners = best_layers(analysis, classifier)
        for metric, dash in [("precision", "dot"), ("recall", "dash"), ("f1", "solid")]:
            values = [
                getattr(getattr(layer, classifier), metric)
                if getattr(layer, classifier) is not None
                else None
                for layer in analysis.layers
            ]
            figure.add_trace(
                go.Scatter(
                    x=[layer.layer for layer in analysis.layers],
                    y=values,
                    name=f"{label} {metric}",
                    mode="lines+markers",
                    line={"color": color, "dash": dash},
                    marker={
                        "size": [12 if layer.layer in winners else 6 for layer in analysis.layers],
                        "symbol": [
                            "diamond" if layer.layer in winners else "circle"
                            for layer in analysis.layers
                        ],
                    },
                )
            )
    figure.update_layout(
        title="Out-of-fold metrics by layer (diamonds mark each classifier's best F1 layers)",
        xaxis_title="Decoder layer (zero-based)",
        yaxis_title="Harmful-class metric",
        yaxis_range=[0, 1.03],
        template="plotly_white",
        height=480,
    )
    return figure


def _scatter_figure(analysis: ActivationAnalysis) -> Figure:
    import numpy as np
    import plotly.graph_objects as go

    figure = go.Figure()
    logistic_best = best_layers(analysis, "logistic")
    midpoint_best = best_layers(analysis, "midpoint")
    initial = next(
        (index for index, layer in enumerate(analysis.layers) if layer.layer in logistic_best), 0
    )
    buttons: list[JsonObject] = []
    for index, layer in enumerate(analysis.layers):
        for harmful, label, color in [(True, "Harmful", "#dc2626"), (False, "Harmless", "#2563eb")]:
            selected = np.flatnonzero(analysis.harmful == harmful)
            hover = [
                [
                    str(analysis.source_row_indices[row]),
                    str(analysis.fold_ids[row]),
                    escape(analysis.prompts[row][:160]),
                ]
                for row in selected
            ]
            figure.add_trace(
                go.Scattergl(
                    x=layer.projection.coordinates[selected, 0],
                    y=layer.projection.coordinates[selected, 1],
                    name=label,
                    customdata=hover,
                    mode="markers",
                    visible=index == initial,
                    marker={"color": color, "size": 6, "opacity": 0.65},
                    hovertemplate="Source row %{customdata[0]}<br>CV fold %{customdata[1]}<br>%{customdata[2]}<br>r coordinate %{x:.4g}<br>Orthogonal PC1 %{y:.4g}<extra>%{fullData.name}</extra>",
                )
            )
        annotations = []
        if layer.layer in logistic_best:
            annotations.append("best logistic F1")
        if layer.layer in midpoint_best:
            annotations.append("best midpoint F1")
        suffix = f" ({', '.join(annotations)})" if annotations else ""
        title = f"Layer {layer.layer}{suffix}: descriptive full-run projection"
        buttons.append(
            {
                "label": f"Layer {layer.layer}{suffix}",
                "method": "update",
                "args": [
                    {"visible": [trace // 2 == index for trace in range(2 * len(analysis.layers))]},
                    {"title": {"text": title}, "xaxis.autorange": True, "yaxis.autorange": True},
                ],
            }
        )
    figure.update_layout(
        title=buttons[initial]["args"][1]["title"],
        xaxis_title="(activation − class-mean midpoint) · normalized r",
        yaxis_title="Centered activation · orthogonal PC1",
        template="plotly_white",
        height=650,
        updatemenus=[{"buttons": buttons, "active": initial, "x": 0, "y": 1.14}],
        margin={"t": 130},
    )
    figure.add_vline(x=0, line_dash="dash", line_color="gray")
    return figure


def _metric_cells(metrics: BinaryMetrics | None, winner: bool) -> str:
    if metrics is None:
        return '<td colspan="3">Unavailable: degenerate training direction</td>'
    style = ' class="best"' if winner else ""
    return "".join(
        f"<td{style}>{getattr(metrics, metric):.6f}</td>"
        for metric in ["precision", "recall", "f1"]
    )


def _report_html(analysis: ActivationAnalysis, summary: JsonObject) -> str:
    winners: dict[Classifier, list[int]] = {
        name: best_layers(analysis, name) for name in ("logistic", "midpoint")
    }
    rows: list[str] = []
    notes: list[str] = []
    for layer in analysis.layers:
        rows.append(
            f"<tr><td>{layer.layer}</td>"
            + _metric_cells(layer.logistic, layer.layer in winners["logistic"])
            + _metric_cells(layer.midpoint, layer.layer in winners["midpoint"])
            + "</tr>"
        )
        for note in [*layer.projection.notes, *layer.notes]:
            notes.append(f"<li>Layer {layer.layer}: {escape(note)}</li>")
    metrics_plot = _metric_figure(analysis).to_html(
        full_html=False, include_plotlyjs=True, div_id="metrics"
    )
    scatter = _scatter_figure(analysis).to_html(
        full_html=False, include_plotlyjs=False, div_id="projection"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation analysis</title><style>
body {{ font-family:system-ui,sans-serif; max-width:1200px; margin:2rem auto; padding:0 1rem; color:#172033; }}
table {{ border-collapse:collapse; width:100%; }} td,th {{ padding:.55rem; border:1px solid #cbd5e1; text-align:right; }}
th {{ background:#f1f5f9; }} .best {{ background:#dcfce7; font-weight:700; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; }} code {{ overflow-wrap:anywhere; }}
</style></head><body><h1>Activation analysis</h1>
<p>Run: <code>{escape(analysis.manifest_uri)}</code></p>
<p>{analysis.harmful.size} rows; harmful: {int(analysis.harmful.sum())}; harmless: {int((~analysis.harmful).sum())}.
{analysis.folds}-fold shuffled stratified cross-validation, seed {analysis.seed}. Harmful is the positive class.
Metrics use pooled out-of-fold predictions; all fitting and scaling use training rows only.</p>
<p><strong>Best logistic F1 layer(s): {escape(str(winners["logistic"]))}.</strong><br>
<strong>Best midpoint F1 layer(s): {escape(str(winners["midpoint"]))}.</strong>
All exact F1 ties are highlighted. Unavailable midpoint layers are excluded from ranking.</p>
<h2>Classifier metrics</h2><table><thead><tr><th rowspan="2">Layer</th><th colspan="3">Logistic regression</th>
<th colspan="3">Mean-difference midpoint</th></tr><tr><th>Precision</th><th>Recall</th><th>F1</th>
<th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
{metrics_plot}<h2>Layer projections</h2>
<p>These descriptive plots use full-run r = harmful mean − harmless mean. The horizontal coordinate uses normalized r,
with zero at the midpoint of the means. The vertical coordinate is PC1 after centering and removing the r component.
The dashed line is the descriptive full-run midpoint boundary; CV directions are fitted separately in each fold.
Classifier metrics use the original activation features, not this two-dimensional projection. Hover text contains prompt excerpts.</p>
{scatter}<h2>Numerical notes</h2><ul>{"".join(notes) if notes else "<li>No degenerate directions or projection axes.</li>"}</ul>
<details><summary>Provenance, settings, versions, and full-precision metrics</summary><pre>{escape(json.dumps(summary, indent=2, allow_nan=False))}</pre></details>
</body></html>"""


def write_analysis_report(analysis: ActivationAnalysis, output_dir: PathInput) -> Path:
    """Write HTML last, after all three artifacts have been generated successfully."""
    import numpy as np

    if not analysis.layers:
        raise ValueError("An analysis report requires at least one layer.")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    summary = analysis_summary(analysis)
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=directory) as temporary:
        staging = Path(temporary)
        (staging / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        np.savez_compressed(
            staging / "directions.npz",
            source_row_indices=analysis.source_row_indices,
            harmful=analysis.harmful,
            fold_ids=analysis.fold_ids,
            harmful_means=np.stack([layer.projection.harmful_mean for layer in analysis.layers]),
            harmless_means=np.stack([layer.projection.harmless_mean for layer in analysis.layers]),
            differences=np.stack([layer.projection.difference for layer in analysis.layers]),
            directions=np.stack([layer.projection.direction for layer in analysis.layers]),
            orthogonal_pc1=np.stack([layer.projection.orthogonal_pc1 for layer in analysis.layers]),
            coordinates=np.stack([layer.projection.coordinates for layer in analysis.layers]),
            logistic_predictions=np.stack(
                [layer.logistic_predictions for layer in analysis.layers]
            ),
            midpoint_predictions=np.stack(
                [layer.midpoint_predictions for layer in analysis.layers]
            ),
            midpoint_valid=np.stack([layer.midpoint_valid for layer in analysis.layers]),
        )
        (staging / "report.html").write_text(_report_html(analysis, summary), encoding="utf-8")
        for filename in ["metrics.json", "directions.npz", "report.html"]:
            os.replace(staging / filename, directory / filename)
    return directory / "report.html"
