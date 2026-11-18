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
from .activation_analysis import (
    ActivationAnalyses,
    ActivationAnalysis,
    BinaryMetrics,
    Classifier,
    best_layers,
    best_regression_layers,
    classification_metrics,
    selected_directions,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def _group_summary(analysis: ActivationAnalysis) -> JsonObject:
    return {
        "schema_version": 2,
        "group": analysis.group,
        "refusal_detector": {
            "threshold": -5,
            "comparison": ">",
            **asdict(classification_metrics(analysis.harmful, analysis.refusal_metric > -5)),
        },
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
            "refusal_regression": {
                "model": "Ridge",
                "alpha": 1.0,
                "scaler": "training-fold StandardScaler",
                "solver": "lsqr",
                "tol": 1e-8,
            },
        },
        "projection": "Full-run descriptive means; normalized mean difference and PC1 of centered orthogonal residuals (full SVD). Not used for evaluation.",
        "versions": {
            name: version(name) for name in ["numpy", "pyarrow", "scikit-learn", "plotly"]
        },
        "best_layers": {
            **{name: best_layers(analysis, name) for name in ("logistic", "midpoint")},
            "refusal_regression": best_regression_layers(analysis),
        },
        "layers": [
            {
                "layer": layer.layer,
                "logistic": asdict(layer.logistic),
                "midpoint": asdict(layer.midpoint) if layer.midpoint is not None else None,
                "refusal_regression": asdict(layer.refusal_regression),
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


def _scatter_figure(
    analysis: ActivationAnalysis,
    *,
    color_by_refusal: bool = False,
    color_range: tuple[float, float] = (-10, 10),
) -> Figure:
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
                    float(analysis.refusal_metric[row]),
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
                    marker={
                        "color": analysis.refusal_metric[selected] if color_by_refusal else color,
                        **({"coloraxis": "coloraxis"} if color_by_refusal else {}),
                        "size": 6,
                        "opacity": 0.65,
                    },
                    hovertemplate="Source row %{customdata[0]}<br>CV fold %{customdata[1]}<br>%{customdata[2]}<br>Refusal metric %{customdata[3]:.4g}<br>r coordinate %{x:.4g}<br>Orthogonal PC1 %{y:.4g}<extra>%{fullData.name}</extra>",
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
        coloraxis={
            "colorscale": "Viridis",
            "cmin": color_range[0],
            "cmax": color_range[1],
            "colorbar": {"title": "Refusal metric"},
        },
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


def analysis_summary(analysis: ActivationAnalyses) -> JsonObject:
    import numpy as np

    directions = selected_directions(analysis)
    vectors = np.stack([direction.vector for direction in directions])
    products = vectors @ vectors.T
    return {
        "schema_version": 2,
        "manifest_uri": analysis.run.manifest_uri,
        "manifest_sha256": analysis.run.manifest_sha256,
        "source_manifest": analysis.run.manifest,
        "groups": {group.group: _group_summary(group) for group in analysis.groups},
        "selected_directions": [
            {"name": item.name, "layer": item.layer, "valid": item.valid} for item in directions
        ],
        "inner_products": [
            [
                float(products[i, j]) if a.valid and b.valid else None
                for j, b in enumerate(directions)
            ]
            for i, a in enumerate(directions)
        ],
        "selection": "Best pooled OOF F1 (classifiers) or R² (refusal regression); earliest layer breaks ties; refit on full subgroup.",
    }


def _histogram(analysis: ActivationAnalyses) -> Figure:
    import numpy as np
    import plotly.graph_objects as go

    values = np.concatenate([group.refusal_metric for group in analysis.groups])
    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    bins = {"start": low, "end": high + (high - low) / 40, "size": (high - low) / 40}
    figure = go.Figure()
    colors = ["#2563eb", "#dc2626", "#0891b2", "#d97706"]
    for group in analysis.groups:
        for harmful in (False, True):
            name = f"{group.group.replace('_', ' ')} / {'harmful' if harmful else 'harmless'}"
            figure.add_trace(
                go.Histogram(
                    x=group.refusal_metric[group.harmful == harmful],
                    name=name,
                    xbins=bins,
                    opacity=0.55,
                    marker_color=colors[2 * int(group.group == "adversarial") + int(harmful)],
                )
            )
    figure.update_layout(
        title="Refusal metric by input type",
        template="plotly_white",
        barmode="overlay",
        xaxis_title="Refusal metric",
        yaxis_title="Count",
    )
    figure.add_vline(x=-5, line_dash="dash", annotation_text="Refusal > −5")
    return figure


def _inner_product_figure(summary: JsonObject) -> Figure:
    import plotly.graph_objects as go

    names = [f"{item['name']} (layer {item['layer']})" for item in summary["selected_directions"]]
    figure = go.Figure(
        go.Heatmap(
            z=summary["inner_products"],
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            texttemplate="%{z:.3f}",
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title="Inner products of selected unit directions",
        template="plotly_white",
        height=700,
        margin={"l": 260, "b": 220},
    )
    return figure


def _report_html(analysis: ActivationAnalyses, summary: JsonObject) -> str:
    import numpy as np

    values = np.concatenate([group.refusal_metric for group in analysis.groups])
    color_range = (float(values.min()), float(values.max()))
    parts = [
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Activation analysis</title>
<style>body {{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}}
table {{border-collapse:collapse;width:100%}} td,th {{padding:.55rem;border:1px solid #cbd5e1;text-align:right}}
th {{background:#f1f5f9}} .best {{background:#dcfce7;font-weight:700}} pre {{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body><h1>Activation analysis</h1><p>Run: <code>{escape(analysis.run.manifest_uri)}</code></p>
<p>Adversarial and non-adversarial samples are fitted and evaluated separately. Metrics use pooled out-of-fold
predictions with fold-local fitting and standardization. Validation rows remain reserved when collecting the default training partition.
Scores used to select layers are not independent held-out validation estimates.</p>
<p>Refusal metric = log(P(refusal tokens)) − log(P(other tokens)); Qwen Chat refusal token IDs: 40, 2121.
Refusal as a harmful-input detector uses the strict threshold &gt; −5. Harmful is the positive class.</p>"""
    ]
    parts.append(
        _histogram(analysis).to_html(
            full_html=False, include_plotlyjs=True, div_id="refusal-histogram"
        )
    )
    parts.append(
        _inner_product_figure(summary).to_html(
            full_html=False, include_plotlyjs=False, div_id="inner-products"
        )
    )
    for group in analysis.groups:
        info = summary["groups"][group.group]
        winners = info["best_layers"]
        parts.append(
            f"<h2>{escape(group.group.replace('_', ' ').title())}</h2>"
            f"<p>{group.harmful.size} rows; {group.folds}-fold CV, seed {group.seed}.</p>"
            f"<p><strong>Best logistic F1 layer(s): {winners['logistic']}.</strong><br>"
            f"<strong>Best midpoint F1 layer(s): {winners['midpoint']}.</strong><br>"
            f"<strong>Best refusal regression R² layer(s): {winners['refusal_regression']}.</strong></p>"
        )
        detector = info["refusal_detector"]
        parts.append(
            "<h3>Refusal as a harmful-input detector</h3><p>"
            + "; ".join(
                f"{metric}: {detector[metric]:.6f}" for metric in ("precision", "recall", "f1")
            )
            + f"; TP={detector['tp']}, FP={detector['fp']}, TN={detector['tn']}, FN={detector['fn']}.</p>"
        )
        rows: list[str] = []
        notes: list[str] = []
        for layer in group.layers:
            regression_style = (
                ' class="best"' if layer.layer in winners["refusal_regression"] else ""
            )
            regression = layer.refusal_regression
            cells = "".join(
                f"<td{regression_style}>{value:.6f}</td>"
                if value is not None
                else f"<td{regression_style}>Undefined</td>"
                for value in (regression.r2, regression.rmse, regression.mae)
            )
            rows.append(
                f"<tr><td>{layer.layer}</td>"
                + _metric_cells(layer.logistic, layer.layer in winners["logistic"])
                + _metric_cells(layer.midpoint, layer.layer in winners["midpoint"])
                + cells
                + "</tr>"
            )
            notes.extend(
                f"<li>Layer {layer.layer}: {escape(note)}</li>"
                for note in (*layer.projection.notes, *layer.notes)
            )
        parts.append("<h3>Per-layer metrics</h3>")
        parts.append(
            "<table><thead><tr><th rowspan='2'>Layer</th><th colspan='3'>Logistic</th>"
            "<th colspan='3'>Mean-difference midpoint</th><th colspan='3'>Refusal Ridge regression</th>"
            "</tr><tr><th>Precision</th><th>Recall</th><th>F1</th><th>Precision</th><th>Recall</th>"
            "<th>F1</th><th>R²</th><th>RMSE</th><th>MAE</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
        parts.append(
            _metric_figure(group).to_html(
                full_html=False, include_plotlyjs=False, div_id=f"{group.group}-metrics"
            )
        )
        parts.append(
            "<h3>Layer projections</h3><p>Horizontal: normalized harmful-minus-harmless mean direction, centered at the midpoint. "
            "Vertical: PC1 of centered activations after removing that direction. Geometry uses this subgroup's full sample; "
            "evaluation fits each fold separately. Both plot sets share coordinates. Hover for source rows and refusal metrics.</p>"
        )
        for refusal, title in [(False, "Harmfulness"), (True, "Refusal metric")]:
            parts.append(f"<h4>Colored by {title}</h4>")
            parts.append(
                _scatter_figure(group, color_by_refusal=refusal, color_range=color_range).to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=f"{group.group}-projection-{refusal}",
                )
            )
        parts.append("<h3>Numerical notes</h3><ul>" + "".join(notes) + "</ul>")
    parts.append(
        "<details><summary>Provenance and full-precision metrics</summary><pre>"
        + escape(json.dumps(summary, indent=2, allow_nan=False))
        + "</pre></details></body></html>"
    )
    return "".join(parts)


def write_analysis_report(analysis: ActivationAnalyses, output_dir: PathInput) -> Path:
    """Write HTML last, after all three artifacts have been generated successfully."""
    import numpy as np

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    summary = analysis_summary(analysis)
    directions = selected_directions(analysis)
    vectors = np.stack([item.vector for item in directions])
    valid = np.array([item.valid for item in directions])
    products = vectors @ vectors.T
    products[~(valid[:, None] & valid[None, :])] = np.nan
    arrays: JsonObject = {
        "source_row_indices": analysis.run.source_row_indices,
        "harmful": analysis.run.harmful,
        "adversarial": analysis.run.adversarial,
        "selected_directions": vectors,
        "selected_names": np.array([item.name for item in directions]),
        "selected_layers": np.array([item.layer for item in directions]),
        "selected_valid": valid,
        "inner_products": products,
    }
    for group in analysis.groups:
        prefix = group.group + "__"
        arrays.update(
            {
                prefix + key: value
                for key, value in {
                    "source_row_indices": group.source_row_indices,
                    "harmful": group.harmful,
                    "fold_ids": group.fold_ids,
                    "refusal_metric": group.refusal_metric,
                    "harmful_means": np.stack(
                        [layer.projection.harmful_mean for layer in group.layers]
                    ),
                    "harmless_means": np.stack(
                        [layer.projection.harmless_mean for layer in group.layers]
                    ),
                    "differences": np.stack(
                        [layer.projection.difference for layer in group.layers]
                    ),
                    "directions": np.stack([layer.projection.direction for layer in group.layers]),
                    "orthogonal_pc1": np.stack(
                        [layer.projection.orthogonal_pc1 for layer in group.layers]
                    ),
                    "coordinates": np.stack(
                        [layer.projection.coordinates for layer in group.layers]
                    ),
                    "logistic_predictions": np.stack(
                        [layer.logistic_predictions for layer in group.layers]
                    ),
                    "midpoint_predictions": np.stack(
                        [layer.midpoint_predictions for layer in group.layers]
                    ),
                    "midpoint_valid": np.stack([layer.midpoint_valid for layer in group.layers]),
                    "refusal_predictions": np.stack(
                        [layer.refusal_predictions for layer in group.layers]
                    ),
                    "logistic_directions": np.stack(
                        [layer.logistic_fit.direction for layer in group.layers]
                    ),
                    "logistic_coefficients": np.stack(
                        [layer.logistic_fit.coefficient for layer in group.layers]
                    ),
                    "logistic_intercepts": np.array(
                        [layer.logistic_fit.intercept for layer in group.layers]
                    ),
                    "logistic_valid": np.array(
                        [layer.logistic_fit.valid for layer in group.layers]
                    ),
                    "refusal_directions": np.stack(
                        [layer.refusal_fit.direction for layer in group.layers]
                    ),
                    "refusal_coefficients": np.stack(
                        [layer.refusal_fit.coefficient for layer in group.layers]
                    ),
                    "refusal_intercepts": np.array(
                        [layer.refusal_fit.intercept for layer in group.layers]
                    ),
                    "refusal_valid": np.array([layer.refusal_fit.valid for layer in group.layers]),
                    "midpoint_direction_valid": np.array(
                        [
                            bool(np.linalg.norm(layer.projection.difference))
                            for layer in group.layers
                        ]
                    ),
                }.items()
            }
        )
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=directory) as temporary:
        staging = Path(temporary)
        (staging / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        np.savez_compressed(staging / "directions.npz", **arrays)
        (staging / "report.html").write_text(_report_html(analysis, summary), encoding="utf-8")
        for filename in ["metrics.json", "directions.npz", "report.html"]:
            os.replace(staging / filename, directory / filename)
    return directory / "report.html"
