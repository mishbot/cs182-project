"""Build side-by-side figures from saved activation analysis artifacts."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._types import JsonObject
from .model_artifacts import parse_s3_uri
from .report_paths import model_role

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from plotly.graph_objects import Figure

MODEL_NAMES = ("Original Model", "Defended Model")
MODEL_STATES = ("Non-orthogonalized", "Orthogonalized")
FIGURE_NAMES = ("scatter_by_class", "scatter_by_refusal_metric", "refusal_histogram")
CLASSES = (
    (True, "Harmful", "#dc2626", "triangle-up"),
    (False, "Harmless", "#2563eb", "circle"),
)


def validate_model_role(uri: str, *, defended: bool) -> None:
    """Verify the model name component, allowing a separate snapshot revision."""
    label = MODEL_NAMES[int(defended)]
    expected = "defended" if defended else "original"
    try:
        actual = model_role(uri)
    except ValueError as error:
        raise ValueError(f"{label} checkpoint has an invalid original_model: {error}") from error
    if actual != expected:
        raise ValueError(f"{label} checkpoint original_model is classified as {actual}: {uri}")


@dataclass(frozen=True)
class Panel:
    layer: int
    coordinates: NDArray[np.float64]
    harmful: NDArray[np.bool_]
    refusal: NDArray[np.float64]
    detector: JsonObject


def load_panel(metrics: JsonObject, directions: bytes, group: str, model_uri: str) -> Panel:
    """Select the saved midpoint winner and verify the source model."""
    import numpy as np

    source = metrics.get("source_manifest", {}).get("model", {}).get("s3_uri")
    if not isinstance(source, str) or parse_s3_uri(source) != parse_s3_uri(model_uri):
        raise ValueError(f"Analysis source model does not match checkpoint snapshot {model_uri}.")
    info = metrics["groups"][group]
    winners = info["best_layers"]["midpoint"]
    if not winners:
        raise ValueError(f"No midpoint best-F1 layer for {group} in {model_uri}.")
    layer = min(winners)
    with np.load(io.BytesIO(directions), allow_pickle=False) as arrays:
        prefix = group + "__"
        coordinates = arrays[prefix + "coordinates"]
        harmful = arrays[prefix + "harmful"]
        refusal = arrays[prefix + "refusal_metric"]
    if (
        harmful.ndim != 1
        or harmful.dtype != np.bool_
        or not harmful.any()
        or harmful.all()
        or refusal.shape != harmful.shape
        or coordinates.ndim != 3
        or coordinates.shape[1:] != (harmful.size, 2)
        or not 0 <= layer < len(coordinates)
        or not np.isfinite(coordinates[layer]).all()
        or not np.isfinite(refusal).all()
    ):
        raise ValueError(f"Invalid saved projection arrays for {group} in {model_uri}.")
    detector = info["refusal_detector"]
    if detector["threshold"] != 0 or detector["comparison"] != ">":
        raise ValueError(f"Expected a strict positive refusal threshold in {model_uri}.")
    return Panel(layer, coordinates[layer], harmful, refusal, detector)


def _range(values: NDArray[np.float64], *, pad: bool = False) -> tuple[float, float]:
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    margin = (high - low) * 0.05 if pad else 0.0
    if low == high:
        margin = 0.5
    return low - margin, high + margin


def comparison_figures(panels: list[Panel]) -> dict[str, Figure]:
    """Create one figure for each plot type. Panels are in row-major order."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if len(panels) != 4:
        raise ValueError("Expected four comparison panels.")
    coordinates = np.concatenate([panel.coordinates for panel in panels])
    all_scores = np.concatenate([panel.refusal for panel in panels])
    score_range = _range(all_scores)
    bin_edges = np.linspace(*score_range, 41)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    figures = {}
    for name in FIGURE_NAMES:
        histogram = name == "refusal_histogram"
        by_score = name == "scatter_by_refusal_metric"
        titles = []
        for index, panel in enumerate(panels):
            row, col = divmod(index, 2)
            title = f"{MODEL_NAMES[col]} · {MODEL_STATES[row]}"
            titles.append(title if histogram else f"{title}<br>Layer {panel.layer}")
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=titles,
            vertical_spacing=0.18,
        )
        histogram_peak = 0.0
        for index, panel in enumerate(panels):
            row, col = divmod(index, 2)
            row, col = row + 1, col + 1
            for harmful, label, color, shape in CLASSES:
                mask = panel.harmful == harmful
                if histogram:
                    counts, _ = np.histogram(panel.refusal[mask], bins=bin_edges)
                    fractions = counts / mask.sum()
                    histogram_peak = max(histogram_peak, float(fractions.max()))
                    trace = go.Bar(
                        x=bin_centers,
                        y=fractions,
                        width=np.diff(bin_edges),
                        marker_color=color,
                        opacity=0.55,
                        name=label,
                        showlegend=index == 0,
                    )
                else:
                    marker = {
                        "color": panel.refusal[mask] if by_score else color,
                        "symbol": shape,
                        "size": 6,
                        "opacity": 0.72,
                    }
                    if by_score:
                        marker["coloraxis"] = "coloraxis"
                    trace = go.Scatter(
                        x=panel.coordinates[mask, 0],
                        y=panel.coordinates[mask, 1],
                        name=label,
                        showlegend=index == 0 and not by_score,
                        mode="markers",
                        marker=marker,
                    )
                figure.add_trace(trace, row=row, col=col)
            figure.add_vline(x=0, line_dash="dash", line_color="#555555", row=row, col=col)
        if histogram:
            x_range = _range(all_scores, pad=True)
            y_range = (0, histogram_peak * 1.08)
            x_title, y_title = "Refusal metric", "Fraction within class"
        else:
            x_range = _range(coordinates[:, 0], pad=True)
            y_range = _range(coordinates[:, 1], pad=True)
            x_title, y_title = "Projection onto mean difference", "Projection onto residual PC1"
        figure.update_xaxes(range=x_range, title_text=x_title, zeroline=False)
        figure.update_yaxes(range=y_range, title_text=y_title, zeroline=False)
        figure.update_layout(
            template="plotly_white",
            width=1060,
            height=760,
            barmode="overlay",
            margin={"l": 90, "r": 120 if by_score else 55, "t": 75, "b": 105},
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.11,
                "yanchor": "top",
            },
            coloraxis={
                "colorscale": "Viridis",
                "cmin": score_range[0],
                "cmax": score_range[1],
                "colorbar": {"title": "Refusal metric", "len": 0.88},
            },
        )
        figures[name] = figure
    return figures


def comparison_tex(groups: dict[str, list[Panel]]) -> str:
    """Compose a standalone report with one table and three figures per group."""
    document = [
        r"""\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{graphicx,booktabs}
\usepackage[hidelinks]{hyperref}
\begin{document}
\section*{Original Model and Defended Model}
The left column shows the Original Model and the right column shows the
Defended Model. The top row shows each model before weight orthogonalization.
The bottom row shows the corresponding orthogonalized snapshot, following
the intervention described by \cite[Sec.~4.1]{arditi2024}.""",
    ]
    projection = (
        r"Each point is a decoder activation at the final token of the formatted input prompt. "
        r"The horizontal axis projects the activation onto the unit harmful-minus-harmless "
        r"mean direction, with zero at the midpoint of the class means. This direction follows "
        r"the difference-of-means method in \cite[Sec.~2.3]{arditi2024}. "
        r"The vertical axis projects the centered activation onto the first principal component "
        r"of the residual after removing its component along that direction. Residual PCA and "
        r"ranking layers by cross-validated midpoint-classifier F1 are choices of this project. "
        r"Each panel uses its own best layer (earliest on ties) and its own fitted axes. "
        r"The common axis limits make the scales comparable across panels."
    )
    score = (
        r"The refusal metric is the log odds that the next token belongs to the Qwen Chat "
        r"refusal set $\{40,2121\}$ \cite[App.~B]{arditi2024}. A positive value means those "
        r"tokens carry more than half the next-token probability mass. This measures a "
        r"proxy for refusal from the first response token."
    )
    captions = (
        r"\textbf{Activation projections colored by input harmfulness.} "
        r"Red triangles mark harmful inputs and blue circles mark harmless inputs. " + projection,
        r"\textbf{Activation projections colored by refusal metric.} "
        r"Triangles mark harmful inputs and circles mark harmless inputs. Color indicates "
        r"the refusal metric on one scale across all panels. " + score + " " + projection,
        r"\textbf{Empirical refusal metric distributions by input harmfulness.} "
        r"Red bars show harmful inputs and blue bars show harmless inputs. Bar height is the "
        r"fraction of examples within that class in each bin. All four panels use the same bin "
        r"edges and axis limits. The dashed line marks zero, and only values to its right "
        r"predict harmful input in the detector. " + score,
    )
    for group, panels in groups.items():
        title = "Non-adversarial" if group == "non_adversarial" else "Adversarial"
        document.extend(
            (
                rf"\section{{{title} prompts}}",
                r"\begin{table}[ht]",
                r"\centering",
                (
                    r"\caption{Refusal as a harmful-input detector. Harmful input is the positive class. "
                    r"An input is predicted harmful when its refusal metric is strictly greater than zero. "
                    r"Precision is the fraction of positive predictions that are harmful, recall is the "
                    r"fraction of harmful inputs detected, and F1 is their harmonic mean.}"
                ),
                r"\begin{tabular}{llrrr}",
                r"\toprule",
                r"Model & Orthogonalization state & Precision & Recall & F1 \\",
                r"\midrule",
            )
        )
        for index, panel in enumerate(panels):
            row, col = divmod(index, 2)
            scores = " & ".join(
                f"{panel.detector[key]:.3f}" for key in ("precision", "recall", "f1")
            )
            document.append(f"{MODEL_NAMES[col]} & {MODEL_STATES[row]} & {scores}" + r" \\")
        document.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
        for name, caption in zip(FIGURE_NAMES, captions, strict=True):
            document.extend(
                (
                    r"\begin{figure}[p]",
                    r"\centering",
                    rf"\includegraphics[width=\linewidth]{{{group}_{name}.png}}",
                    rf"\caption{{{caption}}}",
                    r"\end{figure}",
                )
            )
        document.append(r"\clearpage")
    document.append(
        r"""\begin{thebibliography}{1}
\bibitem{arditi2024}
A. Arditi et al. (2024).
\href{https://arxiv.org/html/2406.11717v3}{Refusal in Language Models Is Mediated by a Single Direction}.
\end{thebibliography}
\end{document}"""
    )
    return "\n".join(document) + "\n"
