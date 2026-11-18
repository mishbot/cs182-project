"""CPU activation geometry and leakage-free out-of-fold classification."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._types import JsonObject
from .activation_runs import ActivationRun

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
Classifier = Literal["logistic", "midpoint"]


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class LayerProjection:
    harmful_mean: NDArray[np.float64]
    harmless_mean: NDArray[np.float64]
    difference: NDArray[np.float64]
    direction: NDArray[np.float64]
    orthogonal_pc1: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    residual_explained_variance_ratio: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float | None
    rmse: float
    mae: float


@dataclass(frozen=True)
class LinearFit:
    coefficient: NDArray[np.float64]
    intercept: float
    direction: NDArray[np.float64]
    valid: bool


@dataclass(frozen=True)
class LayerAnalysis:
    layer: int
    projection: LayerProjection
    logistic: BinaryMetrics
    midpoint: BinaryMetrics | None
    logistic_predictions: NDArray[np.bool_]
    midpoint_predictions: NDArray[np.bool_]
    midpoint_valid: NDArray[np.bool_]
    notes: tuple[str, ...]
    logistic_fit: LinearFit
    refusal_fit: LinearFit
    refusal_regression: RegressionMetrics
    refusal_predictions: NDArray[np.float64]


@dataclass(frozen=True)
class ActivationAnalysis:
    manifest_uri: str
    manifest_sha256: str
    manifest: JsonObject
    source_row_indices: NDArray[np.int64]
    prompts: list[str]
    harmful: NDArray[np.bool_]
    fold_ids: NDArray[np.int64]
    layers: list[LayerAnalysis]
    folds: int
    seed: int
    group: str
    refusal_metric: NDArray[np.float64]


@dataclass(frozen=True)
class ActivationAnalyses:
    run: ActivationRun
    groups: tuple[ActivationAnalysis, ...]


@dataclass(frozen=True)
class SelectedDirection:
    name: str
    layer: int
    vector: NDArray[np.float64]
    valid: bool


def linear_fit(coefficient: NDArray[np.float64], intercept: float) -> LinearFit:
    """Represent fitted coefficients in raw activation coordinates, with a unit direction."""
    import numpy as np

    norm = float(np.linalg.norm(coefficient))
    valid = norm > 0 and bool(np.isfinite(coefficient).all())
    return LinearFit(
        coefficient, intercept, coefficient / norm if valid else np.zeros_like(coefficient), valid
    )


def refusal_metric(
    log_refusal: NDArray[np.float64], log_nonrefusal: NDArray[np.float64]
) -> NDArray[np.float64]:
    import numpy as np

    if (
        log_refusal.shape != log_nonrefusal.shape
        or log_refusal.ndim != 1
        or not np.isfinite(log_refusal).all()
        or not np.isfinite(log_nonrefusal).all()
        or not np.allclose(np.logaddexp(log_refusal, log_nonrefusal), 0, atol=2e-5)
    ):
        raise ValueError("Expected normalized finite refusal and nonrefusal log masses.")
    return log_refusal - log_nonrefusal


def classification_metrics(
    labels: NDArray[np.bool_],
    predictions: NDArray[np.bool_],
) -> BinaryMetrics:
    """Harmful is the positive class; zero-denominator metrics are zero."""
    import numpy as np

    if labels.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must be matching one-dimensional arrays.")
    tp = int(np.count_nonzero(labels & predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    return BinaryMetrics(
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        tp,
        fp,
        tn,
        fn,
    )


def class_means(
    features: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("Expected features [rows, hidden] and matching boolean labels.")
    if not labels.any() or labels.all():
        raise ValueError("Both harmful and harmless examples are required.")
    return features[labels].mean(axis=0), features[~labels].mean(axis=0)


def project_layer(
    features: NDArray[np.float64],
    labels: NDArray[np.bool_],
) -> LayerProjection:
    """Descriptive full-run geometry, independent of classifier evaluation."""
    import numpy as np
    from sklearn.decomposition import PCA

    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    harmful, harmless = class_means(features, labels)
    difference = harmful - harmless
    norm = float(np.linalg.norm(difference))
    notes: list[str] = []
    if norm == 0:
        direction = np.zeros_like(difference)
        notes.append(
            "Zero mean difference: the horizontal direction is undefined and shown as zero."
        )
    else:
        direction = difference / norm
    centered = features - features.mean(axis=0)
    residual = centered - (centered @ direction)[:, None] * direction
    energy = float(np.square(residual).sum())
    original_energy = float(np.square(centered).sum())
    pc = np.zeros_like(direction)
    ratio = 0.0
    if energy <= np.finfo(np.float64).eps * original_energy:
        notes.append("No residual variance: the orthogonal PC1 axis is shown as zero.")
    else:
        pca = PCA(n_components=1, svd_solver="full").fit(residual)
        components = pca.components_
        assert components is not None  # A successful fit supplies components.
        pc = np.asarray(components[0], dtype=np.float64)
        pc = pc - (pc @ direction) * direction
        length = float(np.linalg.norm(pc))
        if length == 0:
            notes.append(
                "No numerically defined orthogonal PC1; the vertical axis is shown as zero."
            )
        else:
            pc /= length
            if pc[int(np.argmax(np.abs(pc)))] < 0:
                pc = -pc
            ratio = float(pca.explained_variance_ratio_[0])
    coordinates = np.column_stack(
        ((features - (harmful + harmless) / 2) @ direction, centered @ pc)
    )
    return LayerProjection(
        harmful, harmless, difference, direction, pc, coordinates, ratio, tuple(notes)
    )


def midpoint_predict(
    training: NDArray[np.float64],
    training_labels: NDArray[np.bool_],
    testing: NDArray[np.float64],
) -> NDArray[np.bool_] | None:
    """Return None when the training means give no defined hyperplane."""
    import numpy as np

    harmful, harmless = class_means(training, training_labels)
    difference = harmful - harmless
    if float(np.linalg.norm(difference)) == 0:
        return None
    return (testing - (harmful + harmless) / 2) @ difference >= 0


def _evaluate_layer(
    layer: int,
    features: NDArray[np.float64],
    labels: NDArray[np.bool_],
    partitions: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    seed: int,
    targets: NDArray[np.float64],
) -> LayerAnalysis:
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    logistic_predictions = np.zeros(labels.size, dtype=np.bool_)
    midpoint_predictions = np.zeros(labels.size, dtype=np.bool_)
    midpoint_valid = np.ones(labels.size, dtype=np.bool_)
    regression_predictions = np.empty(labels.size, dtype=np.float64)
    notes: list[str] = []
    # The final pass refits both models on all subgroup rows for direction export.
    all_rows = np.arange(labels.size, dtype=np.int64)
    fits: list[LinearFit] = []
    for fold, (train, test) in enumerate([*partitions, (all_rows, all_rows)]):
        scaler = StandardScaler().fit(features[train])
        train_features = scaler.transform(features[train])
        logistic = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                logistic.fit(train_features, labels[train])
            except ConvergenceWarning as error:
                raise ValueError(
                    f"Logistic regression did not converge at layer {layer}, fold {fold}."
                ) from error
        regression = Ridge(alpha=1.0, solver="lsqr", tol=1e-8).fit(train_features, targets[train])
        if fold == len(partitions):
            for coefficient, intercept in [
                (logistic.coef_[0], np.asarray(logistic.intercept_).reshape(-1)[0]),
                (regression.coef_, regression.intercept_),
            ]:
                assert scaler.scale_ is not None and scaler.mean_ is not None
                raw = np.asarray(coefficient / scaler.scale_, dtype=np.float64)
                fits.append(linear_fit(raw, float(intercept - raw @ scaler.mean_)))
            break
        test_features = scaler.transform(features[test])
        logistic_predictions[test] = logistic.predict(test_features)
        regression_predictions[test] = regression.predict(test_features)
        predictions = midpoint_predict(features[train], labels[train], features[test])
        if predictions is None:
            midpoint_valid[test] = False
            notes.append(
                f"Midpoint classifier undefined in CV fold {fold}: identical training means."
            )
        else:
            midpoint_predictions[test] = predictions
    midpoint_metrics = (
        classification_metrics(labels, midpoint_predictions) if midpoint_valid.all() else None
    )
    constant_target = float(np.var(targets)) == 0
    if constant_target:
        notes.append(
            "Constant refusal metric: regression R² is undefined and excluded from ranking."
        )
    return LayerAnalysis(
        layer,
        project_layer(features, labels),
        classification_metrics(labels, logistic_predictions),
        midpoint_metrics,
        logistic_predictions,
        midpoint_predictions,
        midpoint_valid,
        tuple(notes),
        fits[0],
        fits[1],
        RegressionMetrics(
            None if constant_target else float(r2_score(targets, regression_predictions)),
            float(mean_squared_error(targets, regression_predictions)) ** 0.5,
            float(mean_absolute_error(targets, regression_predictions)),
        ),
        regression_predictions,
    )


def _analyze_group(run: ActivationRun, group: str, *, folds: int, seed: int) -> ActivationAnalysis:
    """Use identical stratified folds for every layer and both classifiers."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    targets = refusal_metric(run.refusal_log_probability, run.nonrefusal_log_probability)
    labels = run.harmful
    values = run.activations
    if (
        values.ndim != 3
        or labels.shape != (values.shape[0],)
        or labels.dtype != np.bool_
        or len(run.prompts) != labels.size
        or run.source_row_indices.shape != labels.shape
        or values.shape[1] < 1
        or values.shape[2] < 1
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Expected finite activations [rows, layers, hidden] and matching metadata."
        )
    counts = [int((~labels).sum()), int(labels.sum())]
    if folds < 2 or min(counts) < folds:
        raise ValueError(
            f"--folds must be at least 2 and no greater than either class count ({counts}). Use fewer folds for a small run."
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    partitions = list(splitter.split(np.zeros(labels.size), labels))
    fold_ids = np.empty(labels.size, dtype=np.int64)
    for fold, (_, test) in enumerate(partitions):
        fold_ids[test] = fold
    layers: list[LayerAnalysis] = []
    for layer in range(values.shape[1]):
        logger.info("Analyzing layer %s of %s", layer, values.shape[1] - 1)
        features = np.asarray(values[:, layer, :], dtype=np.float64)
        layers.append(_evaluate_layer(layer, features, labels, partitions, seed, targets))
    return ActivationAnalysis(
        run.manifest_uri,
        run.manifest_sha256,
        run.manifest,
        run.source_row_indices,
        run.prompts,
        labels,
        fold_ids,
        layers,
        folds,
        seed,
        group,
        targets,
    )


def best_layers(analysis: ActivationAnalysis, classifier: Classifier) -> list[int]:
    """Return every exact tie for the highest pooled out-of-fold F1."""
    scores = [(layer.layer, getattr(layer, classifier)) for layer in analysis.layers]
    available = [(layer, metrics.f1) for layer, metrics in scores if metrics is not None]
    if not available:
        return []
    maximum = max(score for _, score in available)
    return [layer for layer, score in available if score == maximum]


def analyze_activations(
    run: ActivationRun, *, folds: int = 5, seed: int = 0, include_adversarial: bool = False
) -> ActivationAnalyses:
    """Evaluate adversarial and non-adversarial rows independently; reserve validation data."""
    from dataclasses import replace

    import numpy as np

    if (
        run.activations.ndim != 3
        or run.harmful.ndim != 1
        or run.activations.shape[0] != run.harmful.size
        or run.source_row_indices.shape != run.harmful.shape
        or len(run.prompts) != run.harmful.size
        or run.refusal_log_probability.shape != run.harmful.shape
        or run.nonrefusal_log_probability.shape != run.harmful.shape
    ):
        raise ValueError("Activation arrays and row metadata must have matching dimensions.")
    if run.adversarial.shape != run.harmful.shape or run.adversarial.dtype != np.bool_:
        raise ValueError("Expected a boolean adversarial column matching the run rows.")
    groups: list[ActivationAnalysis] = []
    for flag, name in [(False, "non_adversarial"), (True, "adversarial")]:
        if flag and not include_adversarial:
            continue
        selected = np.flatnonzero(run.adversarial == flag)
        subset = replace(
            run,
            source_row_indices=run.source_row_indices[selected],
            prompts=[run.prompts[i] for i in selected],
            harmful=run.harmful[selected],
            activations=run.activations[selected],
            adversarial=run.adversarial[selected],
            refusal_log_probability=run.refusal_log_probability[selected],
            nonrefusal_log_probability=run.nonrefusal_log_probability[selected],
            top_token_ids=run.top_token_ids[selected],
            top_token_probabilities=run.top_token_probabilities[selected],
        )
        try:
            groups.append(_analyze_group(subset, name, folds=folds, seed=seed))
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    return ActivationAnalyses(run, tuple(groups))


def best_regression_layers(analysis: ActivationAnalysis) -> list[int]:
    scores = [
        (layer.layer, layer.refusal_regression.r2)
        for layer in analysis.layers
        if layer.refusal_regression.r2 is not None
    ]
    if not scores:
        return []
    maximum = max(score for _, score in scores)
    return [layer for layer, score in scores if score == maximum]


def selected_directions(analysis: ActivationAnalyses) -> list[SelectedDirection]:
    import numpy as np

    selected: list[SelectedDirection] = []
    for group in analysis.groups:
        for method in ("logistic", "midpoint", "refusal"):
            winners = (
                best_regression_layers(group) if method == "refusal" else best_layers(group, method)
            )
            if not winners:
                selected.append(
                    SelectedDirection(
                        f"{group.group}_{method}",
                        -1,
                        np.zeros_like(group.layers[0].projection.direction),
                        False,
                    )
                )
                continue
            layer = group.layers[winners[0]]
            fit = (
                layer.logistic_fit
                if method == "logistic"
                else layer.refusal_fit
                if method == "refusal"
                else linear_fit(layer.projection.difference, 0)
            )
            selected.append(
                SelectedDirection(f"{group.group}_{method}", layer.layer, fit.direction, fit.valid)
            )
    return selected
