"""Developer-facing public API for treefi.

The functions in this module form the stable, dataframe-first entry points for
analyzing fitted tree models across supported backends. They intentionally hide
the backend-specific normalization details while keeping the returned data easy
to inspect, filter, join, and export in normal pandas workflows.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from treefi.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
    ModelAdapter,
    SklearnAdapter,
    XGBoostAdapter,
    get_adapter_for_model,
)
from treefi.exceptions import UnfittedModelError
from treefi.metrics import summarize_interactions
from treefi.results import AnalysisResult, CrossValidatedResult
from treefi.schema import empty_importance_frame, empty_interactions_frame

__all__ = [
    "cross_validated_importance",
    "cross_validated_interactions",
    "feature_importance",
    "feature_interactions",
    "summarize_model",
]

InteractionMode = Literal["ordered", "unordered"]

_ADAPTERS: list[ModelAdapter] = [
    SklearnAdapter(),
    XGBoostAdapter(),
    CatBoostAdapter(),
    LightGBMAdapter(),
]

_IMPORTANCE_COLUMNS = [
    "feature",
    "gain",
    "cover",
    "fscore",
    "weighted_fscore",
    "average_weighted_fscore",
    "average_gain",
    "expected_gain",
    "average_tree_index",
    "average_tree_depth",
    "backend",
    "model_type",
    "occurrence_count",
    "tree_count",
]


def cross_validated_interactions(
    model: object,
    X,
    y,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    cv=None,
    groups=None,
    n_splits: int = 5,
    max_interaction_depth: int = 2,
    max_deepening: int = -1,
    interaction_mode: InteractionMode = "unordered",
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
    min_fscore: int | None = None,
) -> CrossValidatedResult:
    """Return fold-level and aggregated interaction summaries across CV splits."""
    _validate_interaction_mode(interaction_mode)
    _validate_top_k(top_k)

    splitter, task = _resolve_cv_splitter(model, y, cv=cv, n_splits=n_splits)
    resolved_n_splits = _resolve_n_splits(splitter, X, y, fallback=n_splits)
    folds = _collect_cv_frames(
        model=model,
        X=X,
        y=y,
        groups=groups,
        splitter=splitter,
        analysis_fn=lambda fitted_model: feature_interactions(
            fitted_model,
            feature_names=feature_names,
            max_interaction_depth=max_interaction_depth,
            max_deepening=max_deepening,
            interaction_mode=interaction_mode,
            sort_by=sort_by,
            ascending=ascending,
            top_k=top_k,
            min_fscore=min_fscore,
        ),
    )

    if folds.empty:
        return CrossValidatedResult(
            folds=empty_interactions_frame(),
            summary=pd.DataFrame(),
            metadata={"task": task, "splitter": _cv_splitter_name(splitter), "n_splits": resolved_n_splits},
        )

    summary = _aggregate_cv_interactions(folds=folds, n_splits=resolved_n_splits)
    return CrossValidatedResult(
        folds=folds,
        summary=summary,
        interaction_folds=folds,
        interaction_summary=summary,
        metadata={"task": task, "splitter": _cv_splitter_name(splitter), "n_splits": resolved_n_splits},
    )


def cross_validated_importance(
    model: object,
    X,
    y,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    cv=None,
    groups=None,
    n_splits: int = 5,
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
) -> CrossValidatedResult:
    """Return fold-level and aggregated feature-importance summaries across CV splits."""
    _validate_top_k(top_k)

    splitter, task = _resolve_cv_splitter(model, y, cv=cv, n_splits=n_splits)
    resolved_n_splits = _resolve_n_splits(splitter, X, y, fallback=n_splits)
    folds = _collect_cv_frames(
        model=model,
        X=X,
        y=y,
        groups=groups,
        splitter=splitter,
        analysis_fn=lambda fitted_model: feature_importance(
            fitted_model,
            feature_names=feature_names,
            sort_by=sort_by,
            ascending=ascending,
            top_k=top_k,
        ),
    )

    if folds.empty:
        return CrossValidatedResult(
            folds=empty_importance_frame(),
            summary=pd.DataFrame(),
            metadata={"task": task, "splitter": _cv_splitter_name(splitter), "n_splits": resolved_n_splits},
        )

    summary = _aggregate_cv_importance(folds=folds, n_splits=resolved_n_splits)
    return CrossValidatedResult(
        folds=folds,
        summary=summary,
        importance_folds=folds,
        importance_summary=summary,
        metadata={"task": task, "splitter": _cv_splitter_name(splitter), "n_splits": resolved_n_splits},
    )


def feature_importance(
    model: object,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    max_trees: int | None = None,
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return feature-level importance metrics for a fitted model.

    Parameters
    ----------
    model:
        A fitted supported estimator or booster. Supported families currently
        include sklearn trees/forests/HistGradientBoosting, XGBoost, CatBoost,
        and LightGBM.
    feature_names:
        Optional explicit feature names. Use this when the fitted model does
        not preserve the original column names or when you want treefi outputs
        to use a specific naming scheme. The model itself is not mutated.
    max_trees:
        Optional cap on the number of trees normalized from the model. This is
        mainly useful for very large ensembles during exploratory debugging.
    sort_by:
        Optional metric used to order the returned dataframe.
    ascending:
        Sort order applied when ``sort_by`` is provided.
    top_k:
        Optional limit on the number of rows returned after sorting.

    Returns
    -------
    pandas.DataFrame
        One row per feature with the normalized importance metrics defined by
        the public schema.

    Notes
    -----
    Internally, feature importance is derived from the depth-0 interaction view
    so that feature and interaction outputs stay consistent.
    """
    _validate_top_k(top_k)
    analysis_model = _unwrap_model_for_analysis(model)
    adapter = _resolve_adapter(analysis_model)
    ensemble = adapter.to_normalized_ensemble(
        analysis_model,
        feature_names=feature_names,
        max_trees=max_trees,
    )

    frames: list[pd.DataFrame] = []
    for tree in ensemble.trees:
        frame = summarize_interactions(
            tree,
            max_interaction_depth=0,
            sort_by=sort_by,
            ascending=ascending,
            top_k=top_k,
        )
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return empty_importance_frame()

    interactions = pd.concat(frames, ignore_index=True)
    importance = interactions.rename(columns={"interaction": "feature"})[
        [
            "feature",
            "gain",
            "cover",
            "fscore",
            "weighted_fscore",
            "average_weighted_fscore",
            "average_gain",
            "expected_gain",
            "average_tree_index",
            "average_tree_depth",
            "path_frequency",
            "tree_frequency",
        ]
    ].copy()
    importance["backend"] = ensemble.backend
    importance["model_type"] = ensemble.model_type
    importance["occurrence_count"] = importance["path_frequency"]
    importance["tree_count"] = importance["tree_frequency"]
    return importance[_IMPORTANCE_COLUMNS]


def feature_interactions(
    model: object,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    max_interaction_depth: int = 2,
    max_deepening: int = -1,
    max_trees: int | None = None,
    interaction_mode: InteractionMode = "unordered",
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
    min_fscore: int | None = None,
) -> pd.DataFrame:
    """Return interaction metrics for a fitted model as a dataframe.

    Parameters
    ----------
    model:
        A fitted supported estimator or booster.
    feature_names:
        Optional explicit feature names. This is forwarded to the backend
        adapter and does not mutate the input model.
    max_interaction_depth:
        Maximum interaction depth relative to the starting split. ``0`` returns
        single-feature effects, ``1`` includes pairwise path interactions, and
        so on.
    max_deepening:
        Maximum path depth at which an interaction may start. ``-1`` disables
        the constraint.
    max_trees:
        Optional cap on the number of normalized trees.
    interaction_mode:
        ``"unordered"`` collapses interactions to canonical feature sets.
        ``"ordered"`` preserves path order and repeated-feature structure.
    sort_by:
        Optional metric used to order the resulting dataframe.
    ascending:
        Sort order applied when ``sort_by`` is provided.
    top_k:
        Optional limit on returned rows after sorting.
    min_fscore:
        Optional minimum occurrence threshold.

    Returns
    -------
    pandas.DataFrame
        One row per interaction with backend-normalized metrics and metadata.

    Notes
    -----
    This is the main entry point for library users. The result is intended to
    be notebook-friendly and immediately usable for downstream filtering,
    plotting, ranking, or export.
    """
    _validate_interaction_mode(interaction_mode)
    _validate_top_k(top_k)
    analysis_model = _unwrap_model_for_analysis(model)
    adapter = _resolve_adapter(analysis_model)
    ensemble = adapter.to_normalized_ensemble(
        analysis_model,
        feature_names=feature_names,
        max_trees=max_trees,
    )

    frames: list[pd.DataFrame] = []
    for tree in ensemble.trees:
        frame = summarize_interactions(
            tree,
            max_interaction_depth=max_interaction_depth,
            max_deepening=max_deepening,
            interaction_mode=interaction_mode,
            sort_by=sort_by,
            ascending=ascending,
            top_k=top_k,
            min_fscore=min_fscore,
        )
        if frame.empty:
            continue
        frame["backend"] = ensemble.backend
        frame["model_type"] = ensemble.model_type
        frame["occurrence_count"] = frame["path_frequency"]
        frame["tree_count"] = frame["tree_frequency"]
        frame["feature_count"] = frame["interaction_order"]
        frame["path_probability_sum"] = frame["weighted_fscore"]
        frames.append(frame)

    if not frames:
        return empty_interactions_frame()

    return _aggregate_interaction_frames(frames)


def summarize_model(
    model: object | None,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    max_interaction_depth: int = 2,
    max_deepening: int = -1,
    max_trees: int | None = None,
    interaction_mode: InteractionMode = "unordered",
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
    min_fscore: int | None = None,
) -> AnalysisResult:
    """Return a grouped analysis bundle for a model.

    This is the convenience entry point for developers who want all currently
    available tabular outputs in one call. It returns feature interactions,
    feature importance, lightweight leaf statistics, and metadata describing
    the normalized backend/model type.

    Passing ``None`` is supported for tests and empty-result plumbing; the
    function returns an empty ``AnalysisResult`` in that case.
    """
    if model is None:
        return AnalysisResult(
            interactions=empty_interactions_frame(),
            importance=empty_importance_frame(),
        )

    interactions = feature_interactions(
        model,
        feature_names=feature_names,
        max_interaction_depth=max_interaction_depth,
        max_deepening=max_deepening,
        max_trees=max_trees,
        interaction_mode=interaction_mode,
        sort_by=sort_by,
        ascending=ascending,
        top_k=top_k,
        min_fscore=min_fscore,
    )
    importance = feature_importance(
        model,
        feature_names=feature_names,
        max_trees=max_trees,
        sort_by=sort_by,
        ascending=ascending,
        top_k=top_k,
    )

    leaf_stats = None
    metadata: dict[str, Any] = {}
    if not interactions.empty:
        leaf_stats = interactions[["interaction", "leaf_effect_mean", "leaf_effect_var"]].copy()
        metadata["backend"] = interactions.iloc[0]["backend"]
        metadata["model_type"] = interactions.iloc[0]["model_type"]

    return AnalysisResult(
        interactions=interactions,
        importance=importance,
        leaf_stats=leaf_stats,
        metadata=metadata,
    )


def _resolve_adapter(model: object) -> ModelAdapter:
    adapter = get_adapter_for_model(model, adapters=_ADAPTERS)
    if not adapter.is_fitted(model):
        model_type = type(model).__name__
        raise UnfittedModelError(f"treefi requires a fitted model, got unfitted {model_type}")
    return adapter


def _unwrap_model_for_analysis(model: object) -> object:
    steps = getattr(model, "steps", None)
    if steps:
        return steps[-1][1]
    return model


def _validate_interaction_mode(interaction_mode: InteractionMode | str) -> None:
    if interaction_mode not in {"ordered", "unordered"}:
        raise ValueError("interaction_mode must be 'ordered' or 'unordered'")


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")


def _coerce_tabular(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    return pd.DataFrame(X)


def _coerce_target(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    return pd.Series(y)


def _default_cv_splitter(model: object, y, *, n_splits: int):
    task = _infer_task(model, y)
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0), task
    return KFold(n_splits=n_splits, shuffle=True, random_state=0), task


def _resolve_cv_splitter(model: object, y, *, cv, n_splits: int):
    task = _infer_task(model, y)
    if cv is None:
        splitter, _ = _default_cv_splitter(model, y, n_splits=n_splits)
        return splitter, task
    return cv, task


def _infer_task(model: object, y) -> str:
    if is_classifier(model):
        return "classification"
    if is_regressor(model):
        return "regression"
    target_type = type_of_target(y)
    if target_type in {"binary", "multiclass"}:
        return "classification"
    return "regression"


def _aggregate_cv_interactions(*, folds: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    grouped = folds.groupby("interaction", as_index=False, sort=False)
    summary = grouped.agg(
        mean_gain=("gain", "mean"),
        std_gain=("gain", "std"),
        mean_expected_gain=("expected_gain", "mean"),
        std_expected_gain=("expected_gain", "std"),
        mean_cover=("cover", "mean"),
        std_cover=("cover", "std"),
        mean_rank=("rank_consensus", "mean"),
        std_rank=("rank_consensus", "std"),
        mean_tree_frequency=("tree_frequency", "mean"),
        std_tree_frequency=("tree_frequency", "std"),
        mean_path_frequency=("path_frequency", "mean"),
        std_path_frequency=("path_frequency", "std"),
        fold_count=("fold", "nunique"),
        backend=("backend", "first"),
        model_type=("model_type", "first"),
    )
    summary["fold_presence_rate"] = summary["fold_count"] / float(n_splits)
    selection_rate = (
        folds.groupby("interaction", sort=False)["fold"]
        .count()
        .reindex(summary["interaction"])
        .reset_index(drop=True)
        / float(n_splits)
    )
    summary["selection_rate_top_k"] = selection_rate
    return _add_cv_stability_columns(summary)


def _aggregate_cv_importance(*, folds: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    grouped = folds.groupby("feature", as_index=False, sort=False)
    summary = grouped.agg(
        mean_gain=("gain", "mean"),
        std_gain=("gain", "std"),
        mean_expected_gain=("expected_gain", "mean"),
        std_expected_gain=("expected_gain", "std"),
        mean_cover=("cover", "mean"),
        std_cover=("cover", "std"),
        mean_rank=("average_gain", "mean"),
        std_rank=("average_gain", "std"),
        mean_tree_count=("tree_count", "mean"),
        std_tree_count=("tree_count", "std"),
        mean_occurrence_count=("occurrence_count", "mean"),
        std_occurrence_count=("occurrence_count", "std"),
        fold_count=("fold", "nunique"),
        backend=("backend", "first"),
        model_type=("model_type", "first"),
    )
    summary["fold_presence_rate"] = summary["fold_count"] / float(n_splits)
    selection_rate = (
        folds.groupby("feature", sort=False)["fold"]
        .count()
        .reindex(summary["feature"])
        .reset_index(drop=True)
        / float(n_splits)
    )
    summary["selection_rate_top_k"] = selection_rate
    return _add_cv_stability_columns(summary)


def _iter_cv_splits(splitter, X_frame: pd.DataFrame, y_series: pd.Series, groups=None):
    if hasattr(splitter, "split"):
        if groups is not None:
            return splitter.split(X_frame, y_series, groups)
        return splitter.split(X_frame, y_series)
    return iter(splitter)


def _cv_splitter_name(splitter) -> str:
    if hasattr(splitter, "split"):
        return type(splitter).__name__
    return "custom"


def _resolve_n_splits(splitter, X, y, *, fallback: int) -> int:
    if hasattr(splitter, "get_n_splits"):
        return int(splitter.get_n_splits(X, y))
    return fallback


def _collect_cv_frames(*, model: object, X, y, groups, splitter, analysis_fn) -> pd.DataFrame:
    X_frame = _coerce_tabular(X)
    y_series = _coerce_target(y)
    groups_series = _coerce_target(groups) if groups is not None else None
    fold_frames: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(_iter_cv_splits(splitter, X_frame, y_series, groups_series)):
        fitted_model = clone(model)
        X_train = X_frame.iloc[train_index]
        y_train = y_series.iloc[train_index]
        fitted_model.fit(X_train, y_train)
        frame = analysis_fn(fitted_model).copy()
        if frame.empty:
            continue
        frame["fold"] = fold
        frame["train_size"] = len(train_index)
        frame["test_size"] = len(test_index)
        fold_frames.append(frame)
    if not fold_frames:
        return pd.DataFrame()
    return pd.concat(fold_frames, ignore_index=True)


def _add_cv_stability_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["std_gain"] = summary["std_gain"].fillna(0.0)
    summary["std_expected_gain"] = summary["std_expected_gain"].fillna(0.0)
    summary["std_rank"] = summary["std_rank"].fillna(0.0)
    summary["gain_cv"] = _safe_cv(summary["std_gain"], summary["mean_gain"])
    summary["expected_gain_cv"] = _safe_cv(summary["std_expected_gain"], summary["mean_expected_gain"])
    summary["rank_stability_score"] = 1.0 / (1.0 + summary["std_rank"])
    summary["consensus_top_k"] = summary["selection_rate_top_k"] >= 0.8
    summary["rare_fold_flag"] = summary["fold_presence_rate"] < 0.5
    summary["overfit_suspect_flag"] = (
        (summary["fold_presence_rate"] < 0.5)
        & (summary["gain_cv"] > 1.0)
        & (summary["mean_gain"] > summary["mean_gain"].median())
    )
    return summary


def _safe_cv(std_series: pd.Series, mean_series: pd.Series) -> pd.Series:
    mean_abs = mean_series.abs()
    nonzero = mean_abs > 0
    result = pd.Series(0.0, index=std_series.index, dtype="float64")
    result.loc[nonzero] = std_series.loc[nonzero] / mean_abs.loc[nonzero]
    return result


def _aggregate_interaction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    numeric_sum_columns = {
        "gain",
        "cover",
        "fscore",
        "weighted_fscore",
        "expected_gain",
        "path_frequency",
        "occurrence_count",
        "path_probability_sum",
        "tree_count",
    }
    numeric_mean_columns = {
        "average_weighted_fscore",
        "average_gain",
        "average_tree_index",
        "average_tree_depth",
        "tree_frequency",
        "first_position_mean",
        "min_depth",
        "max_depth",
        "leaf_effect_mean",
        "leaf_effect_var",
        "rank_gain",
        "rank_fscore",
        "rank_expected_gain",
        "rank_consensus",
        "feature_count",
        "interaction_order",
    }
    grouped = combined.groupby("interaction", as_index=False, sort=False)
    aggregation = {
        column: (
            "sum"
            if column in numeric_sum_columns
            else "mean"
            if column in numeric_mean_columns
            else "first"
        )
        for column in combined.columns
        if column != "interaction"
    }
    return grouped.agg(aggregation)
