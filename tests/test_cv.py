from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import treefi
from treefi.results import CrossValidatedResult


def test_cross_validated_interactions_uses_regression_default_splitter() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_interactions(model, X, y, n_splits=3, top_k=10)

    assert isinstance(result, CrossValidatedResult)
    assert isinstance(result.folds, pd.DataFrame)
    assert isinstance(result.summary, pd.DataFrame)
    assert result.metadata["task"] == "regression"
    assert result.metadata["splitter"] == "KFold"
    assert result.metadata["n_splits"] == 3
    assert set(result.folds["fold"]) == {0, 1, 2}
    assert (result.folds["train_size"] > 0).all()
    assert (result.folds["test_size"] > 0).all()
    assert "fold_presence_rate" in result.summary.columns
    assert "mean_gain" in result.summary.columns


def test_cross_validated_interactions_uses_classification_default_splitter() -> None:
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame[cancer.feature_names].iloc[:180].reset_index(drop=True)
    y = cancer.frame[cancer.target.name].iloc[:180].reset_index(drop=True)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_interactions(model, X, y, n_splits=4, top_k=10)

    assert result.metadata["task"] == "classification"
    assert result.metadata["splitter"] == "StratifiedKFold"
    assert result.metadata["n_splits"] == 4
    assert set(result.folds["fold"]) == {0, 1, 2, 3}
    assert not result.summary.empty
    assert (result.summary["fold_presence_rate"] > 0.0).all()


def test_cross_validated_interactions_accepts_explicit_cv_override() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    cv = KFold(n_splits=4, shuffle=True, random_state=7)

    result = treefi.cross_validated_interactions(model, X, y, cv=cv, top_k=10)

    assert result.metadata["splitter"] == "KFold"
    assert result.metadata["n_splits"] == 4
    assert set(result.folds["fold"]) == {0, 1, 2, 3}


def test_cross_validated_importance_returns_fold_and_summary_frames() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_importance(model, X, y, n_splits=3, top_k=10)

    assert isinstance(result, CrossValidatedResult)
    assert isinstance(result.folds, pd.DataFrame)
    assert isinstance(result.summary, pd.DataFrame)
    assert result.metadata["task"] == "regression"
    assert set(result.folds["fold"]) == {0, 1, 2}
    assert "feature" in result.folds.columns
    assert "mean_gain" in result.summary.columns
    assert "fold_presence_rate" in result.summary.columns
    assert "mean_cover" in result.summary.columns


def test_cross_validated_importance_populates_explicit_importance_fields() -> None:
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame[cancer.feature_names].iloc[:180].reset_index(drop=True)
    y = cancer.frame[cancer.target.name].iloc[:180].reset_index(drop=True)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_importance(model, X, y, n_splits=4, top_k=10)

    assert result.importance_folds is not None
    assert result.importance_summary is not None
    assert not result.importance_folds.empty
    assert not result.importance_summary.empty
    assert result.folds is result.importance_folds
    assert result.summary is result.importance_summary


def test_cross_validated_interactions_add_stability_metrics_and_flags() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_interactions(model, X, y, n_splits=3, top_k=5)

    summary = result.summary
    assert "selection_rate_top_k" in summary.columns
    assert "gain_cv" in summary.columns
    assert "expected_gain_cv" in summary.columns
    assert "rank_stability_score" in summary.columns
    assert "consensus_top_k" in summary.columns
    assert "rare_fold_flag" in summary.columns
    assert "overfit_suspect_flag" in summary.columns
    assert ((summary["selection_rate_top_k"] >= 0.0) & (summary["selection_rate_top_k"] <= 1.0)).all()
    assert summary["consensus_top_k"].isin([True, False]).all()
    assert summary["rare_fold_flag"].isin([True, False]).all()
    assert summary["overfit_suspect_flag"].isin([True, False]).all()


def test_cross_validated_importance_add_stability_metrics_and_flags() -> None:
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame[cancer.feature_names].iloc[:180].reset_index(drop=True)
    y = cancer.frame[cancer.target.name].iloc[:180].reset_index(drop=True)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_importance(model, X, y, n_splits=4, top_k=5)

    summary = result.summary
    assert "selection_rate_top_k" in summary.columns
    assert "gain_cv" in summary.columns
    assert "expected_gain_cv" in summary.columns
    assert "rank_stability_score" in summary.columns
    assert "consensus_top_k" in summary.columns
    assert "rare_fold_flag" in summary.columns
    assert "overfit_suspect_flag" in summary.columns


def test_cross_validated_results_keep_explicit_grouped_frames() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:100].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:100].reset_index(drop=True)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)

    interaction_result = treefi.cross_validated_interactions(model, X, y, n_splits=3, top_k=5)
    importance_result = treefi.cross_validated_importance(model, X, y, n_splits=3, top_k=5)

    assert interaction_result.interaction_folds is not None
    assert interaction_result.interaction_summary is not None
    assert importance_result.importance_folds is not None
    assert importance_result.importance_summary is not None


def test_cross_validated_interactions_support_sklearn_pipeline() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)),
        ]
    )

    result = treefi.cross_validated_interactions(pipeline, X, y, n_splits=3, top_k=5)

    assert not result.folds.empty
    assert result.metadata["task"] == "regression"


def test_cross_validated_interactions_pass_groups_to_group_splitter() -> None:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.frame[diabetes.feature_names].iloc[:120].reset_index(drop=True)
    y = diabetes.frame[diabetes.target.name].iloc[:120].reset_index(drop=True)
    groups = pd.Series([index // 10 for index in range(len(X))])
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_interactions(
        model,
        X,
        y,
        cv=GroupKFold(n_splits=3),
        groups=groups,
        top_k=5,
    )

    assert result.metadata["splitter"] == "GroupKFold"
    assert result.metadata["n_splits"] == 3
    assert set(result.folds["fold"]) == {0, 1, 2}


def test_cross_validated_importance_pass_groups_to_group_splitter() -> None:
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame[cancer.feature_names].iloc[:120].reset_index(drop=True)
    y = cancer.frame[cancer.target.name].iloc[:120].reset_index(drop=True)
    groups = pd.Series([index // 8 for index in range(len(X))])
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)

    result = treefi.cross_validated_importance(
        model,
        X,
        y,
        cv=GroupKFold(n_splits=3),
        groups=groups,
        top_k=5,
    )

    assert result.metadata["splitter"] == "GroupKFold"
    assert result.metadata["n_splits"] == 3
    assert set(result.folds["fold"]) == {0, 1, 2}
