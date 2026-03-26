from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import pandas as pd
import pytest
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import treefi
from treefi.adapters import CatBoostAdapter, LightGBMAdapter, SklearnAdapter, XGBoostAdapter

RegressionBuilder = Callable[[pd.DataFrame, pd.Series], object]
ClassificationBuilder = Callable[[pd.DataFrame, pd.Series], object]


def _xgb_regressor_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=1,
        n_estimators=1,
        learning_rate=0.3,
        random_state=0,
    )
    return model.fit(X, y)


def _xgb_regressor_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=5,
        learning_rate=0.2,
        random_state=0,
    )
    return model.fit(X, y)


def _xgb_classifier_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=1,
        n_estimators=1,
        learning_rate=0.3,
        eval_metric="logloss",
        random_state=0,
    )
    return model.fit(X, y)


def _xgb_classifier_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        n_estimators=5,
        learning_rate=0.2,
        eval_metric="logloss",
        random_state=0,
    )
    return model.fit(X, y)


def _cat_regressor_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = CatBoostRegressor(depth=1, iterations=1, learning_rate=0.3, verbose=False)
    return model.fit(X, y)


def _cat_regressor_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = CatBoostRegressor(depth=3, iterations=5, learning_rate=0.2, verbose=False)
    return model.fit(X, y)


def _cat_classifier_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = CatBoostClassifier(depth=1, iterations=1, learning_rate=0.3, verbose=False)
    return model.fit(X, y)


def _cat_classifier_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = CatBoostClassifier(depth=3, iterations=5, learning_rate=0.2, verbose=False)
    return model.fit(X, y)


def _lgbm_regressor_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = lgb.LGBMRegressor(max_depth=1, n_estimators=1, learning_rate=0.3, min_child_samples=1, verbose=-1)
    return model.fit(X, y)


def _lgbm_regressor_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = lgb.LGBMRegressor(max_depth=3, n_estimators=5, learning_rate=0.2, min_child_samples=1, verbose=-1)
    return model.fit(X, y)


def _lgbm_classifier_stump(X: pd.DataFrame, y: pd.Series) -> object:
    model = lgb.LGBMClassifier(max_depth=1, n_estimators=1, learning_rate=0.3, min_child_samples=1, verbose=-1)
    return model.fit(X, y)


def _lgbm_classifier_large(X: pd.DataFrame, y: pd.Series) -> object:
    model = lgb.LGBMClassifier(max_depth=3, n_estimators=5, learning_rate=0.2, min_child_samples=1, verbose=-1)
    return model.fit(X, y)


REGRESSION_CASES: list[tuple[str, object, RegressionBuilder]] = [
    ("sklearn-dt-stump", SklearnAdapter(), lambda X, y: DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)),
    ("sklearn-dt-large", SklearnAdapter(), lambda X, y: DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)),
    ("sklearn-rf-stump", SklearnAdapter(), lambda X, y: RandomForestRegressor(n_estimators=3, max_depth=1, random_state=0).fit(X, y)),
    ("sklearn-rf-large", SklearnAdapter(), lambda X, y: RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X, y)),
    ("histgb-stump", SklearnAdapter(), lambda X, y: HistGradientBoostingRegressor(max_depth=1, max_iter=2, min_samples_leaf=5, learning_rate=0.3, random_state=0).fit(X, y)),
    ("histgb-large", SklearnAdapter(), lambda X, y: HistGradientBoostingRegressor(max_depth=4, max_iter=10, min_samples_leaf=5, learning_rate=0.2, random_state=0).fit(X, y)),
    ("xgb-stump", XGBoostAdapter(), _xgb_regressor_stump),
    ("xgb-large", XGBoostAdapter(), _xgb_regressor_large),
    ("cat-stump", CatBoostAdapter(), _cat_regressor_stump),
    ("cat-large", CatBoostAdapter(), _cat_regressor_large),
    ("lgbm-stump", LightGBMAdapter(), _lgbm_regressor_stump),
    ("lgbm-large", LightGBMAdapter(), _lgbm_regressor_large),
]


CLASSIFICATION_CASES: list[tuple[str, object, ClassificationBuilder]] = [
    ("sklearn-dt-stump", SklearnAdapter(), lambda X, y: DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)),
    ("sklearn-dt-large", SklearnAdapter(), lambda X, y: DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)),
    ("sklearn-rf-stump", SklearnAdapter(), lambda X, y: RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0).fit(X, y)),
    ("sklearn-rf-large", SklearnAdapter(), lambda X, y: RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(X, y)),
    ("histgb-stump", SklearnAdapter(), lambda X, y: HistGradientBoostingClassifier(max_depth=1, max_iter=2, min_samples_leaf=5, learning_rate=0.3, random_state=0).fit(X, y)),
    ("histgb-large", SklearnAdapter(), lambda X, y: HistGradientBoostingClassifier(max_depth=4, max_iter=10, min_samples_leaf=5, learning_rate=0.2, random_state=0).fit(X, y)),
    ("xgb-stump", XGBoostAdapter(), _xgb_classifier_stump),
    ("xgb-large", XGBoostAdapter(), _xgb_classifier_large),
    ("cat-stump", CatBoostAdapter(), _cat_classifier_stump),
    ("cat-large", CatBoostAdapter(), _cat_classifier_large),
    ("lgbm-stump", LightGBMAdapter(), _lgbm_classifier_stump),
    ("lgbm-large", LightGBMAdapter(), _lgbm_classifier_large),
]


@pytest.mark.parametrize(("case_name", "adapter", "builder"), REGRESSION_CASES, ids=[case[0] for case in REGRESSION_CASES])
def test_regression_adapters_handle_realistic_dataset(
    case_name: str,
    adapter,
    builder: RegressionBuilder,
    regression_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = regression_dataset
    model = builder(X, y)

    ensemble = adapter.to_normalized_ensemble(model)
    frame = treefi.feature_interactions(model, max_interaction_depth=1)

    assert ensemble.trees
    assert frame is not None
    assert "backend" in frame.columns


@pytest.mark.parametrize(("case_name", "adapter", "builder"), CLASSIFICATION_CASES, ids=[case[0] for case in CLASSIFICATION_CASES])
def test_classification_adapters_handle_realistic_dataset(
    case_name: str,
    adapter,
    builder: ClassificationBuilder,
    classification_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_dataset
    model = builder(X, y)

    ensemble = adapter.to_normalized_ensemble(model)
    frame = treefi.feature_interactions(model, max_interaction_depth=1)

    assert ensemble.trees
    assert frame is not None
    assert "backend" in frame.columns


@pytest.mark.parametrize(
    ("case_name", "builder"),
    [
        ("sklearn-dt-large", lambda X, y: DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)),
        (
            "sklearn-rf-large",
            lambda X, y: RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X, y),
        ),
        (
            "histgb-large",
            lambda X, y: HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=10,
                min_samples_leaf=5,
                learning_rate=0.2,
                random_state=0,
            ).fit(X, y),
        ),
        ("cat-large", _cat_regressor_large),
    ],
    ids=["sklearn-dt-large", "sklearn-rf-large", "histgb-large", "cat-large"],
)
def test_realistic_regression_models_expose_positive_gain_and_cover(
    case_name: str,
    builder: RegressionBuilder,
    regression_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = regression_dataset
    model = builder(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=0)

    assert not frame.empty
    assert frame["gain"].notna().all()
    assert frame["cover"].notna().all()
    assert (frame["gain"] >= 0.0).all()
    assert (frame["cover"] > 0.0).all()


@pytest.mark.parametrize(
    ("case_name", "builder"),
    [
        ("sklearn-rf-large", lambda X, y: RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0).fit(X, y)),
        ("histgb-large", lambda X, y: HistGradientBoostingRegressor(max_depth=4, max_iter=10, min_samples_leaf=5, learning_rate=0.2, random_state=0).fit(X, y)),
        ("xgb-large", _xgb_regressor_large),
        ("cat-large", _cat_regressor_large),
        ("lgbm-large", _lgbm_regressor_large),
    ],
    ids=["sklearn-rf-large", "histgb-large", "xgb-large", "cat-large", "lgbm-large"],
)
def test_feature_importance_returns_one_row_per_feature_after_aggregation(
    case_name: str,
    builder: RegressionBuilder,
    regression_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = regression_dataset
    model = builder(X, y)

    frame = treefi.feature_importance(model, top_k=50)

    assert not frame.empty
    assert frame["feature"].nunique() == len(frame)
