import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import treefi


def test_feature_importance_returns_dataframe_for_sklearn_decision_tree() -> None:
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    frame = treefi.feature_importance(model)

    assert frame["feature"].tolist() == ["f0"]
    assert frame.iloc[0]["gain"] == 1.0
    assert frame.iloc[0]["cover"] == 4.0
    assert "gain" in frame.columns
    assert "fscore" in frame.columns


def test_feature_interactions_returns_dataframe_for_sklearn_decision_tree() -> None:
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=0)

    assert frame["interaction"].tolist() == ["f0"]
    assert frame.iloc[0]["gain"] == 1.0
    assert frame.iloc[0]["cover"] == 4.0
    assert "gain" in frame.columns
    assert "interaction_order" in frame.columns


def test_feature_interactions_aggregates_across_sklearn_forest_trees() -> None:
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    model = RandomForestRegressor(n_estimators=2, max_depth=1, random_state=0).fit(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=0)

    assert frame["interaction"].tolist() == ["f0"]
    assert frame.iloc[0]["gain"] > 0.0
    assert frame.iloc[0]["cover"] > 0.0
    assert frame.iloc[0]["fscore"] == 2
    assert frame.iloc[0]["tree_count"] == 2.0


def test_feature_interactions_returns_dataframe_for_xgboost_booster() -> None:
    dtrain = xgb.DMatrix([[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["f0"])
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 1, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=1,
    )

    frame = treefi.feature_interactions(booster, max_interaction_depth=0)

    assert frame["interaction"].tolist() == ["f0"]
    assert frame.iloc[0]["backend"] == "xgboost"


def test_feature_interactions_returns_dataframe_for_catboost_model() -> None:
    model = CatBoostRegressor(depth=1, learning_rate=1.0, iterations=1, verbose=False)
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0])

    frame = treefi.feature_interactions(model, max_interaction_depth=0)

    assert frame["interaction"].tolist() == ["f0"]
    assert frame.iloc[0]["backend"] == "catboost"
    assert frame.iloc[0]["gain"] > 0.0
    assert frame.iloc[0]["cover"] == 4.0


def test_feature_interactions_returns_dataframe_for_hist_gradient_boosting() -> None:
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    model = HistGradientBoostingRegressor(
        max_depth=2,
        max_iter=5,
        min_samples_leaf=1,
        learning_rate=1.0,
        random_state=0,
    ).fit(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=0)

    assert not frame.empty
    assert frame.iloc[0]["backend"] == "sklearn"
    assert frame.iloc[0]["gain"] > 0.0
    assert frame.iloc[0]["cover"] > 0.0


def test_feature_interactions_returns_dataframe_for_lightgbm_booster() -> None:
    X = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    train = lgb.Dataset(pd.DataFrame({"f0": X}), label=y)
    booster = lgb.train(
        {
            "objective": "regression",
            "max_depth": 2,
            "num_leaves": 4,
            "learning_rate": 1.0,
            "min_data_in_leaf": 1,
            "min_sum_hessian_in_leaf": 0.0,
            "verbose": -1,
        },
        train_set=train,
        num_boost_round=2,
    )

    frame = treefi.feature_interactions(booster, max_interaction_depth=0)

    assert not frame.empty
    assert frame.iloc[0]["backend"] == "lightgbm"
