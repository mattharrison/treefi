from __future__ import annotations

import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import treefi


def test_titanic_xgboost_interactions_smoke(titanic_frame) -> None:
    X = titanic_frame.drop(columns=["survived"])
    y = titanic_frame["survived"]
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 2, "eta": 0.5, "seed": 0},
        dtrain=dtrain,
        num_boost_round=2,
    )

    frame = treefi.feature_interactions(booster, max_interaction_depth=1, sort_by="gain")

    assert not frame.empty
    assert frame.iloc[0]["backend"] == "xgboost"
    assert set(frame["interaction"]).intersection({"sex", "fare", "pclass", "sex|fare", "pclass|sex"})


def test_titanic_sklearn_tree_interactions_smoke(titanic_frame) -> None:
    X = titanic_frame.drop(columns=["survived"])
    y = titanic_frame["survived"]
    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=1, sort_by="gain")

    assert not frame.empty
    assert frame.iloc[0]["backend"] == "sklearn"
    assert set(frame["interaction"]).intersection({"sex", "fare", "pclass", "sex|fare", "pclass|sex"})


def test_titanic_catboost_interactions_smoke(titanic_frame) -> None:
    X = titanic_frame.drop(columns=["survived"])
    y = titanic_frame["survived"]
    model = CatBoostRegressor(depth=2, learning_rate=0.5, iterations=2, verbose=False)
    model.fit(X, y)

    frame = treefi.feature_interactions(model, max_interaction_depth=1, sort_by="interaction", ascending=True)

    assert not frame.empty
    assert frame.iloc[0]["backend"] == "catboost"
    assert set(frame["interaction"]).intersection({"sex", "fare", "pclass", "sex|fare", "pclass|sex"})


def test_titanic_random_forest_importance_smoke(titanic_frame) -> None:
    X = titanic_frame.drop(columns=["survived"])
    y = titanic_frame["survived"]
    model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0).fit(X, y)

    frame = treefi.feature_importance(model)

    assert not frame.empty
    assert set(frame["feature"]).intersection({"sex", "fare", "pclass"})
    assert set(frame["backend"]) == {"sklearn"}
