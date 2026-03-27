import lightgbm as lgb
import pandas as pd
import pytest
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from treefi.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
    MetricCapabilities,
    MetricCapability,
    ModelAdapter,
    SklearnAdapter,
    XGBoostAdapter,
    get_adapter_for_model,
    resolve_feature_names,
)
from treefi.exceptions import UnsupportedModelError


class DummySupportedModel:
    pass


class DummyUnsupportedModel:
    pass


class DummyFeatureModel:
    feature_names_in_ = ("age", "income")


class DummyAdapter(ModelAdapter):
    model_types = (DummySupportedModel,)

    def is_fitted(self, model) -> bool:
        return True

    def get_feature_names(self, model, feature_names=None) -> list[str]:
        return ["f0"]

    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None):
        return None

    def metric_capabilities(self):
        return MetricCapabilities(
            {
                "gain": MetricCapability(status="exact"),
                "cover": MetricCapability(
                    status="approximate", detail="normalized from sample counts"
                ),
            }
        )


def test_get_adapter_for_model_selects_matching_adapter() -> None:
    adapter = get_adapter_for_model(DummySupportedModel(), adapters=[DummyAdapter()])

    assert isinstance(adapter, DummyAdapter)


def test_get_adapter_for_model_rejects_unknown_models() -> None:
    with pytest.raises(UnsupportedModelError):
        get_adapter_for_model(DummyUnsupportedModel(), adapters=[DummyAdapter()])


def test_metric_capabilities_store_status_and_detail() -> None:
    capabilities = DummyAdapter().metric_capabilities()

    assert capabilities["gain"].status == "exact"
    assert capabilities["gain"].detail is None
    assert capabilities["cover"].status == "approximate"
    assert capabilities["cover"].detail == "normalized from sample counts"


def test_resolve_feature_names_prefers_explicit_names() -> None:
    names = resolve_feature_names(DummyFeatureModel(), feature_names=["x0", "x1"], n_features=2)

    assert names == ["x0", "x1"]


def test_resolve_feature_names_uses_model_feature_names_when_available() -> None:
    names = resolve_feature_names(DummyFeatureModel(), feature_names=None, n_features=2)

    assert names == ["age", "income"]


def test_resolve_feature_names_generates_fallback_names() -> None:
    names = resolve_feature_names(object(), feature_names=None, n_features=3)

    assert names == ["f0", "f1", "f2"]


def test_sklearn_adapter_normalizes_decision_tree_regressor(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "DecisionTreeRegressor"
    assert len(ensemble.trees) == 1

    tree = ensemble.trees[0]
    assert tree.root.feature == "f0"
    assert tree.root.left_child is not None
    assert tree.root.right_child is not None
    assert tree.get_node(tree.root.left_child).is_leaf is True
    assert tree.get_node(tree.root.right_child).is_leaf is True


def test_sklearn_adapter_normalizes_decision_tree_classifier(
    tiny_classification_data: tuple[list[list[float]], list[int]],
) -> None:
    X, y = tiny_classification_data
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "DecisionTreeClassifier"
    assert len(ensemble.trees) == 1

    tree = ensemble.trees[0]
    assert tree.root.feature == "f0"
    assert tree.get_node(tree.root.left_child).leaf_value is not None
    assert tree.get_node(tree.root.right_child).leaf_value is not None


def test_sklearn_adapter_normalizes_random_forest_regressor(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = RandomForestRegressor(n_estimators=2, max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "RandomForestRegressor"
    assert len(ensemble.trees) == 2
    assert ensemble.trees[0].tree_index == 0
    assert ensemble.trees[1].tree_index == 1


def test_sklearn_adapter_normalizes_random_forest_classifier(
    tiny_classification_data: tuple[list[list[float]], list[int]],
) -> None:
    X, y = tiny_classification_data
    model = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "RandomForestClassifier"
    assert len(ensemble.trees) == 2


def test_sklearn_metric_capabilities_document_approximate_gain_and_cover() -> None:
    capabilities = SklearnAdapter().metric_capabilities()

    assert capabilities["gain"].status == "approximate"
    assert "derivation" in capabilities["gain"].detail
    assert capabilities["cover"].status == "approximate"
    assert "sample counts" in capabilities["cover"].detail
    assert capabilities["weight"].status == "exact"
    assert capabilities["total_gain"].status == "approximate"
    assert capabilities["total_cover"].status == "approximate"
    assert capabilities["average_cover"].status == "approximate"


def test_backend_metric_capability_statuses_remain_explicit() -> None:
    xgboost_capabilities = XGBoostAdapter().metric_capabilities()
    lightgbm_capabilities = LightGBMAdapter().metric_capabilities()
    catboost_capabilities = CatBoostAdapter().metric_capabilities()

    assert xgboost_capabilities["gain"].status == "exact"
    assert xgboost_capabilities["cover"].status == "exact"
    assert xgboost_capabilities["total_gain"].status == "exact"
    assert xgboost_capabilities["total_cover"].status == "exact"
    assert xgboost_capabilities["weight"].status == "exact"
    assert xgboost_capabilities["average_cover"].status == "exact"
    assert lightgbm_capabilities["gain"].status == "exact"
    assert lightgbm_capabilities["cover"].status == "approximate"
    assert lightgbm_capabilities["total_gain"].status == "exact"
    assert lightgbm_capabilities["total_cover"].status == "approximate"
    assert lightgbm_capabilities["weight"].status == "exact"
    assert lightgbm_capabilities["average_cover"].status == "approximate"
    assert catboost_capabilities["gain"].status == "approximate"
    assert catboost_capabilities["cover"].status == "approximate"
    assert catboost_capabilities["total_gain"].status == "synthetic"
    assert catboost_capabilities["total_cover"].status == "approximate"
    assert catboost_capabilities["weight"].status == "exact"
    assert catboost_capabilities["average_cover"].status == "approximate"


def test_sklearn_adapter_derives_gain_and_cover_for_decision_tree_regressor(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    root = ensemble.trees[0].root
    assert root.gain == pytest.approx(1.0)
    assert root.cover == pytest.approx(4.0)


def test_sklearn_adapter_derives_gain_and_cover_for_random_forest_trees(
    tiny_regression_data_with_tail: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data_with_tail
    model = RandomForestRegressor(n_estimators=2, max_depth=1, random_state=0).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert len(ensemble.trees) == 2
    for tree in ensemble.trees:
        assert tree.root.gain is not None
        assert tree.root.gain > 0.0
        assert tree.root.cover is not None
        assert tree.root.cover > 0.0


def test_xgboost_adapter_normalizes_booster() -> None:
    dtrain = xgb.DMatrix(
        [[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["f0"]
    )
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 1, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=1,
    )

    ensemble = XGBoostAdapter().to_normalized_ensemble(booster)

    assert ensemble.backend == "xgboost"
    assert ensemble.model_type == "Booster"
    assert len(ensemble.trees) == 1
    assert ensemble.trees[0].root.feature == "f0"


def test_xgboost_adapter_normalizes_sklearn_wrapper_via_get_booster() -> None:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=1,
        learning_rate=1.0,
        n_estimators=1,
    )
    model.fit([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0])

    ensemble = XGBoostAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "xgboost"
    assert ensemble.model_type == "XGBRegressor"
    assert len(ensemble.trees) == 1


def test_xgboost_feature_name_resolution_does_not_mutate_source_booster() -> None:
    dtrain = xgb.DMatrix(
        [[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["orig"]
    )
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 1, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=1,
    )

    before = list(booster.feature_names)
    names = XGBoostAdapter().get_feature_names(booster, feature_names=["override"])
    after = list(booster.feature_names)

    assert names == ["override"]
    assert before == ["orig"]
    assert after == ["orig"]


def test_catboost_adapter_can_extract_tiny_tree_structure(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = CatBoostRegressor(depth=1, learning_rate=1.0, iterations=1, verbose=False)
    model.fit(X, y)

    ensemble = CatBoostAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "catboost"
    assert ensemble.model_type == "CatBoostRegressor"
    assert len(ensemble.trees) == 1
    assert ensemble.trees[0].root.feature == "f0"


def test_catboost_adapter_derives_cover_from_leaf_weights(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = CatBoostRegressor(depth=1, learning_rate=1.0, iterations=1, verbose=False)
    model.fit(X, y)

    ensemble = CatBoostAdapter().to_normalized_ensemble(model)

    root = ensemble.trees[0].root
    left = ensemble.trees[0].get_node(root.left_child)
    right = ensemble.trees[0].get_node(root.right_child)
    assert left.cover is not None
    assert right.cover is not None
    assert left.cover > 0.0
    assert right.cover > 0.0
    assert left.cover + right.cover == pytest.approx(4.0)
    assert root.cover == pytest.approx(4.0)


def test_catboost_adapter_derives_approximate_gain_from_leaf_values_and_weights(
    tiny_regression_step_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_step_data
    model = CatBoostRegressor(depth=2, learning_rate=1.0, iterations=1, verbose=False)
    model.fit(X, y)

    ensemble = CatBoostAdapter().to_normalized_ensemble(model)

    root = ensemble.trees[0].root
    assert root.gain is not None
    assert root.gain > 0.0


def test_catboost_metric_capabilities_mark_gain_and_cover_approximate() -> None:
    capabilities = CatBoostAdapter().metric_capabilities()

    assert capabilities["gain"].status == "approximate"
    assert "variance reduction" in capabilities["gain"].detail
    assert capabilities["cover"].status == "approximate"
    assert "leaf weights" in capabilities["cover"].detail
    assert capabilities["total_gain"].status == "synthetic"
    assert capabilities["weight"].status == "exact"


def test_catboost_adapter_explicitly_marks_categorical_split_normalization_unsupported() -> None:
    assert CatBoostAdapter().supports_categorical_splits() is False


def test_sklearn_adapter_normalizes_hist_gradient_boosting_regressor(
    tiny_regression_step_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_step_data
    model = HistGradientBoostingRegressor(
        max_depth=2,
        max_iter=5,
        min_samples_leaf=1,
        learning_rate=1.0,
        random_state=0,
    ).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "HistGradientBoostingRegressor"
    assert len(ensemble.trees) == 5
    assert ensemble.trees[0].root.feature == "f0"


def test_sklearn_adapter_preserves_hist_gradient_boosting_gain_and_cover(
    tiny_regression_step_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_step_data
    model = HistGradientBoostingRegressor(
        max_depth=2,
        max_iter=3,
        min_samples_leaf=1,
        learning_rate=1.0,
        random_state=0,
    ).fit(X, y)

    ensemble = SklearnAdapter().to_normalized_ensemble(model)

    root = ensemble.trees[0].root
    assert root.gain is not None
    assert root.gain > 0.0
    assert root.cover is not None
    assert root.cover > 0.0


def test_lightgbm_adapter_normalizes_structured_tree_dump(
    tiny_regression_step_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_step_data
    train = lgb.Dataset(pd.DataFrame({"f0": [row[0] for row in X]}), label=y)
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

    ensemble = LightGBMAdapter().to_normalized_ensemble(booster)

    assert ensemble.backend == "lightgbm"
    assert ensemble.model_type == "Booster"
    assert len(ensemble.trees) == 2
    assert ensemble.trees[0].root.feature == "f0"


def test_feature_name_parity_across_histgb_lightgbm_and_xgboost(
    tiny_regression_step_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_step_data
    flat_X = [row[0] for row in X]

    hist_model = HistGradientBoostingRegressor(
        max_depth=2,
        max_iter=2,
        min_samples_leaf=1,
        learning_rate=1.0,
        random_state=0,
    ).fit(X, y)
    hist_name = SklearnAdapter().to_normalized_ensemble(hist_model).trees[0].root.feature

    train = lgb.Dataset(pd.DataFrame({"fare": flat_X}), label=y)
    lgb_model = lgb.train(
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
        num_boost_round=1,
    )
    lgb_name = LightGBMAdapter().to_normalized_ensemble(lgb_model).trees[0].root.feature

    dtrain = xgb.DMatrix(pd.DataFrame({"fare": flat_X}), label=y, feature_names=["fare"])
    xgb_model = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 2, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=1,
    )
    xgb_name = XGBoostAdapter().to_normalized_ensemble(xgb_model).trees[0].root.feature

    assert hist_name == "f0"
    assert lgb_name == "fare"
    assert xgb_name == "fare"
