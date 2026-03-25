import pytest
from sklearn.tree import DecisionTreeRegressor

import treefi


def test_public_api_raises_unsupported_model_error_for_unknown_objects() -> None:
    with pytest.raises(treefi.UnsupportedModelError):
        treefi.feature_interactions(object())


def test_public_api_raises_unfitted_model_error_for_sklearn_estimators() -> None:
    model = DecisionTreeRegressor()

    with pytest.raises(treefi.UnfittedModelError):
        treefi.feature_interactions(model)


def test_public_api_rejects_invalid_interaction_mode() -> None:
    model = DecisionTreeRegressor(max_depth=1).fit([[0.0], [1.0]], [0.0, 1.0])

    with pytest.raises(ValueError, match="interaction_mode"):
        treefi.feature_interactions(model, interaction_mode="diagonal")


def test_public_api_rejects_negative_top_k() -> None:
    model = DecisionTreeRegressor(max_depth=1).fit([[0.0], [1.0]], [0.0, 1.0])

    with pytest.raises(ValueError, match="top_k"):
        treefi.feature_interactions(model, top_k=-1)
