from sklearn.tree import DecisionTreeRegressor

import treefi


def test_summarize_model_exposes_leaf_stats_frame(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    result = treefi.summarize_model(model, max_interaction_depth=0)

    assert result.leaf_stats is not None
    assert result.leaf_stats.columns.tolist() == [
        "interaction",
        "leaf_effect_mean",
        "leaf_effect_var",
    ]
