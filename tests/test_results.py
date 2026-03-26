import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import treefi
from treefi.results import AnalysisResult
from treefi.schema import empty_importance_frame, empty_interactions_frame


def test_summarize_model_returns_grouped_empty_frames_for_none_input() -> None:
    result = treefi.summarize_model(None)

    assert isinstance(result, AnalysisResult)
    assert isinstance(result.interactions, pd.DataFrame)
    assert isinstance(result.importance, pd.DataFrame)
    assert result.interactions.columns.tolist() == empty_interactions_frame().columns.tolist()
    assert result.importance.columns.tolist() == empty_importance_frame().columns.tolist()


def test_summarize_model_returns_grouped_outputs_for_supported_model(
    tiny_regression_data: tuple[list[list[float]], list[float]],
) -> None:
    X, y = tiny_regression_data
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)

    result = treefi.summarize_model(model, max_interaction_depth=0)

    assert isinstance(result, AnalysisResult)
    assert not result.interactions.empty
    assert not result.importance.empty
    assert result.leaf_stats is not None
    assert "interaction" in result.leaf_stats.columns
    assert result.metadata["backend"] == "sklearn"
