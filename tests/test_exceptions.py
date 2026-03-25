import pytest

import treefi


def test_public_api_raises_unsupported_model_error_for_unknown_models() -> None:
    with pytest.raises(treefi.UnsupportedModelError):
        treefi.feature_importance(object())

    with pytest.raises(treefi.UnsupportedModelError):
        treefi.feature_interactions(object())

    with pytest.raises(treefi.UnsupportedModelError):
        treefi.summarize_model(object())
