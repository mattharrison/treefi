from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes


@pytest.fixture
def titanic_frame() -> pd.DataFrame:
    """Return a tiny local Titanic-like dataset for functional tests."""
    return pd.DataFrame(
        [
            {"pclass": 1, "sex": 0, "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.3375, "survived": 1},
            {"pclass": 3, "sex": 1, "age": 35.0, "sibsp": 0, "parch": 0, "fare": 8.05, "survived": 0},
            {"pclass": 3, "sex": 1, "age": 2.0, "sibsp": 3, "parch": 1, "fare": 21.075, "survived": 0},
            {"pclass": 1, "sex": 0, "age": 38.0, "sibsp": 1, "parch": 0, "fare": 71.2833, "survived": 1},
            {"pclass": 2, "sex": 0, "age": 26.0, "sibsp": 0, "parch": 0, "fare": 13.0, "survived": 1},
            {"pclass": 3, "sex": 1, "age": 20.0, "sibsp": 0, "parch": 0, "fare": 7.8958, "survived": 0},
            {"pclass": 1, "sex": 1, "age": 54.0, "sibsp": 0, "parch": 0, "fare": 51.8625, "survived": 0},
            {"pclass": 2, "sex": 1, "age": 14.0, "sibsp": 1, "parch": 0, "fare": 30.0708, "survived": 0},
            {"pclass": 1, "sex": 0, "age": 4.0, "sibsp": 1, "parch": 2, "fare": 81.8583, "survived": 1},
            {"pclass": 3, "sex": 0, "age": 30.0, "sibsp": 0, "parch": 0, "fare": 7.75, "survived": 1},
            {"pclass": 2, "sex": 1, "age": 40.0, "sibsp": 0, "parch": 0, "fare": 10.5, "survived": 0},
            {"pclass": 3, "sex": 0, "age": 18.0, "sibsp": 0, "parch": 0, "fare": 7.2292, "survived": 1},
        ]
    )


@pytest.fixture
def regression_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return a realistic regression dataset from sklearn."""
    bunch = load_diabetes(as_frame=True)
    X = bunch.frame[bunch.feature_names].copy()
    y = bunch.frame[bunch.target.name].copy()
    return X.iloc[:200].reset_index(drop=True), y.iloc[:200].reset_index(drop=True)


@pytest.fixture
def classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return a realistic binary classification dataset from sklearn."""
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.frame[bunch.feature_names].copy()
    y = bunch.frame[bunch.target.name].copy()
    return X.iloc[:250].reset_index(drop=True), y.iloc[:250].reset_index(drop=True)


@pytest.fixture
def tiny_regression_data() -> tuple[list[list[float]], list[float]]:
    """Return a tiny single-feature regression dataset used in unit-style tests."""
    return [[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 1.0, 1.0]


@pytest.fixture
def tiny_classification_data() -> tuple[list[list[float]], list[int]]:
    """Return a tiny single-feature binary classification dataset used in adapter tests."""
    return [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]


@pytest.fixture
def tiny_regression_data_with_tail() -> tuple[list[list[float]], list[float]]:
    """Return a tiny regression dataset with a longer tail for shallow forest tests."""
    return [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


@pytest.fixture
def tiny_regression_step_data() -> tuple[list[list[float]], list[float]]:
    """Return a simple step-function regression dataset used in boosting tests."""
    return (
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )


@pytest.fixture
def cv_regression_dataset(regression_dataset: tuple[pd.DataFrame, pd.Series]) -> tuple[pd.DataFrame, pd.Series]:
    """Return the medium regression slice used throughout CV tests."""
    X, y = regression_dataset
    return X.iloc[:120].reset_index(drop=True), y.iloc[:120].reset_index(drop=True)


@pytest.fixture
def cv_regression_dataset_small(
    regression_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the smaller regression slice used for grouped-result CV tests."""
    X, y = regression_dataset
    return X.iloc[:100].reset_index(drop=True), y.iloc[:100].reset_index(drop=True)


@pytest.fixture
def cv_classification_dataset(
    classification_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the medium classification slice used throughout CV tests."""
    X, y = classification_dataset
    return X.iloc[:180].reset_index(drop=True), y.iloc[:180].reset_index(drop=True)


@pytest.fixture
def cv_classification_dataset_small(
    classification_dataset: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the smaller classification slice used in grouped CV tests."""
    X, y = classification_dataset
    return X.iloc[:120].reset_index(drop=True), y.iloc[:120].reset_index(drop=True)


@pytest.fixture
def titanic_xy(titanic_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the Titanic-like frame into features and target."""
    return titanic_frame.drop(columns=["survived"]), titanic_frame["survived"]
