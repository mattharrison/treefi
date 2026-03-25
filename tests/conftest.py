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
