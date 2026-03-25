"""Dataframe schema definitions for treefi outputs."""

from __future__ import annotations

import pandas as pd

INTERACTION_COLUMNS = [
    "interaction",
    "features",
    "depth",
    "gain",
    "cover",
    "fscore",
    "weighted_fscore",
    "average_weighted_fscore",
    "average_gain",
    "expected_gain",
    "average_tree_index",
    "average_tree_depth",
    "backend",
    "model_type",
    "occurrence_count",
    "tree_count",
    "feature_count",
    "path_probability_sum",
]

IMPORTANCE_COLUMNS = [
    "feature",
    "gain",
    "cover",
    "fscore",
    "weighted_fscore",
    "average_weighted_fscore",
    "average_gain",
    "expected_gain",
    "average_tree_index",
    "average_tree_depth",
    "backend",
    "model_type",
    "occurrence_count",
    "tree_count",
]


def empty_interactions_frame() -> pd.DataFrame:
    """Return an empty interaction dataframe using the public schema."""
    return pd.DataFrame(columns=INTERACTION_COLUMNS)


def empty_importance_frame() -> pd.DataFrame:
    """Return an empty feature-importance dataframe using the public schema."""
    return pd.DataFrame(columns=IMPORTANCE_COLUMNS)
