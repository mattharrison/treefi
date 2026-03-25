"""Grouped result objects for treefi analyses."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class AnalysisResult:
    """Container for grouped dataframe outputs from model analysis."""

    interactions: pd.DataFrame
    importance: pd.DataFrame
    leaf_stats: pd.DataFrame | None = None
    split_histograms: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class CrossValidatedResult:
    """Container for fold-level and aggregated dataframe outputs from CV analysis."""

    folds: pd.DataFrame
    summary: pd.DataFrame
    interaction_folds: pd.DataFrame | None = None
    interaction_summary: pd.DataFrame | None = None
    importance_folds: pd.DataFrame | None = None
    importance_summary: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interaction_folds is None:
            self.interaction_folds = self.folds
        if self.interaction_summary is None:
            self.interaction_summary = self.summary
