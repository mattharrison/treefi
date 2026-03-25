"""Public package surface for treefi."""

from importlib.metadata import PackageNotFoundError, version

from treefi.api import (
    cross_validated_importance,
    cross_validated_interactions,
    feature_importance,
    feature_interactions,
    summarize_model,
)
from treefi.exceptions import TreeFIError, UnfittedModelError, UnsupportedModelError

__all__ = [
    "__version__",
    "cross_validated_importance",
    "cross_validated_interactions",
    "feature_importance",
    "feature_interactions",
    "summarize_model",
    "TreeFIError",
    "UnsupportedModelError",
    "UnfittedModelError",
]

try:
    __version__ = version("treefi")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
