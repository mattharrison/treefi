"""User-facing exception types for treefi."""


class TreeFIError(Exception):
    """Base exception for treefi."""


class UnsupportedModelError(TreeFIError):
    """Raised when treefi does not support the provided model type."""


class UnfittedModelError(TreeFIError):
    """Raised when treefi receives a model that has not been fitted."""
