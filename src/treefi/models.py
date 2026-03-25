"""Normalized internal model objects for treefi."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class NormalizedNode:
    """Backend-agnostic representation of a tree node."""

    node_id: int
    is_leaf: bool
    feature: str | None = None
    split_condition: str | None = None
    gain: float | None = None
    cover: float | None = None
    left_child: int | None = None
    right_child: int | None = None
    leaf_value: float | None = None

    def __post_init__(self) -> None:
        if self.is_leaf and self.leaf_value is None:
            raise ValueError("leaf_value is required for leaf nodes")


@dataclass(slots=True)
class NormalizedTree:
    """Backend-agnostic representation of a single decision tree."""

    tree_index: int
    root_id: int
    nodes: list[NormalizedNode]
    _nodes_by_id: dict[int, NormalizedNode] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nodes_by_id = {node.node_id: node for node in self.nodes}
        if self.root_id not in self._nodes_by_id:
            raise ValueError("root_id must reference a node in the tree")

    @property
    def root(self) -> NormalizedNode:
        return self._nodes_by_id[self.root_id]

    def get_node(self, node_id: int) -> NormalizedNode:
        return self._nodes_by_id[node_id]


@dataclass(slots=True)
class NormalizedEnsemble:
    """Backend-agnostic representation of one fitted tree-based model."""

    trees: list[NormalizedTree]
    backend: str
    model_type: str
    _trees_by_index: dict[int, NormalizedTree] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._trees_by_index = {tree.tree_index: tree for tree in self.trees}
        if len(self._trees_by_index) != len(self.trees):
            raise ValueError("tree_index values must be unique within an ensemble")

    def get_tree(self, tree_index: int) -> NormalizedTree:
        return self._trees_by_index[tree_index]


@dataclass(frozen=True, slots=True)
class PathSegment:
    """One feature-bearing segment within a root-to-node path."""

    feature: str
    node_id: int
    depth: int


@dataclass(frozen=True, slots=True)
class InteractionKey:
    """Canonical representation of an interaction in ordered or unordered form."""

    features: tuple[str, ...]
    mode: str = "unordered"

    def __post_init__(self) -> None:
        if self.mode not in {"ordered", "unordered"}:
            raise ValueError("mode must be 'ordered' or 'unordered'")

    @property
    def label(self) -> str:
        if self.mode == "ordered":
            return " -> ".join(self.features)
        return "|".join(sorted(set(self.features)))

    @property
    def feature_count(self) -> int:
        if self.mode == "ordered":
            return len(self.features)
        return len(set(self.features))
