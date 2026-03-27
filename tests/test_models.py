import pytest

from treefi.models import (
    InteractionKey,
    NormalizedEnsemble,
    NormalizedNode,
    NormalizedTree,
    PathSegment,
)


def test_normalized_node_stores_split_node_fields() -> None:
    node = NormalizedNode(
        node_id=0,
        feature="f0",
        split_condition="< 1.5",
        gain=2.5,
        cover=10.0,
        left_child=1,
        right_child=2,
        is_leaf=False,
    )

    assert node.node_id == 0
    assert node.feature == "f0"
    assert node.split_condition == "< 1.5"
    assert node.gain == 2.5
    assert node.cover == 10.0
    assert node.left_child == 1
    assert node.right_child == 2
    assert node.is_leaf is False
    assert node.leaf_value is None


def test_normalized_node_stores_leaf_fields() -> None:
    node = NormalizedNode(
        node_id=3,
        is_leaf=True,
        leaf_value=0.75,
        cover=4.0,
    )

    assert node.node_id == 3
    assert node.is_leaf is True
    assert node.leaf_value == 0.75
    assert node.cover == 4.0
    assert node.feature is None
    assert node.left_child is None
    assert node.right_child is None


def test_normalized_node_requires_leaf_value_for_leaf_nodes() -> None:
    with pytest.raises(ValueError, match="leaf_value"):
        NormalizedNode(node_id=4, is_leaf=True)


def test_normalized_tree_indexes_nodes_and_root() -> None:
    root = NormalizedNode(
        node_id=0,
        feature="f0",
        split_condition="< 1.5",
        left_child=1,
        right_child=2,
        is_leaf=False,
    )
    left = NormalizedNode(node_id=1, is_leaf=True, leaf_value=0.1)
    right = NormalizedNode(node_id=2, is_leaf=True, leaf_value=0.9)

    tree = NormalizedTree(tree_index=3, root_id=0, nodes=[root, left, right])

    assert tree.tree_index == 3
    assert tree.root.node_id == 0
    assert tree.get_node(1) is left
    assert tree.get_node(2) is right


def test_normalized_tree_requires_declared_root_node() -> None:
    leaf = NormalizedNode(node_id=1, is_leaf=True, leaf_value=0.5)

    with pytest.raises(ValueError, match="root_id"):
        NormalizedTree(tree_index=0, root_id=0, nodes=[leaf])


def test_normalized_ensemble_preserves_tree_order_and_lookup() -> None:
    tree0 = NormalizedTree(
        tree_index=0,
        root_id=0,
        nodes=[NormalizedNode(node_id=0, is_leaf=True, leaf_value=0.1)],
    )
    tree1 = NormalizedTree(
        tree_index=1,
        root_id=0,
        nodes=[NormalizedNode(node_id=0, is_leaf=True, leaf_value=0.9)],
    )

    ensemble = NormalizedEnsemble(
        trees=[tree0, tree1], backend="sklearn", model_type="DecisionTreeRegressor"
    )

    assert ensemble.backend == "sklearn"
    assert ensemble.model_type == "DecisionTreeRegressor"
    assert ensemble.trees == [tree0, tree1]
    assert ensemble.get_tree(1) is tree1


def test_normalized_ensemble_rejects_duplicate_tree_indexes() -> None:
    tree0 = NormalizedTree(
        tree_index=0,
        root_id=0,
        nodes=[NormalizedNode(node_id=0, is_leaf=True, leaf_value=0.1)],
    )
    duplicate = NormalizedTree(
        tree_index=0,
        root_id=0,
        nodes=[NormalizedNode(node_id=0, is_leaf=True, leaf_value=0.9)],
    )

    with pytest.raises(ValueError, match="tree_index"):
        NormalizedEnsemble(
            trees=[tree0, duplicate], backend="sklearn", model_type="RandomForestRegressor"
        )


def test_path_segment_captures_feature_position_and_node() -> None:
    segment = PathSegment(feature="f1", node_id=7, depth=2)

    assert segment.feature == "f1"
    assert segment.node_id == 7
    assert segment.depth == 2


def test_interaction_key_supports_ordered_and_unordered_views() -> None:
    ordered = InteractionKey(features=("f0", "f1", "f0"), mode="ordered")
    unordered = InteractionKey(features=("f0", "f1", "f0"), mode="unordered")

    assert ordered.label == "f0 -> f1 -> f0"
    assert ordered.feature_count == 3

    assert unordered.label == "f0|f1"
    assert unordered.feature_count == 2


def test_interaction_key_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        InteractionKey(features=("f0",), mode="sideways")
