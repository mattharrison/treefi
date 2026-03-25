from treefi.models import NormalizedNode, NormalizedTree
from treefi.traversal import extract_interactions


def make_linear_tree(*features: str) -> NormalizedTree:
    nodes: list[NormalizedNode] = []
    for index, feature in enumerate(features):
        nodes.append(
            NormalizedNode(
                node_id=index,
                feature=feature,
                split_condition="< 0.5",
                left_child=index + 1,
                right_child=index + 1,
                is_leaf=False,
            )
        )
    nodes.append(NormalizedNode(node_id=len(features), is_leaf=True, leaf_value=1.0))
    return NormalizedTree(tree_index=0, root_id=0, nodes=nodes)


def test_extract_interactions_traverses_single_tree_with_depth_limit() -> None:
    tree = make_linear_tree("f0", "f1", "f2")

    interactions = extract_interactions(tree, max_interaction_depth=1)

    assert [interaction.label for interaction in interactions] == [
        "f0",
        "f0|f1",
        "f1",
        "f1|f2",
        "f2",
    ]


def test_extract_interactions_respects_max_deepening_for_interaction_starts() -> None:
    tree = make_linear_tree("f0", "f1", "f2")

    interactions = extract_interactions(tree, max_interaction_depth=2, max_deepening=0)

    assert [interaction.label for interaction in interactions] == [
        "f0",
        "f0|f1",
        "f0|f1|f2",
    ]


def test_extract_interactions_preserves_repeated_features_in_ordered_mode() -> None:
    tree = make_linear_tree("f0", "f1", "f0")

    interactions = extract_interactions(
        tree,
        max_interaction_depth=2,
        interaction_mode="ordered",
    )

    assert [interaction.label for interaction in interactions] == [
        "f0",
        "f0 -> f1",
        "f1",
        "f0 -> f1 -> f0",
        "f1 -> f0",
        "f0",
    ]


def test_extract_interactions_supports_unordered_output_mode() -> None:
    tree = make_linear_tree("f0", "f1", "f0")

    interactions = extract_interactions(
        tree,
        max_interaction_depth=2,
        interaction_mode="unordered",
    )

    assert [interaction.label for interaction in interactions] == [
        "f0",
        "f0|f1",
        "f1",
        "f0|f1",
        "f0|f1",
        "f0",
    ]
