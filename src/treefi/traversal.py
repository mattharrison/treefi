"""Traversal helpers for extracting feature interactions from normalized trees."""

from __future__ import annotations

from treefi.models import InteractionKey, NormalizedTree, PathSegment


def extract_interactions(
    tree: NormalizedTree,
    *,
    max_interaction_depth: int = 2,
    max_deepening: int = -1,
    interaction_mode: str = "unordered",
) -> list[InteractionKey]:
    """Extract path-based interactions from a normalized tree."""
    interactions: list[InteractionKey] = []

    def visit(node_id: int, path: list[PathSegment]) -> None:
        node = tree.get_node(node_id)
        if node.is_leaf:
            return

        current_path = [
            *path,
            PathSegment(
                feature=node.feature or "",
                node_id=node.node_id,
                depth=len(path),
            ),
        ]

        for start_index, segment in enumerate(current_path):
            interaction_depth = len(current_path) - start_index - 1
            if max_interaction_depth >= 0 and interaction_depth > max_interaction_depth:
                continue
            if max_deepening >= 0 and segment.depth > max_deepening:
                continue

            interactions.append(
                InteractionKey(
                    features=tuple(part.feature for part in current_path[start_index:]),
                    mode=interaction_mode,
                )
            )

        if node.left_child is not None:
            visit(node.left_child, current_path)
        if node.right_child is not None and node.right_child != node.left_child:
            visit(node.right_child, current_path)

    visit(tree.root_id, [])
    return interactions
