"""Metric aggregation for normalized tree interactions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import cast

import pandas as pd

from treefi.models import InteractionKey, NormalizedTree, PathSegment


def summarize_interactions(
    tree: NormalizedTree,
    *,
    max_interaction_depth: int = 2,
    max_deepening: int = -1,
    interaction_mode: str = "unordered",
    sort_by: str | None = None,
    ascending: bool = False,
    top_k: int | None = None,
    min_fscore: int | None = None,
) -> pd.DataFrame:
    """Aggregate interaction metrics for one normalized tree."""
    records: list[dict[str, object]] = []

    def visit(
        node_id: int,
        path: list[PathSegment],
        gain_sum: float,
        cover_sum: float,
        path_probability: float,
    ) -> None:
        node = tree.get_node(node_id)
        if node.is_leaf:
            for record in records:
                if record["leaf_effect"] is None and record["path_signature"] == tuple(
                    seg.node_id for seg in path
                ):
                    record["leaf_effect"] = node.leaf_value
            return

        current_gain = gain_sum + (node.gain or 0.0)
        current_cover = cover_sum + (node.cover or 0.0)
        current_path = [
            *path,
            PathSegment(feature=node.feature or "", node_id=node.node_id, depth=len(path)),
        ]

        for start_index, segment in enumerate(current_path):
            interaction_depth = len(current_path) - start_index - 1
            if max_interaction_depth >= 0 and interaction_depth > max_interaction_depth:
                continue
            if max_deepening >= 0 and segment.depth > max_deepening:
                continue

            path_slice = current_path[start_index:]
            key = InteractionKey(
                features=tuple(part.feature for part in path_slice),
                mode=interaction_mode,
            )
            record_gain = (
                current_gain - gain_sum if start_index == len(current_path) - 1 else current_gain
            )
            record_cover = (
                current_cover - cover_sum if start_index == len(current_path) - 1 else current_cover
            )
            records.append(
                {
                    "interaction": key.label,
                    "interaction_order": key.feature_count,
                    "gain": record_gain,
                    "cover": record_cover,
                    "fscore": 1,
                    "weighted_fscore": path_probability,
                    "average_weighted_fscore": path_probability,
                    "average_gain": record_gain,
                    "expected_gain": record_gain * path_probability,
                    "average_tree_index": float(tree.tree_index),
                    "average_tree_depth": float(len(current_path) - 1),
                    "tree_frequency": 1.0,
                    "path_frequency": 1,
                    "first_position_mean": float(segment.depth),
                    "min_depth": len(current_path) - 1,
                    "max_depth": len(current_path) - 1,
                    "leaf_effect": None,
                    "path_signature": tuple(seg.node_id for seg in current_path),
                    "tree_index": tree.tree_index,
                }
            )

        if node.left_child is not None:
            visit(
                node.left_child,
                current_path,
                current_gain,
                current_cover,
                _child_path_probability(
                    tree=tree,
                    parent_node=node,
                    child_id=node.left_child,
                    path_probability=path_probability,
                ),
            )
        if node.right_child is not None and node.right_child != node.left_child:
            visit(
                node.right_child,
                current_path,
                current_gain,
                current_cover,
                _child_path_probability(
                    tree=tree,
                    parent_node=node,
                    child_id=node.right_child,
                    path_probability=path_probability,
                ),
            )

    visit(tree.root_id, [], 0.0, 0.0, 1.0)

    grouped: dict[str, dict[str, object]] = {}
    tree_count_by_interaction: dict[str, set[int]] = defaultdict(set)
    leaf_effects: dict[str, list[float]] = defaultdict(list)
    first_positions: dict[str, list[float]] = defaultdict(list)
    depths: dict[str, list[int]] = defaultdict(list)

    for record in records:
        name = cast(str, record["interaction"])
        tree_count_by_interaction[name].add(int(cast(int | float | str, record["tree_index"])))
        if record["leaf_effect"] is not None:
            leaf_effects[name].append(float(cast(int | float | str, record["leaf_effect"])))
        first_positions[name].append(float(cast(int | float | str, record["first_position_mean"])))
        depths[name].append(int(cast(int | float | str, record["min_depth"])))

        if name not in grouped:
            grouped[name] = {
                key: value
                for key, value in record.items()
                if key not in {"leaf_effect", "path_signature", "tree_index"}
            }
            continue

        grouped[name]["gain"] = float(cast(float, grouped[name]["gain"])) + float(
            cast(int | float | str, record["gain"])
        )
        grouped[name]["cover"] = float(cast(float, grouped[name]["cover"])) + float(
            cast(int | float | str, record["cover"])
        )
        grouped[name]["fscore"] = int(cast(int, grouped[name]["fscore"])) + 1
        grouped[name]["weighted_fscore"] = float(
            cast(float, grouped[name]["weighted_fscore"])
        ) + float(cast(int | float | str, record["weighted_fscore"]))
        grouped[name]["expected_gain"] = float(cast(float, grouped[name]["expected_gain"])) + float(
            cast(int | float | str, record["expected_gain"])
        )

    rows: list[dict[str, object]] = []
    for name, row in grouped.items():
        row["average_weighted_fscore"] = float(cast(float, row["weighted_fscore"])) / float(
            cast(int, row["fscore"])
        )
        row["average_gain"] = float(cast(float, row["gain"])) / float(cast(int, row["fscore"]))
        row["tree_frequency"] = float(len(tree_count_by_interaction[name]))
        row["path_frequency"] = row["fscore"]
        row["first_position_mean"] = sum(first_positions[name]) / len(first_positions[name])
        row["min_depth"] = min(depths[name])
        row["max_depth"] = max(depths[name])
        if leaf_effects[name]:
            mean = sum(leaf_effects[name]) / len(leaf_effects[name])
            row["leaf_effect_mean"] = mean
            row["leaf_effect_var"] = sum((value - mean) ** 2 for value in leaf_effects[name]) / len(
                leaf_effects[name]
            )
        else:
            row["leaf_effect_mean"] = math.nan
            row["leaf_effect_var"] = math.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["rank_gain"] = _rank_desc(frame["gain"].tolist())
    frame["rank_fscore"] = _rank_desc(frame["fscore"].tolist())
    frame["rank_expected_gain"] = _rank_desc(frame["expected_gain"].tolist())
    frame["rank_consensus"] = (
        frame["rank_gain"] + frame["rank_fscore"] + frame["rank_expected_gain"]
    ) / 3.0

    if min_fscore is not None:
        frame = frame.loc[frame["fscore"] >= min_fscore].reset_index(drop=True)

    if sort_by is not None and not frame.empty:
        frame = frame.sort_values(
            by=[sort_by, "interaction"],
            ascending=[ascending, True],
            kind="mergesort",
        ).reset_index(drop=True)

    if top_k is not None:
        frame = frame.head(top_k).reset_index(drop=True)

    return frame


def _rank_desc(values: list[float]) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    ranks = [0] * len(values)
    for rank, (index, _) in enumerate(indexed, start=1):
        ranks[index] = rank
    return ranks


def _child_path_probability(
    tree: NormalizedTree, parent_node, child_id: int, path_probability: float
) -> float:
    parent_cover = parent_node.cover
    child_cover = tree.get_node(child_id).cover
    if parent_cover in {None, 0} or child_cover is None:
        return path_probability
    return path_probability * (float(child_cover) / float(parent_cover))
