"""Adapter selection and base protocol for treefi backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from treefi.exceptions import UnsupportedModelError
from treefi.models import NormalizedEnsemble, NormalizedNode, NormalizedTree


@dataclass(frozen=True, slots=True)
class MetricCapability:
    """Capability description for one metric on one backend."""

    status: str
    detail: str | None = None


MetricCapabilities = dict[str, MetricCapability]


class ModelAdapter(ABC):
    """Base class for backend-specific model adapters."""

    model_types: tuple[type, ...] = ()

    def can_handle(self, model) -> bool:
        return isinstance(model, self.model_types)

    @abstractmethod
    def is_fitted(self, model) -> bool: ...

    @abstractmethod
    def get_feature_names(self, model, feature_names=None) -> list[str]: ...

    @abstractmethod
    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None): ...

    @abstractmethod
    def metric_capabilities(self) -> MetricCapabilities: ...


class SklearnAdapter(ModelAdapter):
    """Adapter for sklearn tree-based models."""

    def can_handle(self, model) -> bool:
        return type(model).__module__.startswith("sklearn.")

    def is_fitted(self, model) -> bool:
        return (
            hasattr(model, "tree_")
            or hasattr(model, "estimators_")
            or hasattr(model, "_predictors")
        )

    def get_feature_names(self, model, feature_names=None) -> list[str]:
        n_features = getattr(model, "n_features_in_", None)
        return resolve_feature_names(model, feature_names=feature_names, n_features=n_features)

    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None):
        feature_names_resolved = self.get_feature_names(model, feature_names=feature_names)
        estimators = [model]
        if hasattr(model, "estimators_"):
            estimators = list(model.estimators_)
            if max_trees is not None:
                estimators = estimators[:max_trees]
            trees = [
                self._normalize_tree(estimator.tree_, feature_names_resolved, tree_index=index)
                for index, estimator in enumerate(estimators)
            ]
        elif hasattr(model, "_predictors"):
            predictors = list(model._predictors)
            if max_trees is not None:
                predictors = predictors[:max_trees]
            trees = [
                self._normalize_hist_tree(
                    predictor_row[0], feature_names_resolved, tree_index=index
                )
                for index, predictor_row in enumerate(predictors)
            ]
        else:
            trees = [self._normalize_tree(model.tree_, feature_names_resolved, tree_index=0)]

        return NormalizedEnsemble(
            trees=trees,
            backend="sklearn",
            model_type=type(model).__name__,
        )

    def metric_capabilities(self) -> MetricCapabilities:
        return {
            "gain": MetricCapability(
                status="approximate", detail="requires sklearn-specific derivation"
            ),
            "cover": MetricCapability(status="approximate", detail="derived from sample counts"),
            "weight": MetricCapability(status="exact", detail="split count"),
            "total_gain": MetricCapability(
                status="approximate", detail="sum of derived sklearn split gains"
            ),
            "total_cover": MetricCapability(
                status="approximate", detail="sum of derived sklearn node sample counts"
            ),
            "average_cover": MetricCapability(
                status="approximate", detail="derived from sample counts per split occurrence"
            ),
        }

    def _normalize_tree(
        self, sklearn_tree, feature_names: list[str], tree_index: int
    ) -> NormalizedTree:
        nodes: list[NormalizedNode] = []

        for node_id in range(sklearn_tree.node_count):
            left_child = int(sklearn_tree.children_left[node_id])
            right_child = int(sklearn_tree.children_right[node_id])
            is_leaf = left_child == -1 and right_child == -1
            cover = self._get_tree_cover(sklearn_tree, node_id)

            if is_leaf:
                value = sklearn_tree.value[node_id].ravel()[0]
                nodes.append(
                    NormalizedNode(
                        node_id=node_id,
                        is_leaf=True,
                        leaf_value=float(value),
                        cover=cover,
                    )
                )
                continue

            feature_index = int(sklearn_tree.feature[node_id])
            nodes.append(
                NormalizedNode(
                    node_id=node_id,
                    is_leaf=False,
                    feature=feature_names[feature_index],
                    split_condition=f"< {float(sklearn_tree.threshold[node_id])}",
                    gain=self._derive_sklearn_gain(sklearn_tree, node_id, left_child, right_child),
                    cover=cover,
                    left_child=left_child,
                    right_child=right_child,
                )
            )

        return NormalizedTree(tree_index=tree_index, root_id=0, nodes=nodes)

    def _derive_sklearn_gain(
        self, sklearn_tree, node_id: int, left_child: int, right_child: int
    ) -> float:
        node_weight = self._get_tree_cover(sklearn_tree, node_id)
        left_weight = self._get_tree_cover(sklearn_tree, left_child)
        right_weight = self._get_tree_cover(sklearn_tree, right_child)
        impurity = float(sklearn_tree.impurity[node_id])
        left_impurity = float(sklearn_tree.impurity[left_child])
        right_impurity = float(sklearn_tree.impurity[right_child])
        return (
            (impurity * node_weight)
            - (left_impurity * left_weight)
            - (right_impurity * right_weight)
        )

    def _get_tree_cover(self, sklearn_tree, node_id: int) -> float:
        weighted_counts = getattr(sklearn_tree, "weighted_n_node_samples", None)
        if weighted_counts is not None:
            return float(weighted_counts[node_id])
        return float(sklearn_tree.n_node_samples[node_id])

    def _normalize_hist_tree(
        self, predictor, feature_names: list[str], tree_index: int
    ) -> NormalizedTree:
        nodes: list[NormalizedNode] = []

        for node_id, row in enumerate(predictor.nodes):
            if bool(row["is_leaf"]):
                nodes.append(
                    NormalizedNode(
                        node_id=node_id,
                        is_leaf=True,
                        leaf_value=float(row["value"]),
                        cover=float(row["count"]),
                    )
                )
                continue

            feature_index = int(row["feature_idx"])
            nodes.append(
                NormalizedNode(
                    node_id=node_id,
                    is_leaf=False,
                    feature=feature_names[feature_index],
                    split_condition=f"< {float(row['num_threshold'])}",
                    gain=float(row["gain"]),
                    cover=float(row["count"]),
                    left_child=int(row["left"]),
                    right_child=int(row["right"]),
                )
            )

        return NormalizedTree(tree_index=tree_index, root_id=0, nodes=nodes)


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost boosters and sklearn wrappers."""

    def can_handle(self, model) -> bool:
        return type(model).__module__.startswith("xgboost")

    def is_fitted(self, model) -> bool:
        return self.can_handle(model)

    def get_feature_names(self, model, feature_names=None) -> list[str]:
        if feature_names is not None:
            return list(feature_names)

        booster = self._get_booster(model)
        feature_names_attr = getattr(booster, "feature_names", None)
        if feature_names_attr is not None:
            return list(feature_names_attr)
        return []

    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None):
        booster = self._get_booster(model)
        tree_frame = booster.trees_to_dataframe()
        if max_trees is not None:
            tree_frame = tree_frame.loc[tree_frame["Tree"] < max_trees]

        trees: list[NormalizedTree] = []
        for tree_index, group in tree_frame.groupby("Tree", sort=True):
            nodes: list[NormalizedNode] = []
            for row in group.itertuples(index=False):
                if row.Feature == "Leaf":
                    nodes.append(
                        NormalizedNode(
                            node_id=int(row.Node),
                            is_leaf=True,
                            leaf_value=float(row.Gain),
                            cover=float(row.Cover),
                        )
                    )
                    continue

                nodes.append(
                    NormalizedNode(
                        node_id=int(row.Node),
                        is_leaf=False,
                        feature=str(row.Feature),
                        split_condition=f"< {float(row.Split)}",
                        gain=float(row.Gain),
                        cover=float(row.Cover),
                        left_child=_parse_xgb_child_id(row.Yes),
                        right_child=_parse_xgb_child_id(row.No),
                    )
                )

            trees.append(NormalizedTree(tree_index=int(tree_index), root_id=0, nodes=nodes))

        model_type = type(model).__name__
        return NormalizedEnsemble(trees=trees, backend="xgboost", model_type=model_type)

    def metric_capabilities(self) -> MetricCapabilities:
        return {
            "gain": MetricCapability(status="exact"),
            "cover": MetricCapability(status="exact"),
            "weight": MetricCapability(status="exact", detail="XGBoost split count"),
            "total_gain": MetricCapability(status="exact"),
            "total_cover": MetricCapability(status="exact"),
            "average_cover": MetricCapability(status="exact"),
        }

    def _get_booster(self, model):
        if hasattr(model, "trees_to_dataframe"):
            return model
        if hasattr(model, "get_booster"):
            return model.get_booster()
        raise UnsupportedModelError(f"treefi does not support model type: {type(model).__name__}")


def _parse_xgb_child_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float):
        return None
    return int(str(value).split("-")[-1])


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models using the exposed tree internals."""

    def can_handle(self, model) -> bool:
        return hasattr(model, "get_tree_leaf_counts") and hasattr(model, "_get_tree_splits")

    def is_fitted(self, model) -> bool:
        return self.can_handle(model)

    def get_feature_names(self, model, feature_names=None) -> list[str]:
        if feature_names is not None:
            return list(feature_names)

        names = getattr(model, "feature_names_", None)
        if names:
            return [f"f{int(name)}" if str(name).isdigit() else str(name) for name in names]

        n_features = getattr(model, "n_features_in_", None) or getattr(
            model, "_n_features_in", None
        )
        return resolve_feature_names(model, n_features=n_features)

    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None):
        feature_names_resolved = self.get_feature_names(model, feature_names=feature_names)
        model_dump = self._dump_model_json(model)
        tree_specs = model_dump["oblivious_trees"]
        if max_trees is not None:
            tree_specs = tree_specs[:max_trees]

        trees = [
            self._normalize_oblivious_tree(tree_spec, feature_names_resolved, tree_index)
            for tree_index, tree_spec in enumerate(tree_specs)
        ]

        return NormalizedEnsemble(trees=trees, backend="catboost", model_type=type(model).__name__)

    def metric_capabilities(self) -> MetricCapabilities:
        return {
            "gain": MetricCapability(
                status="approximate",
                detail="derived from weighted variance reduction over CatBoost leaf values",
            ),
            "cover": MetricCapability(
                status="approximate", detail="derived from CatBoost leaf weights"
            ),
            "weight": MetricCapability(status="exact", detail="split count"),
            "total_gain": MetricCapability(
                status="synthetic",
                detail="sum of CatBoost gain proxies derived from exported leaf values",
            ),
            "total_cover": MetricCapability(
                status="approximate", detail="sum of cover derived from CatBoost leaf weights"
            ),
            "average_cover": MetricCapability(
                status="approximate", detail="average cover derived from CatBoost leaf weights"
            ),
        }

    def supports_categorical_splits(self) -> bool:
        """Return whether CatBoost categorical split normalization is implemented."""
        return False

    def _dump_model_json(self, model) -> dict:
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "catboost-model.json"
            model.save_model(model_path, format="json")
            return json.loads(model_path.read_text())

    def _normalize_oblivious_tree(
        self,
        tree_spec: dict,
        feature_names: list[str],
        tree_index: int,
    ) -> NormalizedTree:
        splits = list(tree_spec.get("splits", []))
        leaf_values = [float(value) for value in tree_spec.get("leaf_values", [])]
        leaf_weights = [float(value) for value in tree_spec.get("leaf_weights", [])]
        depth = len(splits)
        leaf_start = (1 << depth) - 1
        nodes: list[NormalizedNode] = []

        for level, split in enumerate(splits):
            start = (1 << level) - 1
            stop = (1 << (level + 1)) - 1
            feature_name, split_condition = _parse_catboost_json_split(split, feature_names)
            for node_id in range(start, stop):
                leaf_offset = node_id - start
                span = 1 << (depth - level)
                weight_start = leaf_offset * span
                weight_stop = weight_start + span
                nodes.append(
                    NormalizedNode(
                        node_id=node_id,
                        is_leaf=False,
                        feature=feature_name,
                        split_condition=split_condition,
                        gain=self._catboost_gain_proxy(
                            leaf_values=leaf_values,
                            leaf_weights=leaf_weights,
                            start=weight_start,
                            stop=weight_stop,
                        ),
                        cover=sum(leaf_weights[weight_start:weight_stop]),
                        left_child=(2 * node_id) + 1,
                        right_child=(2 * node_id) + 2,
                    )
                )

        for leaf_index, leaf_value in enumerate(leaf_values):
            nodes.append(
                NormalizedNode(
                    node_id=leaf_start + leaf_index,
                    is_leaf=True,
                    leaf_value=leaf_value,
                    cover=leaf_weights[leaf_index] if leaf_index < len(leaf_weights) else None,
                )
            )

        return NormalizedTree(tree_index=tree_index, root_id=0, nodes=nodes)

    def _catboost_gain_proxy(
        self,
        *,
        leaf_values: list[float],
        leaf_weights: list[float],
        start: int,
        stop: int,
    ) -> float:
        midpoint = start + ((stop - start) // 2)
        parent_sse = self._weighted_leaf_sse(
            leaf_values=leaf_values,
            leaf_weights=leaf_weights,
            start=start,
            stop=stop,
        )
        left_sse = self._weighted_leaf_sse(
            leaf_values=leaf_values,
            leaf_weights=leaf_weights,
            start=start,
            stop=midpoint,
        )
        right_sse = self._weighted_leaf_sse(
            leaf_values=leaf_values,
            leaf_weights=leaf_weights,
            start=midpoint,
            stop=stop,
        )
        return max(parent_sse - left_sse - right_sse, 0.0)

    def _weighted_leaf_sse(
        self,
        *,
        leaf_values: list[float],
        leaf_weights: list[float],
        start: int,
        stop: int,
    ) -> float:
        weights = leaf_weights[start:stop]
        values = leaf_values[start:stop]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return 0.0
        mean = (
            sum(weight * value for weight, value in zip(weights, values, strict=False))
            / total_weight
        )
        return sum(
            weight * ((value - mean) ** 2) for weight, value in zip(weights, values, strict=False)
        )


def _parse_catboost_split(split: str, feature_names: list[str]) -> tuple[str, str]:
    feature_token = split.split(",", 1)[0].strip()
    feature_name = feature_token
    if feature_token.isdigit():
        feature_index = int(feature_token)
        if 0 <= feature_index < len(feature_names):
            feature_name = feature_names[feature_index]
            if feature_name == str(feature_index):
                feature_name = f"f{feature_index}"
        else:
            feature_name = f"f{feature_index}"
    return feature_name, split


def _parse_catboost_json_split(split: dict, feature_names: list[str]) -> tuple[str, str]:
    if split.get("split_type") != "FloatFeature":
        return str(split.get("split_type", "unknown")), str(split)

    feature_index = int(split["float_feature_index"])
    feature_name = (
        feature_names[feature_index]
        if 0 <= feature_index < len(feature_names)
        else f"f{feature_index}"
    )
    if str(feature_name).isdigit():
        feature_name = f"f{feature_name}"
    return feature_name, f"< {float(split['border'])}"


class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM boosters using structured model dumps."""

    def can_handle(self, model) -> bool:
        return type(model).__module__.startswith("lightgbm")

    def is_fitted(self, model) -> bool:
        return self.can_handle(model)

    def get_feature_names(self, model, feature_names=None) -> list[str]:
        if feature_names is not None:
            return list(feature_names)
        booster = self._get_booster(model)
        model_dump = booster.dump_model()
        return list(model_dump.get("feature_names", []))

    def to_normalized_ensemble(self, model, feature_names=None, max_trees=None):
        booster = self._get_booster(model)
        model_dump = booster.dump_model()
        feature_names_resolved = self.get_feature_names(model, feature_names=feature_names)
        tree_info = model_dump["tree_info"]
        if max_trees is not None:
            tree_info = tree_info[:max_trees]

        trees = [
            self._normalize_tree_info(
                tree["tree_structure"], feature_names_resolved, tree_index=tree["tree_index"]
            )
            for tree in tree_info
        ]
        return NormalizedEnsemble(trees=trees, backend="lightgbm", model_type=type(model).__name__)

    def metric_capabilities(self) -> MetricCapabilities:
        return {
            "gain": MetricCapability(status="exact"),
            "cover": MetricCapability(
                status="approximate", detail="derived from LightGBM counts/weights"
            ),
            "weight": MetricCapability(status="exact", detail="split count"),
            "total_gain": MetricCapability(status="exact"),
            "total_cover": MetricCapability(
                status="approximate", detail="sum of LightGBM exported counts/weights"
            ),
            "average_cover": MetricCapability(
                status="approximate", detail="average LightGBM exported counts/weights"
            ),
        }

    def _get_booster(self, model):
        if hasattr(model, "dump_model"):
            return model
        if hasattr(model, "booster_"):
            return model.booster_
        raise UnsupportedModelError(f"treefi does not support model type: {type(model).__name__}")

    def _normalize_tree_info(
        self, tree_structure: dict, feature_names: list[str], tree_index: int
    ) -> NormalizedTree:
        nodes: list[NormalizedNode] = []

        def visit(node: dict) -> int:
            if "leaf_value" in node:
                node_id = int(node["leaf_index"]) + 1_000_000
                nodes.append(
                    NormalizedNode(
                        node_id=node_id,
                        is_leaf=True,
                        leaf_value=float(node["leaf_value"]),
                        cover=float(node.get("leaf_count", 0.0)),
                    )
                )
                return node_id

            node_id = int(node["split_index"])
            left_child = visit(node["left_child"])
            right_child = visit(node["right_child"])
            feature_index = int(node["split_feature"])
            feature_name = (
                feature_names[feature_index]
                if 0 <= feature_index < len(feature_names)
                else f"f{feature_index}"
            )
            nodes.append(
                NormalizedNode(
                    node_id=node_id,
                    is_leaf=False,
                    feature=feature_name,
                    split_condition=f"{node.get('decision_type', '<=')} {node['threshold']}",
                    gain=float(node.get("split_gain", 0.0)),
                    cover=float(node.get("internal_count", 0.0)),
                    left_child=left_child,
                    right_child=right_child,
                )
            )
            return node_id

        root_id = visit(tree_structure)
        return NormalizedTree(
            tree_index=tree_index,
            root_id=root_id,
            nodes=sorted(nodes, key=lambda node: node.node_id),
        )


def get_adapter_for_model(model, adapters: list[ModelAdapter] | None = None) -> ModelAdapter:
    """Return the first adapter that can handle the provided model."""
    for adapter in adapters or []:
        if adapter.can_handle(model):
            return adapter
    raise UnsupportedModelError(f"treefi does not support model type: {type(model).__name__}")


def resolve_feature_names(model, feature_names=None, n_features: int | None = None) -> list[str]:
    """Resolve feature names from explicit input, model metadata, or generated fallbacks."""
    if feature_names is not None:
        return list(feature_names)

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        return list(model_feature_names)

    if n_features is None:
        return []

    return [f"f{i}" for i in range(n_features)]
