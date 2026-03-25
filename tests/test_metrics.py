import math
import sys
from pathlib import Path

import pytest

from treefi.metrics import summarize_interactions
from treefi.models import NormalizedNode, NormalizedTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xgbfir"))
import xgboost as xgb  # noqa: E402

import xgbfir  # type: ignore  # noqa: E402


def make_metric_tree() -> NormalizedTree:
    return NormalizedTree(
        tree_index=2,
        root_id=0,
        nodes=[
            NormalizedNode(
                node_id=0,
                feature="f0",
                split_condition="< 0.5",
                gain=10.0,
                cover=100.0,
                left_child=1,
                right_child=1,
                is_leaf=False,
            ),
            NormalizedNode(
                node_id=1,
                feature="f1",
                split_condition="< 0.5",
                gain=4.0,
                cover=40.0,
                left_child=2,
                right_child=2,
                is_leaf=False,
            ),
            NormalizedNode(node_id=2, is_leaf=True, leaf_value=0.25, cover=20.0),
        ],
    )


def test_summarize_interactions_computes_legacy_xgbfir_style_metrics() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1)

    assert frame["interaction"].tolist() == ["f0", "f0|f1", "f1"]

    root = frame.loc[frame["interaction"] == "f0"].iloc[0]
    pair = frame.loc[frame["interaction"] == "f0|f1"].iloc[0]
    child = frame.loc[frame["interaction"] == "f1"].iloc[0]

    assert root["gain"] == 10.0
    assert root["cover"] == 100.0
    assert root["fscore"] == 1
    assert root["weighted_fscore"] == 1.0
    assert root["average_weighted_fscore"] == 1.0
    assert root["average_gain"] == 10.0
    assert root["expected_gain"] == 10.0
    assert root["average_tree_index"] == 2.0
    assert root["average_tree_depth"] == 0.0

    assert pair["gain"] == 14.0
    assert pair["cover"] == 140.0
    assert pair["fscore"] == 1
    assert pair["weighted_fscore"] == pytest.approx(0.4)
    assert pair["average_gain"] == 14.0
    assert pair["expected_gain"] == pytest.approx(5.6)
    assert pair["average_tree_depth"] == 1.0

    assert child["gain"] == 4.0
    assert child["cover"] == 40.0
    assert child["weighted_fscore"] == pytest.approx(0.4)
    assert child["expected_gain"] == pytest.approx(1.6)
    assert child["average_tree_depth"] == 1.0


def test_summarize_interactions_computes_new_v1_interaction_metrics() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1)
    pair = frame.loc[frame["interaction"] == "f0|f1"].iloc[0]

    assert pair["interaction_order"] == 2
    assert pair["tree_frequency"] == 1.0
    assert pair["path_frequency"] == 1
    assert pair["first_position_mean"] == 0.0
    assert pair["min_depth"] == 1
    assert pair["max_depth"] == 1
    assert pair["leaf_effect_mean"] == 0.25
    assert pair["leaf_effect_var"] == 0.0


def test_summarize_interactions_uses_nan_for_unavailable_leaf_effect_metrics() -> None:
    tree = NormalizedTree(
        tree_index=0,
        root_id=0,
        nodes=[
            NormalizedNode(
                node_id=0,
                feature="f0",
                split_condition="< 0.5",
                gain=1.0,
                cover=5.0,
                left_child=1,
                right_child=None,
                is_leaf=False,
            ),
            NormalizedNode(
                node_id=1,
                feature="f1",
                split_condition="< 0.5",
                gain=1.0,
                cover=3.0,
                left_child=None,
                right_child=None,
                is_leaf=False,
            ),
        ],
    )

    frame = summarize_interactions(tree, max_interaction_depth=1)
    pair = frame.loc[frame["interaction"] == "f0|f1"].iloc[0]

    assert math.isnan(pair["leaf_effect_mean"])
    assert math.isnan(pair["leaf_effect_var"])


def test_summarize_interactions_adds_rank_columns() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1)

    assert "rank_gain" in frame.columns
    assert "rank_fscore" in frame.columns
    assert "rank_expected_gain" in frame.columns
    assert "rank_consensus" in frame.columns

    assert frame["rank_gain"].tolist() == [2, 1, 3]


def test_summarize_interactions_supports_sorting_by_metric() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1, sort_by="gain")

    assert frame["interaction"].tolist() == ["f0|f1", "f0", "f1"]


def test_summarize_interactions_supports_top_k() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1, sort_by="gain", top_k=2)

    assert frame["interaction"].tolist() == ["f0|f1", "f0"]


def test_summarize_interactions_supports_min_fscore_filtering() -> None:
    tree = make_metric_tree()

    frame = summarize_interactions(tree, max_interaction_depth=1, min_fscore=2)

    assert frame.empty


def test_summarize_interactions_uses_deterministic_tie_breaking() -> None:
    tree = NormalizedTree(
        tree_index=0,
        root_id=0,
        nodes=[
            NormalizedNode(
                node_id=0,
                feature="b",
                split_condition="< 0.5",
                gain=1.0,
                cover=1.0,
                left_child=1,
                right_child=1,
                is_leaf=False,
            ),
            NormalizedNode(
                node_id=1,
                feature="a",
                split_condition="< 0.5",
                gain=1.0,
                cover=1.0,
                left_child=2,
                right_child=2,
                is_leaf=False,
            ),
            NormalizedNode(node_id=2, is_leaf=True, leaf_value=0.0),
        ],
    )

    frame = summarize_interactions(tree, max_interaction_depth=0, sort_by="gain")

    assert frame["interaction"].tolist() == ["a", "b"]


def test_xgboost_metrics_remain_directionally_consistent_with_xgbfir() -> None:
    dtrain = xgb.DMatrix([[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["f0"])
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 1, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=1,
    )

    parser = xgbfir.main.XgbModelParser()
    dump = booster.get_dump("", with_stats=True)
    xgb_model = parser.GetXgbModelFromMemory(dump, maxTrees=100)
    xgb_interactions = xgb_model.GetFeatureInteractions(maxInteractionDepth=0, maxDeepening=-1)
    xgbfir_gain = xgb_interactions.interactions["f0"].Gain
    xgbfir_fscore = xgb_interactions.interactions["f0"].FScore
    xgbfir_weighted_fscore = xgb_interactions.interactions["f0"].FScoreWeighted
    xgbfir_expected_gain = xgb_interactions.interactions["f0"].ExpectedGain

    frame = summarize_interactions(
        NormalizedTree(
            tree_index=0,
            root_id=0,
            nodes=[
                NormalizedNode(
                    node_id=0,
                    feature="f0",
                    split_condition="< 2.0",
                    gain=xgbfir_gain,
                    cover=4.0,
                    left_child=1,
                    right_child=2,
                    is_leaf=False,
                ),
                NormalizedNode(node_id=1, is_leaf=True, leaf_value=-0.333333, cover=2.0),
                NormalizedNode(node_id=2, is_leaf=True, leaf_value=0.333333, cover=2.0),
            ],
        ),
        max_interaction_depth=0,
    )

    assert frame.iloc[0]["gain"] == xgbfir_gain
    assert frame.iloc[0]["fscore"] == xgbfir_fscore
    assert frame.iloc[0]["weighted_fscore"] == xgbfir_weighted_fscore
    assert frame.iloc[0]["expected_gain"] == xgbfir_expected_gain
