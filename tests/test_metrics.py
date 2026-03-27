import math
import sys
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import pytest

from treefi.api import _aggregate_importance_rows
from treefi.metrics import summarize_interactions
from treefi.models import NormalizedNode, NormalizedTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xgbfir"))
import xgboost as xgb  # noqa: E402

if find_spec("xgbfir") is not None:
    import xgbfir  # type: ignore  # noqa: E402
else:  # pragma: no cover - depends on local reference checkout
    xgbfir = None


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
    if xgbfir is None:
        pytest.skip("xgbfir reference package is not available in this checkout")

    dtrain = xgb.DMatrix(
        [[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["f0"]
    )
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


def test_feature_importance_aggregation_recomputes_mean_style_metrics_correctly() -> None:
    frame = _aggregate_importance_rows(
        pd.DataFrame(
            [
                {
                    "feature": "f0",
                    "gain": 10.0,
                    "cover": 100.0,
                    "fscore": 2,
                    "weighted_fscore": 0.5,
                    "average_weighted_fscore": 0.25,
                    "average_gain": 5.0,
                    "expected_gain": 3.0,
                    "average_tree_index": 1.0,
                    "average_tree_depth": 0.0,
                    "path_frequency": 2,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 2,
                    "tree_count": 1.0,
                },
                {
                    "feature": "f0",
                    "gain": 4.0,
                    "cover": 40.0,
                    "fscore": 1,
                    "weighted_fscore": 1.0,
                    "average_weighted_fscore": 1.0,
                    "average_gain": 4.0,
                    "expected_gain": 4.0,
                    "average_tree_index": 5.0,
                    "average_tree_depth": 2.0,
                    "path_frequency": 1,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 1,
                    "tree_count": 1.0,
                },
            ]
        )
    )

    row = frame.iloc[0]
    assert row["gain"] == pytest.approx(14.0 / 3.0)
    assert row["weight"] == 3
    assert row["total_gain"] == 14.0
    assert row["total_cover"] == 140.0
    assert row["weighted_fscore"] == 1.5
    assert row["average_weighted_fscore"] == pytest.approx(0.5)
    assert row["cover"] == pytest.approx(140.0 / 3.0)
    assert row["average_tree_index"] == pytest.approx(7.0 / 3.0)
    assert row["average_tree_depth"] == pytest.approx(2.0 / 3.0)


def test_feature_importance_aggregation_uses_deterministic_tie_breaking() -> None:
    frame = _aggregate_importance_rows(
        pd.DataFrame(
            [
                {
                    "feature": "b",
                    "gain": 1.0,
                    "cover": 1.0,
                    "fscore": 1,
                    "weighted_fscore": 1.0,
                    "average_weighted_fscore": 1.0,
                    "average_gain": 1.0,
                    "expected_gain": 1.0,
                    "average_tree_index": 0.0,
                    "average_tree_depth": 0.0,
                    "path_frequency": 1,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 1,
                    "tree_count": 1.0,
                },
                {
                    "feature": "a",
                    "gain": 1.0,
                    "cover": 1.0,
                    "fscore": 1,
                    "weighted_fscore": 1.0,
                    "average_weighted_fscore": 1.0,
                    "average_gain": 1.0,
                    "expected_gain": 1.0,
                    "average_tree_index": 0.0,
                    "average_tree_depth": 0.0,
                    "path_frequency": 1,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 1,
                    "tree_count": 1.0,
                },
            ]
        )
    ).sort_values(by=["gain", "feature"], ascending=[False, True], kind="mergesort")

    assert frame["feature"].tolist() == ["a", "b"]


def test_feature_importance_aggregation_remains_directionally_consistent_with_xgbfir() -> None:
    if xgbfir is None:
        pytest.skip("xgbfir reference package is not available in this checkout")

    dtrain = xgb.DMatrix(
        [[0.0], [1.0], [2.0], [3.0]], label=[0.0, 0.0, 1.0, 1.0], feature_names=["f0"]
    )
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 1, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=2,
    )

    parser = xgbfir.main.XgbModelParser()
    dump = booster.get_dump("", with_stats=True)
    xgb_model = parser.GetXgbModelFromMemory(dump, maxTrees=100)
    xgb_interactions = xgb_model.GetFeatureInteractions(maxInteractionDepth=0, maxDeepening=-1)
    feature_rows = []
    for feature_name, interaction in xgb_interactions.interactions.items():
        feature_rows.append(
            {
                "feature": feature_name,
                "gain": interaction.Gain,
                "cover": interaction.Cover,
                "fscore": interaction.FScore,
                "weighted_fscore": interaction.FScoreWeighted,
                "average_weighted_fscore": interaction.AverageFScoreWeighted,
                "average_gain": interaction.AverageGain,
                "expected_gain": interaction.ExpectedGain,
                "average_tree_index": interaction.AverageTreeIndex,
                "average_tree_depth": interaction.AverageTreeDepth,
                "path_frequency": interaction.FScore,
                "tree_frequency": 1.0,
                "backend": "xgboost",
                "model_type": "Booster",
                "occurrence_count": interaction.FScore,
                "tree_count": 1.0,
            }
        )

    frame = _aggregate_importance_rows(pd.DataFrame(feature_rows))
    row = frame.loc[frame["feature"] == "f0"].iloc[0]
    reference = xgb_interactions.interactions["f0"]

    assert row["gain"] == reference.AverageGain
    assert row["weight"] == reference.FScore
    assert row["total_gain"] == reference.Gain
    assert row["cover"] == pytest.approx(reference.Cover / reference.FScore)
    assert row["total_cover"] == reference.Cover
    assert row["weighted_fscore"] == reference.FScoreWeighted
    assert row["expected_gain"] == reference.ExpectedGain


def test_xgboost_compatible_alias_columns_match_booster_importance_types() -> None:
    dtrain = xgb.DMatrix(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        label=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        feature_names=["f0"],
    )
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 2, "eta": 1.0},
        dtrain=dtrain,
        num_boost_round=3,
    )

    from treefi import feature_importance

    frame = feature_importance(booster)
    row = frame.loc[frame["feature"] == "f0"].iloc[0]

    xgb_weight = booster.get_score(importance_type="weight")["f0"]
    xgb_gain = booster.get_score(importance_type="gain")["f0"]
    xgb_cover = booster.get_score(importance_type="cover")["f0"]
    xgb_total_gain = booster.get_score(importance_type="total_gain")["f0"]
    xgb_total_cover = booster.get_score(importance_type="total_cover")["f0"]

    assert row["weight"] == xgb_weight
    assert row["gain"] == pytest.approx(xgb_gain)
    assert row["total_gain"] == pytest.approx(xgb_total_gain)
    assert row["cover"] == pytest.approx(xgb_cover)
    assert row["total_cover"] == pytest.approx(xgb_total_cover)


def test_feature_importance_accepts_non_breaking_xgboost_style_sort_aliases() -> None:
    frame = _aggregate_importance_rows(
        pd.DataFrame(
            [
                {
                    "feature": "a",
                    "gain": 10.0,
                    "cover": 20.0,
                    "fscore": 2,
                    "weighted_fscore": 1.0,
                    "average_weighted_fscore": 0.5,
                    "average_gain": 5.0,
                    "expected_gain": 4.0,
                    "average_tree_index": 0.0,
                    "average_tree_depth": 0.0,
                    "path_frequency": 2,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 2,
                    "tree_count": 1.0,
                },
                {
                    "feature": "b",
                    "gain": 12.0,
                    "cover": 3.0,
                    "fscore": 1,
                    "weighted_fscore": 1.0,
                    "average_weighted_fscore": 1.0,
                    "average_gain": 12.0,
                    "expected_gain": 12.0,
                    "average_tree_index": 0.0,
                    "average_tree_depth": 0.0,
                    "path_frequency": 1,
                    "tree_frequency": 1.0,
                    "backend": "x",
                    "model_type": "m",
                    "occurrence_count": 1,
                    "tree_count": 1.0,
                },
            ]
        )
    )

    assert frame.sort_values("gain", ascending=False)["feature"].tolist() == ["b", "a"]
    assert frame.sort_values("total_gain", ascending=False)["feature"].tolist() == ["b", "a"]
    assert frame.sort_values("weight", ascending=False)["feature"].tolist() == ["a", "b"]
    assert frame.sort_values("cover", ascending=False)["feature"].tolist() == ["a", "b"]


def test_feature_importance_sort_aliases_map_legacy_names_to_new_canonical_columns() -> None:
    from sklearn.ensemble import RandomForestRegressor

    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    model = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0).fit(X, y)

    from treefi import feature_importance

    by_gain = feature_importance(model, sort_by="gain")
    by_average_gain = feature_importance(model, sort_by="average_gain")
    by_cover = feature_importance(model, sort_by="cover")
    by_average_cover = feature_importance(model, sort_by="average_cover")
    by_weight = feature_importance(model, sort_by="weight")
    by_fscore = feature_importance(model, sort_by="fscore")

    assert by_gain["feature"].tolist() == by_average_gain["feature"].tolist()
    assert by_cover["feature"].tolist() == by_average_cover["feature"].tolist()
    assert by_weight["feature"].tolist() == by_fscore["feature"].tolist()
