from pathlib import Path


def test_readme_mentions_quickstart_and_supported_backends() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "uv run pytest" in readme
    assert "feature_interactions" in readme
    assert "scikit-learn" in readme
    assert "XGBoost" in readme
    assert "CatBoost" in readme


def test_readme_includes_xgbfir_migration_example() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "xgbfir.saveXgbFI" in readme
    assert "treefi.feature_interactions" in readme


def test_readme_documents_get_score_mapping_and_backend_neutral_guidance() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "get_score" in readme
    assert "backend-neutral" in readme
    assert "total_gain" in readme
    assert "weight" in readme


def test_readme_documents_suspicious_feature_taxonomy() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "unstable features" in readme
    assert "low-density features" in readme
    assert "weak-consensus features" in readme
    assert "cv_instability_flag" in readme
    assert "high_weight_low_gain_flag" in readme
    assert "low_consensus_top_k_flag" in readme


def test_readme_separates_compatibility_and_treefi_native_metrics() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Compatibility Names Vs treefi-native Names" in readme
    assert "compatibility names" in readme
    assert "treefi-native names" in readme
    assert "Booster.get_score" in readme
    assert "expected_gain" in readme
