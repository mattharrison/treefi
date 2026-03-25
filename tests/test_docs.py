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
