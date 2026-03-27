from pathlib import Path

import yaml


def test_prek_config_enables_relevant_hooks() -> None:
    config_path = Path(".pre-commit-config.yaml")

    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repos = config["repos"]
    hook_ids = {
        hook["id"]
        for repo in repos
        for hook in repo.get("hooks", [])
    }

    assert "ruff-check" in hook_ids
    assert "ruff-format" in hook_ids
    assert "check-yaml" in hook_ids
    assert "end-of-file-fixer" in hook_ids
    assert "trailing-whitespace" in hook_ids
    assert "check-merge-conflict" in hook_ids

    ruff_hooks = [
        hook
        for repo in repos
        for hook in repo.get("hooks", [])
        if hook["id"] in {"ruff-check", "ruff-format"}
    ]
    assert ruff_hooks
    assert all(hook.get("types_or") == ["python", "pyi"] for hook in ruff_hooks)


def test_readme_or_contributing_documents_prek_usage() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, contributing])

    assert "prek" in combined
    assert "uv tool run prek run --all-files" in combined or "uvx prek run --all-files" in combined
    assert "notebook" in combined.lower()
