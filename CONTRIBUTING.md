# Contributing

## Workflow

This project uses test-driven development.

Required loop:

1. Write a failing test first.
2. Implement the smallest change that makes the test pass.
3. Refactor only after the test suite is green.

## Environment

Use `uv` for local development.

Common commands:

- `uv run pytest`
- `uv run pytest tests/test_adapters.py -q`
- `uv run pytest tests/test_sklearn_integration.py -q`

## Scope Rules

- Keep the public API dataframe-first.
- Use vendored `xgbfir/` for ideas, parity checks, and logic reference only.
- Do not turn `xgbfir` into a runtime dependency or copy its Excel-centric design into `treefi`.
- Preserve backend caveats explicitly instead of hiding approximations behind generic names.

## Planning

Before starting new implementation work, check:

- `context/PRD.md`
- `context/TASKS.md`
- `AGENTS.md`
