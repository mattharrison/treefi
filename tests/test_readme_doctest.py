from __future__ import annotations

import doctest
from pathlib import Path


def _pycon_blocks(markdown: str) -> str:
    blocks: list[str] = []
    inside_block = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if not inside_block and stripped == "```pycon":
            inside_block = True
            continue
        if inside_block and stripped == "```":
            inside_block = False
            continue
        if inside_block:
            blocks.append(line)

    return "\n".join(blocks)


def test_readme_pycon_examples_run_as_doctests() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    examples = _pycon_blocks(readme)

    assert examples.strip(), "README.md must contain at least one runnable pycon block"

    parser = doctest.DocTestParser()
    doctest_case = parser.get_doctest(
        examples,
        globs={},
        name="README.md",
        filename="README.md",
        lineno=0,
    )
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    runner.run(doctest_case)
    result = runner.summarize(verbose=False)

    assert result.failed == 0
