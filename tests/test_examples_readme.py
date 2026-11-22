"""Sync guards for ``examples/README.md``.

The index must list every example script exactly once and must not reference
scripts that do not exist — the same consistency discipline as the #53
report checker, applied to the examples directory itself.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
README = EXAMPLES_DIR / "README.md"
PROJECT_BOARD = REPO_ROOT / "docs" / "PROJECT.md"
PHASE4_REPORT = REPO_ROOT / "docs" / "PHASE4_RESULTS.md"

BACKTICKED_PY = re.compile(r"`([A-Za-z0-9_]+\.py)`")


def _example_scripts():
    return sorted(
        p.name for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("__")
    )


def _readme_text():
    assert README.is_file(), "examples/README.md is missing"
    return README.read_text(encoding="utf-8")


def test_every_example_script_is_listed():
    text = _readme_text()
    for name in _example_scripts():
        assert f"`{name}`" in text, f"{name} is not indexed in examples/README.md"


def test_no_dangling_example_references():
    text = _readme_text()
    scripts = set(_example_scripts())
    for token in BACKTICKED_PY.findall(text):
        assert token in scripts, (
            f"examples/README.md references `{token}`, which does not exist"
        )


def test_phase4_table_structure():
    text = _readme_text()
    header = "| Example | Issue / PR | What it shows | Runtime |"
    assert header in text, "Phase 4 index table header changed or missing"
    separator = "|---|---|---|---|"
    assert separator in text, "Phase 4 index table separator row missing"
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| `") and line.endswith("|")
    ]
    # Ten Phase 4 tooling rows + two "Other" rows (basic usage, #17
    # memory walkthrough).
    assert len(rows) == len(_example_scripts()) == 12
    other_section = text.split("## Other", 1)[1]
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        expected = 2 if row in other_section else 4
        assert len(cells) == expected, f"malformed table row: {row}"
        assert BACKTICKED_PY.fullmatch(cells[0]), f"bad example cell: {cells[0]}"


def test_conventions_are_documented():
    text = _readme_text()
    for needle in (
        "PYTHONPATH",
        "Honest negatives",
        "no wall-clock",
        "GDN-2 and EDA",
    ):
        assert needle in text, f"examples/README.md dropped the {needle!r} note"


def test_project_board_links_index_and_report():
    """docs/PROJECT.md must point at the examples index and the report.

    The Phase 4 epic table is the entry point most contributors read first;
    if the links drift (renamed files, moved docs) newcomers land on 404s.
    """
    text = PROJECT_BOARD.read_text(encoding="utf-8")
    assert "](../examples/README.md)" in text, (
        "docs/PROJECT.md no longer links the examples index"
    )
    assert "](PHASE4_RESULTS.md)" in text, (
        "docs/PROJECT.md no longer links the Phase 4 report"
    )
    assert PHASE4_REPORT.is_file(), "docs/PHASE4_RESULTS.md is missing"
    assert "`examples/phase4_results_report.py`" in text, (
        "docs/PROJECT.md no longer names the report generator script"
    )
