"""Audit the remaining generated tables in the Phase 4 report (#34).

``examples/report_consistency.py`` cross-checks the report against the
#50/#51/#52/#54 extension sweeps. This module covers the complementary set:
the tables rendered directly from the #34 capacity/interference and
overwrite/recency experiment configs that are *not* duplicated in those
extension sweeps — the interference sweep, fresh-recall and stale-dominance
tables, and the two edit-locality tables.

The audit recomputes the checked-in default :class:`CapacityConfig` and
:class:`OverwriteConfig`, parses ``docs/PHASE4_RESULTS.md``, and compares the
formatted strings exactly as a reader sees them (fixed three-decimal style).
Mismatches are reported, never corrected.

Scope: synthetic deterministic experiments only; no production-performance or
kernel-level claims are involved.
"""

from __future__ import annotations

from .report_consistency import (
    format_report_ci,
    format_report_float,
    parse_pipe_tables,
)
from .report_experiments import (
    CapacityConfig,
    OverwriteConfig,
    run_capacity_experiment,
    run_overwrite_experiment,
)

__all__ = [
    "FRESH_RECALL_HEADING",
    "KEY_AXIS_ALIGNMENT_HEADING",
    "RELATIVE_UPDATE_NORM_HEADING",
    "STALE_DOMINANCE_HEADING",
    "audit_report_tables",
    "interference_expectations",
    "interference_heading",
    "overwrite_expectations",
]

FRESH_RECALL_HEADING = "### Fresh recall accuracy"
STALE_DOMINANCE_HEADING = "### Stale-dominance rate"
RELATIVE_UPDATE_NORM_HEADING = "### Relative update norm"
KEY_AXIS_ALIGNMENT_HEADING = "### Key-axis alignment"


def interference_heading(capacity_report: dict) -> str:
    """Heading of the report's interference sweep for this config."""

    variants = list(capacity_report["interference_sweep"])
    num_pairs = capacity_report["interference_sweep"][variants[0]]["num_pairs"]
    return f"### Interference sweep (fixed load {num_pairs} pairs)"


def interference_expectations(capacity_report: dict) -> dict[tuple[str, str], str]:
    """Expected interference-table cells from a capacity experiment report.

    Maps ``(variant, "cos=<collision cosine>")`` to the report-formatted
    recall-accuracy string ``value [ci]``.
    """

    expectations: dict[tuple[str, str], str] = {}
    for variant, section in capacity_report["interference_sweep"].items():
        for point in section["points"]:
            column = f"cos={format_report_float(point['collision_cosine'])}"
            expectations[(variant, column)] = format_report_ci(
                point["recall_accuracy"]
            )
    return expectations


def overwrite_expectations(
    overwrite_report: dict,
) -> dict[str, dict[tuple[str, str], str]]:
    """Expected overwrite/locality table cells from an overwrite report.

    Returns one ``(variant, "updates=<n>") -> formatted text`` mapping per
    report table: fresh recall, stale dominance, relative update norm, and
    key-axis alignment.
    """

    tables: dict[str, dict[tuple[str, str], str]] = {
        "fresh": {},
        "stale": {},
        "relative": {},
        "alignment": {},
    }
    for variant, section in overwrite_report["variants"].items():
        for point in section["points"]:
            column = f"updates={point['num_updates']}"
            tables["fresh"][(variant, column)] = format_report_ci(
                point["fresh_recall_accuracy"]
            )
            tables["stale"][(variant, column)] = format_report_ci(
                point["stale_dominance_rate"]
            )
            tables["relative"][(variant, column)] = format_report_float(
                point["edit_relative_update_norm"]
            )
            tables["alignment"][(variant, column)] = format_report_float(
                point["edit_key_axis_alignment"]
            )
    return tables


def _table_by_heading(tables: dict[str, list[dict]], heading: str) -> dict | None:
    """The first pipe table under ``heading``, or ``None`` if absent."""

    found = tables.get(heading)
    return found[0] if found else None


def _compare_cells(
    checks: list[dict],
    *,
    table_name: str,
    table: dict | None,
    expectations: dict[tuple[str, str], str],
) -> None:
    """Record one check per expected cell against a parsed markdown table.

    Missing tables, rows, and columns are failed checks with explanatory
    notes rather than exceptions, so one structural problem does not hide the
    remaining comparisons.
    """

    for (row_label, column_label), expected in sorted(expectations.items()):
        check = {
            "table": table_name,
            "row": row_label,
            "column": column_label,
            "expected": expected,
            "actual": None,
            "match": False,
        }
        if table is None:
            check["note"] = f"table '{table_name}' not found in the report"
        else:
            header = table["header"]
            rows = {row[0]: row for row in table["rows"] if row}
            if column_label not in header:
                check["note"] = f"column '{column_label}' not found"
            elif row_label not in rows:
                check["note"] = f"row '{row_label}' not found"
            else:
                actual = rows[row_label][header.index(column_label)]
                check["actual"] = actual
                check["match"] = actual == expected
        checks.append(check)


def audit_report_tables(
    markdown: str,
    *,
    capacity_report: dict | None = None,
    overwrite_report: dict | None = None,
) -> dict:
    """Cross-check the report tables rendered directly from the #34 configs.

    Args:
        markdown: Text of the report to audit (normally the checked-in
            ``docs/PHASE4_RESULTS.md``).
        capacity_report: Optional result of
            :func:`run_capacity_experiment`; defaults to a fresh run of the
            checked-in :class:`CapacityConfig`.
        overwrite_report: Optional result of
            :func:`run_overwrite_experiment`; defaults to a fresh run of the
            checked-in :class:`OverwriteConfig`.

    Returns:
        A JSON-serializable dict with one entry per compared cell and an
        ``all_match`` summary. The default audit covers 73 cells: 25
        interference cells, 24 overwrite-rate cells, and 24 edit-locality
        cells.
    """

    capacity_report = capacity_report or run_capacity_experiment(CapacityConfig())
    overwrite_report = overwrite_report or run_overwrite_experiment(OverwriteConfig())
    parsed = parse_pipe_tables(markdown)
    checks: list[dict] = []

    heading = interference_heading(capacity_report)
    _compare_cells(
        checks,
        table_name=heading,
        table=_table_by_heading(parsed, heading),
        expectations=interference_expectations(capacity_report),
    )

    overwrite_tables = overwrite_expectations(overwrite_report)
    for table_name, key in (
        (FRESH_RECALL_HEADING, "fresh"),
        (STALE_DOMINANCE_HEADING, "stale"),
        (RELATIVE_UPDATE_NORM_HEADING, "relative"),
        (KEY_AXIS_ALIGNMENT_HEADING, "alignment"),
    ):
        _compare_cells(
            checks,
            table_name=table_name,
            table=_table_by_heading(parsed, table_name),
            expectations=overwrite_tables[key],
        )

    mismatches = [check for check in checks if not check["match"]]
    return {
        "experiment": "phase4-report-table-audit",
        "issue": 34,
        "description": (
            "Cross-check the Phase 4 report's interference, overwrite/"
            "recency, and edit-locality tables against fresh recomputation "
            "of the checked-in #34 default configs. Compared as formatted "
            "strings; mismatches are reported, never corrected."
        ),
        "num_checks": len(checks),
        "num_mismatches": len(mismatches),
        "all_match": not mismatches,
        "checks": checks,
    }
