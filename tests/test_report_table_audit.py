"""Synthetic tests for the #34 report-table audit.

The audit (:mod:`kda.experiments.report_table_audit`) covers the report
tables rendered directly from the #34 capacity/interference and
overwrite/recency configs and not duplicated in the extension sweeps:
interference, fresh recall, stale dominance, relative update norm, and
key-axis alignment.
"""

from __future__ import annotations

import json
from pathlib import Path

from kda.experiments import (
    CapacityConfig,
    OverwriteConfig,
    run_capacity_experiment,
    run_overwrite_experiment,
)
from kda.experiments.report_table_audit import (
    FRESH_RECALL_HEADING,
    KEY_AXIS_ALIGNMENT_HEADING,
    RELATIVE_UPDATE_NORM_HEADING,
    STALE_DOMINANCE_HEADING,
    audit_report_tables,
    interference_expectations,
    interference_heading,
    overwrite_expectations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_IN_REPORT = REPO_ROOT / "docs" / "PHASE4_RESULTS.md"


CAPACITY_REPORT = {
    "interference_sweep": {
        "kda": {
            "num_pairs": 2,
            "points": [
                {
                    "collision_cosine": 0.5,
                    "recall_accuracy": {"value": 0.75, "ci": [0.5, 1.0]},
                }
            ],
        }
    }
}

OVERWRITE_REPORT = {
    "variants": {
        "kda": {
            "points": [
                {
                    "num_updates": 2,
                    "fresh_recall_accuracy": {"value": 0.875, "ci": [0.5, 1.0]},
                    "stale_dominance_rate": {"value": 0.125, "ci": [0.0, 0.5]},
                    "edit_relative_update_norm": 0.4567,
                    "edit_key_axis_alignment": 0.98765,
                }
            ]
        }
    }
}


def _synthetic_markdown(
    *,
    interference: str = "0.750 [0.500, 1.000]",
    fresh: str = "0.875 [0.500, 1.000]",
    stale: str = "0.125 [0.000, 0.500]",
    relative: str = "0.457",
    alignment: str = "0.988",
) -> str:
    return "\n".join(
        [
            interference_heading(CAPACITY_REPORT),
            "",
            "| variant | cos=0.500 |",
            "|---|---|",
            f"| kda | {interference} |",
            "",
            FRESH_RECALL_HEADING,
            "",
            "| variant | updates=2 |",
            "|---|---|",
            f"| kda | {fresh} |",
            "",
            STALE_DOMINANCE_HEADING,
            "",
            "| variant | updates=2 |",
            "|---|---|",
            f"| kda | {stale} |",
            "",
            RELATIVE_UPDATE_NORM_HEADING,
            "",
            "| variant | updates=2 |",
            "|---|---|",
            f"| kda | {relative} |",
            "",
            KEY_AXIS_ALIGNMENT_HEADING,
            "",
            "| variant | updates=2 |",
            "|---|---|",
            f"| kda | {alignment} |",
        ]
    )


def test_interference_expectations_hand_computed():
    assert interference_heading(CAPACITY_REPORT) == (
        "### Interference sweep (fixed load 2 pairs)"
    )
    assert interference_expectations(CAPACITY_REPORT) == {
        ("kda", "cos=0.500"): "0.750 [0.500, 1.000]"
    }


def test_overwrite_expectations_hand_computed():
    assert overwrite_expectations(OVERWRITE_REPORT) == {
        "fresh": {("kda", "updates=2"): "0.875 [0.500, 1.000]"},
        "stale": {("kda", "updates=2"): "0.125 [0.000, 0.500]"},
        "relative": {("kda", "updates=2"): "0.457"},
        "alignment": {("kda", "updates=2"): "0.988"},
    }


def test_audit_accepts_synthetic_matching_report():
    result = audit_report_tables(
        _synthetic_markdown(),
        capacity_report=CAPACITY_REPORT,
        overwrite_report=OVERWRITE_REPORT,
    )
    assert result["all_match"]
    assert result["num_checks"] == 5
    assert result["num_mismatches"] == 0
    assert {check["table"] for check in result["checks"]} == {
        "### Interference sweep (fixed load 2 pairs)",
        FRESH_RECALL_HEADING,
        STALE_DOMINANCE_HEADING,
        RELATIVE_UPDATE_NORM_HEADING,
        KEY_AXIS_ALIGNMENT_HEADING,
    }


def test_audit_flags_exactly_the_corrupted_cell():
    corrupted = _synthetic_markdown(fresh="0.999 [0.998, 1.000]")
    result = audit_report_tables(
        corrupted,
        capacity_report=CAPACITY_REPORT,
        overwrite_report=OVERWRITE_REPORT,
    )
    assert not result["all_match"]
    bad = [check for check in result["checks"] if not check["match"]]
    assert len(bad) == 1
    assert bad[0]["table"] == FRESH_RECALL_HEADING
    assert bad[0]["row"] == "kda"
    assert bad[0]["column"] == "updates=2"
    assert bad[0]["actual"] == "0.999 [0.998, 1.000]"


def test_missing_tables_are_reported_not_raised():
    result = audit_report_tables(
        "# empty report\n",
        capacity_report=CAPACITY_REPORT,
        overwrite_report=OVERWRITE_REPORT,
    )
    assert not result["all_match"]
    assert result["num_mismatches"] == result["num_checks"] == 5
    assert all("not found" in check["note"] for check in result["checks"])


def test_result_schema_is_json_serializable():
    result = audit_report_tables(
        _synthetic_markdown(),
        capacity_report=CAPACITY_REPORT,
        overwrite_report=OVERWRITE_REPORT,
    )
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["num_checks"] == result["num_checks"]
    for check in result["checks"]:
        assert set(check) >= {"table", "row", "column", "expected", "match"}


def test_deterministic_on_tiny_recomputed_configs():
    capacity = CapacityConfig(
        seeds=(0,),
        batch_size=2,
        capacity_loads=(2,),
        interference_cosines=(0.0,),
        bootstrap_resamples=10,
    )
    overwrite = OverwriteConfig(
        seeds=(0,),
        batch_size=2,
        update_counts=(2,),
        bootstrap_resamples=10,
    )
    first = audit_report_tables(
        "# empty report\n",
        capacity_report=run_capacity_experiment(capacity),
        overwrite_report=run_overwrite_experiment(overwrite),
    )
    second = audit_report_tables(
        "# empty report\n",
        capacity_report=run_capacity_experiment(capacity),
        overwrite_report=run_overwrite_experiment(overwrite),
    )
    assert first == second
    # 5 interference cells (5 default variants x 1 cosine) plus 4 tables x
    # 4 overwrite variants x 1 update count.
    assert first["num_checks"] == 5 + 4 * 4


def test_checked_in_report_matches_recomputed_default_tables():
    """The checked-in report agrees with fresh runs of the #34 defaults.

    25 interference cells (5 variants x 5 cosines), 24 overwrite-rate cells
    (4 variants x 3 update counts x 2 rates), and 24 edit-locality cells.
    """
    result = audit_report_tables(CHECKED_IN_REPORT.read_text())
    assert result["num_checks"] == 73
    assert result["checks"], "expected the audit to compare cells"
    assert result["all_match"], [
        check for check in result["checks"] if not check["match"]
    ]
