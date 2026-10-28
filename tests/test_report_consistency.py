"""Synthetic tests for the #34 cross-experiment consistency checker.

The checker (:mod:`kda.experiments.report_consistency`) verifies that the
checked-in ``docs/PHASE4_RESULTS.md`` tables agree with the default cells of
the #50/#51/#52 extension sweeps (which reproduce the #28/#34/#38 pipelines
bit-for-bit). Tests cover the markdown parser on synthetic input, the
formatting helpers against hand-computed strings, expectation construction
on hand-built reports, mismatch detection on corrupted reports, and the full
check against the checked-in report on freshly recomputed default cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kda.experiments import (
    check_report_consistency,
    default_ablation_config,
    default_calibration_robustness_config,
    default_routing_robustness_config,
    parse_pipe_tables,
    run_calibration_robustness,
    run_precision_gated_ablation,
    run_routing_robustness,
)
from kda.experiments.report_consistency import (
    AUROC_HEADING,
    CAPACITY_HEADING,
    ROUTING_GAINS_HEADING,
    capacity_expectations,
    calibration_expectations,
    format_report_ci,
    format_report_float,
    routing_expectations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_IN_REPORT = REPO_ROOT / "docs" / "PHASE4_RESULTS.md"

# Restricted sweep sizes for the synthetic (non-checked-in-report) tests, so
# construction/validation tests stay fast; the checked-in-report tests use
# the default seeds/batch sizes to compare against the real numbers.
TINY_SEEDS = (0,)
TINY_BATCH = 2


@pytest.fixture(scope="module")
def ablation_report() -> dict:
    return run_precision_gated_ablation(default_ablation_config())


@pytest.fixture(scope="module")
def calibration_robustness_report() -> dict:
    return run_calibration_robustness(default_calibration_robustness_config())


@pytest.fixture(scope="module")
def routing_robustness_report() -> dict:
    return run_routing_robustness(default_routing_robustness_config())


@pytest.fixture(scope="module")
def report_markdown() -> str:
    return CHECKED_IN_REPORT.read_text()


# --- parser -----------------------------------------------------------------

SYNTHETIC_MARKDOWN = """# Title

Some prose with | a pipe | that is not a table.

## Section one

### Table heading

| variant | pairs=8 | pairs=16 |
|---|---|---|
| kda | 0.411 [0.380, 0.456] | 0.220 [0.202, 0.238] |
| precision-gated-kda | 0.534 [0.513, 0.555] | 0.271 [0.255, 0.286] |

```json
{"config": "| not | a | table |"}
```

### Second table

| signal | kda |
|---|---|
| gate_entropy | unavailable |
"""


def test_parser_extracts_tables_under_headings():
    tables = parse_pipe_tables(SYNTHETIC_MARKDOWN)
    assert set(tables) == {"### Table heading", "### Second table"}
    table = tables["### Table heading"][0]
    assert table["header"] == ["variant", "pairs=8", "pairs=16"]
    assert table["rows"] == [
        ["kda", "0.411 [0.380, 0.456]", "0.220 [0.202, 0.238]"],
        ["precision-gated-kda", "0.534 [0.513, 0.555]", "0.271 [0.255, 0.286]"],
    ]


def test_parser_ignores_fenced_code_and_non_table_pipes():
    tables = parse_pipe_tables(SYNTHETIC_MARKDOWN)
    # The fenced JSON block with pipe characters produced no table.
    assert len(tables) == 2
    assert all(len(block) == 1 for block in tables.values())


def test_parser_on_checked_in_report_finds_all_cross_checked_tables():
    tables = parse_pipe_tables(CHECKED_IN_REPORT.read_text())
    for heading in (
        CAPACITY_HEADING,
        "### Failure rates",
        AUROC_HEADING,
        "### Endpoints",
        ROUTING_GAINS_HEADING,
    ):
        assert heading in tables, f"missing heading {heading!r}"
        assert tables[heading], f"no table under {heading!r}"


# --- formatting helpers -----------------------------------------------------


def test_format_report_float_fixed_three_decimals():
    assert format_report_float(0.5344) == "0.534"
    assert format_report_float(1.0) == "1.000"
    assert format_report_float(-0.0465) == "-0.046" or format_report_float(-0.0465) == "-0.047"


def test_format_report_ci_layout():
    entry = {"value": 0.5337, "ci": [0.5126, 0.5549]}
    assert format_report_ci(entry) == "0.534 [0.513, 0.555]"


# --- expectation construction on hand-built reports --------------------------


def test_capacity_expectations_require_single_default_cell():
    bogus = {
        "operating_points": [
            {
                "num_pairs": 8,
                "reference": {},
                "cells": [
                    {"is_default": True, "recall_accuracy": {"value": 0.5, "ci": [0.4, 0.6]}},
                    {"is_default": True, "recall_accuracy": {"value": 0.6, "ci": [0.5, 0.7]}},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="exactly one default cell"):
        capacity_expectations(bogus)


def test_calibration_expectations_require_single_cell():
    with pytest.raises(ValueError, match="single-cell"):
        calibration_expectations({"cells": [{"variants": {}}, {"variants": {}}]})


def test_routing_expectations_unavailable_and_best_worst():
    report = {
        "cells": [
            {
                "num_pairs": 8,
                "collision_cosine": 0.9,
                "variants": {
                    "kda": {
                        "compressed_accuracy": 0.4,
                        "exact_accuracy": 1.0,
                        "exact_headroom": 0.6,
                        "signals": {
                            "precision_along_key": {"available": False},
                            "gate_entropy": {
                                "available": True,
                                "operating_points": [
                                    {"gain_vs_fixed_ratio": -0.02, "fallback_rate": 0.5},
                                    {"gain_vs_fixed_ratio": 0.01, "fallback_rate": 0.1},
                                ],
                            },
                        },
                    }
                },
            }
        ]
    }
    expectations = routing_expectations(report)
    assert expectations["endpoints"][("kda", "compressed accuracy")] == "0.400"
    assert expectations["endpoints"][("kda", "headroom")] == "0.600"
    assert expectations["gains"][("precision_along_key", "kda")] == "unavailable"
    assert (
        expectations["gains"][("gate_entropy", "kda")]
        == "0.010 @ 0.100 / -0.020 @ 0.500"
    )


# --- full check against the checked-in report --------------------------------


def test_checked_in_report_matches_extension_default_cells(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report,
):
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
    )
    assert result["num_checks"] > 0
    mismatches = [c for c in result["checks"] if not c["match"]]
    assert mismatches == [], f"report/sweep mismatches: {mismatches}"
    assert result["all_match"]


def test_result_schema_is_json_serializable(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report,
):
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
    )
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["num_checks"] == result["num_checks"]
    for check in result["checks"]:
        assert set(check) >= {"table", "row", "column", "expected", "match"}


def test_mismatch_detection_on_corrupted_report(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report,
):
    corrupted = report_markdown.replace("0.534 [0.513, 0.555]", "0.999 [0.998, 1.000]")
    assert corrupted != report_markdown
    result = check_report_consistency(
        corrupted,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
    )
    assert not result["all_match"]
    bad = [c for c in result["checks"] if not c["match"]]
    # The corrupted cell appears in the capacity sweep AND the interference
    # sweep; only the capacity table is cross-checked, so exactly one check
    # fails, on the precision-gated pairs=8 cell.
    assert len(bad) == 1
    assert bad[0]["table"] == CAPACITY_HEADING
    assert bad[0]["row"] == "precision-gated-kda"
    assert bad[0]["column"] == "pairs=8"
    assert bad[0]["actual"] == "0.999 [0.998, 1.000]"


def test_missing_table_is_reported_not_raised(
    ablation_report, calibration_robustness_report, routing_robustness_report,
):
    result = check_report_consistency(
        "# empty report\n",
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
    )
    assert not result["all_match"]
    assert result["num_mismatches"] == result["num_checks"]
    assert all("not found" in c["note"] for c in result["checks"])


def test_unavailable_signals_cross_checked(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report,
):
    """Baseline KDA exposes no precision/evidence signals: the report's
    literal 'unavailable' cells must match the sweeps' unavailable records."""
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
    )
    unavailable = [
        c for c in result["checks"] if c["expected"] == "unavailable"
    ]
    assert unavailable, "expected some unavailable-signal checks"
    assert all(c["match"] and c["actual"] == "unavailable" for c in unavailable)


def test_deterministic_across_runs(report_markdown):
    first = check_report_consistency(
        report_markdown,
        ablation_report=run_precision_gated_ablation(default_ablation_config()),
        calibration_robustness_report=run_calibration_robustness(
            default_calibration_robustness_config()
        ),
        routing_robustness_report=run_routing_robustness(
            default_routing_robustness_config()
        ),
    )
    second = check_report_consistency(
        report_markdown,
        ablation_report=run_precision_gated_ablation(default_ablation_config()),
        calibration_robustness_report=run_calibration_robustness(
            default_calibration_robustness_config()
        ),
        routing_robustness_report=run_routing_robustness(
            default_routing_robustness_config()
        ),
    )
    assert first == second


def test_positive_control_tiny_sweeps_match_report(tmp_path):
    """A tiny synthetic end-to-end control: restricted sweeps produce
    well-formed expectations and the checker flags no structural problems
    when the numbers are recomputed into a fresh mini-report."""

    ablation = run_precision_gated_ablation(
        default_ablation_config().__class__(
            lambda_grid=(1.0,),
            forgetting_grid=(1.0,),
            process_noise_grid=(0.0,),
            operating_points=((4, 0.9),),
            seeds=TINY_SEEDS,
            batch_size=TINY_BATCH,
            bootstrap_resamples=10,
        )
    )
    capacity = capacity_expectations(ablation)
    assert ("kda", 4) in capacity
    assert ("precision-gated-kda", 4) in capacity

    # Build a mini-report from the recomputed strings: the checker must
    # accept it, proving the expectations are in the report's own format.
    mini = [
        CAPACITY_HEADING,
        "",
        "| variant | pairs=4 |",
        "|---|---|",
        f"| kda | {capacity[('kda', 4)]} |",
        f"| precision-gated-kda | {capacity[('precision-gated-kda', 4)]} |",
        "",
        "### Failure rates",
        "",
        "| variant | failure rate | mean recall cosine |",
        "|---|---|",
        "",
        AUROC_HEADING,
        "",
        "| signal | kda |",
        "|---|---|",
        "",
        "### Endpoints",
        "",
        "| variant | compressed accuracy | exact accuracy | headroom |",
        "|---|---|",
        "",
        ROUTING_GAINS_HEADING,
        "",
        "| signal | kda |",
        "|---|---|",
    ]
    calibration = run_calibration_robustness(
        default_calibration_robustness_config().__class__(
            num_pairs_grid=(4,),
            collision_cosine_grid=(0.9,),
            seeds=TINY_SEEDS,
            batch_size=TINY_BATCH,
            bootstrap_resamples=10,
        )
    )
    routing = run_routing_robustness(
        default_routing_robustness_config().__class__(
            num_pairs_grid=(4,),
            collision_cosine_grid=(0.9,),
            seeds=TINY_SEEDS,
            batch_size=TINY_BATCH,
        )
    )
    result = check_report_consistency(
        "\n".join(mini),
        ablation_report=ablation,
        calibration_robustness_report=calibration,
        routing_robustness_report=routing,
    )
    # Capacity cells match by construction; the empty calibration/routing
    # tables are flagged as missing rows (structural honesty), which this
    # control asserts explicitly rather than hiding.
    capacity_checks = [
        c for c in result["checks"] if c["table"] == CAPACITY_HEADING
    ]
    assert capacity_checks and all(c["match"] for c in capacity_checks)
    other = [c for c in result["checks"] if c["table"] != CAPACITY_HEADING]
    assert other and all(not c["match"] for c in other)
