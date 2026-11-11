"""Synthetic tests for the #34 cross-experiment consistency checker.

The checker (:mod:`kda.experiments.report_consistency`) verifies that the
checked-in ``docs/PHASE4_RESULTS.md`` tables agree with the recomputed
#50/#51/#52/#54 extension sweeps: the sweep default cells reproduce the
#28/#34/#36/#38 pipelines bit-for-bit (capacity/calibration/routing
cross-checks), and the report's robustness sections embed the #50/#51/#52/#54
grids in full, cross-checked cell-by-cell against fresh full-grid
runs. Tests cover the markdown parser on synthetic input, the formatting
helpers against hand-computed strings, expectation construction on
hand-built reports, mismatch detection on corrupted reports, and the full
check against the checked-in report on freshly recomputed sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kda.experiments import (
    CalibrationRobustnessConfig,
    PrecisionGatedAblationConfig,
    RLSLiteAblationConfig,
    RoutingRobustnessConfig,
    check_report_consistency,
    default_regime_cell,
    parse_pipe_tables,
    run_calibration_robustness,
    run_precision_gated_ablation,
    run_rls_lite_ablation,
    run_routing_robustness,
)
from kda.experiments.report_consistency import (
    AUROC_HEADING,
    CALIBRATION_REGIME_AUROC_HEADING,
    CALIBRATION_REGIME_FAILURE_HEADER,
    CALIBRATION_REGIME_FAILURE_HEADING,
    CAPACITY_HEADING,
    PRECISION_GATED_ABLATION_HEADER,
    ROUTING_GAINS_HEADING,
    ROUTING_REGIME_ENDPOINT_HEADING,
    ROUTING_REGIME_GAIN_HEADING,
    _compare_rows,
    calibration_regime_auroc_header,
    calibration_regime_auroc_rows,
    calibration_regime_failure_rows,
    capacity_expectations,
    calibration_expectations,
    format_report_ci,
    format_report_float,
    precision_gated_ablation_heading,
    precision_gated_ablation_rows,
    rls_lite_ablation_rows,
    rls_lite_capacity_expectations,
    routing_expectations,
    routing_regime_endpoint_rows,
    routing_regime_gain_rows,
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
    """Full-grid #51 ablation (the default config; the grids are cheap)."""
    return run_precision_gated_ablation(PrecisionGatedAblationConfig())


@pytest.fixture(scope="module")
def calibration_robustness_report() -> dict:
    """Full-grid #50 calibration robustness sweep (the default config)."""
    return run_calibration_robustness(CalibrationRobustnessConfig())


@pytest.fixture(scope="module")
def routing_robustness_report() -> dict:
    """Full-grid #52 routing robustness sweep (the default config)."""
    return run_routing_robustness(RoutingRobustnessConfig())


@pytest.fixture(scope="module")
def rls_lite_ablation_report() -> dict:
    """Full-grid #54 ablation (the default config; the grids are cheap)."""
    return run_rls_lite_ablation(RLSLiteAblationConfig())


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
        CALIBRATION_REGIME_FAILURE_HEADING,
        CALIBRATION_REGIME_AUROC_HEADING,
        ROUTING_REGIME_GAIN_HEADING,
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


def test_rls_lite_capacity_expectations_require_single_default_cell():
    bogus = {
        "operating_points": [
            {
                "num_pairs": 8,
                "cells": [
                    {"is_default": True, "recall_accuracy": {"value": 0.5, "ci": [0.4, 0.6]}},
                    {"is_default": True, "recall_accuracy": {"value": 0.6, "ci": [0.5, 0.7]}},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="exactly one default cell"):
        rls_lite_capacity_expectations(bogus)


def test_rls_lite_capacity_expectations_formatting():
    report = {
        "operating_points": [
            {
                "num_pairs": 8,
                "cells": [
                    {
                        "is_default": False,
                        "recall_accuracy": {"value": 0.1, "ci": [0.0, 0.2]},
                    },
                    {
                        "is_default": True,
                        "recall_accuracy": {"value": 0.4687, "ci": [0.4302, 0.5176]},
                    },
                ],
            }
        ]
    }
    expectations = rls_lite_capacity_expectations(report)
    assert expectations == {("kda-rls-lite", 8): "0.469 [0.430, 0.518]"}


# --- ablation row builders (report robustness section) -----------------------

HAND_BUILT_ABLATION_REPORT = {
    "operating_points": [
        {
            "num_pairs": 8,
            "collision_cosine": 0.9,
            "cells": [
                {
                    "lambda_": 1.0,
                    "forgetting": 0.9,
                    "process_noise": 0.0,
                    "is_default": False,
                    "recall_accuracy": {"value": 0.40084, "ci": [0.3724, 0.4304]},
                    "mean_recall_cosine": 0.46076,
                },
                {
                    "lambda_": 1.0,
                    "forgetting": 1.0,
                    "process_noise": 0.0,
                    "is_default": True,
                    "recall_accuracy": {"value": 0.53374, "ci": [0.51264, 0.55484]},
                    "mean_recall_cosine": 0.53062,
                },
            ],
        }
    ]
}

PRECISION_ABLATION_HEADING_8 = (
    "### Precision-gated KDA ablation (pairs=8, collision cosine 0.900)"
)


def test_ablation_heading_format():
    assert (
        precision_gated_ablation_heading(8, 0.9) == PRECISION_ABLATION_HEADING_8
    )
    assert (
        precision_gated_ablation_heading(16, 0.9)
        == "### Precision-gated KDA ablation (pairs=16, collision cosine 0.900)"
    )


def test_precision_gated_ablation_rows_hand_computed():
    rows = precision_gated_ablation_rows(HAND_BUILT_ABLATION_REPORT)
    assert rows == {
        PRECISION_ABLATION_HEADING_8: [
            ["1.000", "0.900", "0.000", "", "0.401 [0.372, 0.430]", "0.461"],
            ["1.000", "1.000", "0.000", "*", "0.534 [0.513, 0.555]", "0.531"],
        ]
    }


def test_rls_lite_ablation_rows_hand_computed():
    report = {
        "operating_points": [
            {
                "num_pairs": 16,
                "collision_cosine": 0.9,
                "cells": [
                    {
                        "evidence_decay": 1.0,
                        "beta_min": 0.0,
                        "beta_max": 1.0,
                        "is_default": True,
                        "recall_accuracy": {
                            "value": 0.24587,
                            "ci": [0.22794, 0.26344],
                        },
                        "mean_recall_cosine": 0.27304,
                    },
                ],
            }
        ]
    }
    rows = rls_lite_ablation_rows(report)
    assert rows == {
        "### KDA-RLS-lite ablation (pairs=16, collision cosine 0.900)": [
            ["1.000", "0.000", "1.000", "*", "0.246 [0.228, 0.263]", "0.273"],
        ]
    }


def test_compare_rows_flags_row_count_and_missing_row():
    """A truncated report table must fail honestly: row-count mismatch plus a
    per-row missing note, while the surviving row's cells still compare."""
    expected = precision_gated_ablation_rows(HAND_BUILT_ABLATION_REPORT)[
        PRECISION_ABLATION_HEADING_8
    ]
    markdown = "\n".join(
        [
            PRECISION_ABLATION_HEADING_8,
            "",
            "| lambda | forgetting | process_noise | default | recall accuracy | mean recall cosine |",
            "|---|---|---|---|---|---|",
            "| 1.000 | 0.900 | 0.000 |  | 0.401 [0.372, 0.430] | 0.461 |",
        ]
    )
    table = parse_pipe_tables(markdown)[PRECISION_ABLATION_HEADING_8][0]
    checks: list[dict] = []
    _compare_rows(
        checks,
        table_name=PRECISION_ABLATION_HEADING_8,
        table=table,
        expected_header=PRECISION_GATED_ABLATION_HEADER,
        expected_rows=expected,
    )
    # Header matches; the surviving row 0 matches cell-by-cell.
    header_checks = [c for c in checks if c["row"] == "(header)"]
    assert len(header_checks) == 1 and header_checks[0]["match"]
    row0 = [c for c in checks if c["row"] == "row 0"]
    assert len(row0) == len(PRECISION_GATED_ABLATION_HEADER)
    assert all(c["match"] for c in row0)
    # The missing second row is flagged twice: a row-count mismatch and a
    # per-row missing note — reported, never raised or hidden.
    count_checks = [c for c in checks if c["row"] == "(row count)"]
    assert len(count_checks) == 1 and not count_checks[0]["match"]
    assert "expected 2 data rows, found 1" in count_checks[0]["note"]
    row1 = [c for c in checks if c["row"] == "row 1"]
    assert len(row1) == 1 and not row1[0]["match"]
    assert row1[0]["actual"] is None
    assert "row missing" in row1[0]["note"]


def test_calibration_expectations_require_single_cell():
    with pytest.raises(ValueError, match="single-cell"):
        calibration_expectations({"cells": [{"variants": {}}, {"variants": {}}]})


# --- regime row builders (#50/#52 robustness section) ------------------------

HAND_BUILT_CAL_ROBUSTNESS_REPORT = {
    "config": {
        "variants": ["kda", "precision-gated-kda"],
        "signals": ["precision_along_key", "gate_entropy"],
    },
    "cells": [
        {
            "num_pairs": 8,
            "collision_cosine": 0.9,
            "variants": {
                "kda": {
                    "failure_rate": {"value": 0.58854, "ci": [0.54427, 0.61979]},
                    "mean_recall_cosine": 0.46797,
                    "signals": {
                        "precision_along_key": {"available": False},
                        "gate_entropy": {
                            "available": True,
                            "auroc": {"value": 0.45864, "ci": [0.39708, 0.52031]},
                        },
                    },
                },
                "precision-gated-kda": {
                    "failure_rate": {"value": 0.46615, "ci": [0.44531, 0.48698]},
                    "mean_recall_cosine": 0.53062,
                    "signals": {
                        "precision_along_key": {
                            "available": True,
                            "auroc": {"value": 0.87153, "ci": [0.84323, 0.89740]},
                        },
                        "gate_entropy": {
                            "available": True,
                            "auroc": {"value": 0.48646, "ci": [0.43490, 0.54766]},
                        },
                    },
                },
            },
        },
        {
            "num_pairs": 16,
            "collision_cosine": 0.5,
            "variants": {
                "kda": {
                    "failure_rate": {"value": 0.69531, "ci": [0.67760, 0.71354]},
                    "mean_recall_cosine": 0.29974,
                    "signals": {
                        "precision_along_key": {"available": False},
                        "gate_entropy": {
                            "available": True,
                            "auroc": {"value": 0.49896, "ci": [0.42969, 0.56510]},
                        },
                    },
                },
                "precision-gated-kda": {
                    "failure_rate": {"value": 0.65104, "ci": [0.63802, 0.66667]},
                    "mean_recall_cosine": 0.33420,
                    "signals": {
                        "precision_along_key": {
                            "available": True,
                            "auroc": {"value": 0.97604, "ci": [0.96354, 0.98698]},
                        },
                        "gate_entropy": {
                            "available": True,
                            "auroc": {"value": 0.49479, "ci": [0.42969, 0.56510]},
                        },
                    },
                },
            },
        },
    ],
}


def test_default_regime_cell_extraction():
    cell = default_regime_cell(HAND_BUILT_CAL_ROBUSTNESS_REPORT)
    assert cell["num_pairs"] == 8 and cell["collision_cosine"] == 0.9


def test_default_regime_cell_requires_exactly_one_default():
    with pytest.raises(ValueError, match="exactly one default regime cell"):
        default_regime_cell({"cells": [{"num_pairs": 4, "collision_cosine": 0.5}]})
    with pytest.raises(ValueError, match="exactly one default regime cell"):
        default_regime_cell(
            {
                "cells": [
                    {"num_pairs": 8, "collision_cosine": 0.9},
                    {"num_pairs": 8, "collision_cosine": 0.9},
                ]
            }
        )


def test_calibration_regime_failure_rows_hand_computed():
    rows = calibration_regime_failure_rows(HAND_BUILT_CAL_ROBUSTNESS_REPORT)
    assert rows == {
        CALIBRATION_REGIME_FAILURE_HEADING: [
            ["8", "0.900", "kda", "*", "0.589 [0.544, 0.620]", "0.468"],
            ["8", "0.900", "precision-gated-kda", "*", "0.466 [0.445, 0.487]", "0.531"],
            ["16", "0.500", "kda", "", "0.695 [0.678, 0.714]", "0.300"],
            ["16", "0.500", "precision-gated-kda", "", "0.651 [0.638, 0.667]", "0.334"],
        ]
    }


def test_calibration_regime_auroc_rows_hand_computed():
    report = HAND_BUILT_CAL_ROBUSTNESS_REPORT
    assert calibration_regime_auroc_header(report) == [
        "num_pairs", "collision_cosine", "signal", "default",
        "kda", "precision-gated-kda",
    ]
    rows = calibration_regime_auroc_rows(report)
    assert rows == {
        CALIBRATION_REGIME_AUROC_HEADING: [
            ["8", "0.900", "precision_along_key", "*", "unavailable", "0.872 [0.843, 0.897]"],
            ["8", "0.900", "gate_entropy", "*", "0.459 [0.397, 0.520]", "0.486 [0.435, 0.548]"],
            ["16", "0.500", "precision_along_key", "", "unavailable", "0.976 [0.964, 0.987]"],
            ["16", "0.500", "gate_entropy", "", "0.499 [0.430, 0.565]", "0.495 [0.430, 0.565]"],
        ]
    }


HAND_BUILT_ROUTING_ROBUSTNESS_REPORT = {
    "config": {
        "variants": ["kda", "kda-rls-lite"],
        "signals": ["evidence_along_key", "gate_entropy"],
    },
    "cells": [
        {
            "num_pairs": 8,
            "collision_cosine": 0.9,
            "variants": {
                "kda": {
                    "compressed_accuracy": 0.41146,
                    "exact_accuracy": 1.0,
                    "exact_headroom": 0.58854,
                    "signals": {
                        "evidence_along_key": {"available": False},
                        "gate_entropy": {
                            "available": True,
                            "operating_points": [
                                {"gain_vs_fixed_ratio": 0.00521, "fallback_rate": 0.01042},
                                {"gain_vs_fixed_ratio": -0.01563, "fallback_rate": 0.5},
                            ],
                        },
                    },
                },
                "kda-rls-lite": {
                    "compressed_accuracy": 0.46875,
                    "exact_accuracy": 1.0,
                    "exact_headroom": 0.53125,
                    "signals": {
                        "evidence_along_key": {
                            "available": True,
                            "operating_points": [
                                {"gain_vs_fixed_ratio": 0.01042, "fallback_rate": 0.01042},
                                {"gain_vs_fixed_ratio": 0.09375, "fallback_rate": 0.29948},
                            ],
                        },
                        "gate_entropy": {
                            "available": True,
                            "operating_points": [
                                {"gain_vs_fixed_ratio": 0.00521, "fallback_rate": 0.01042},
                                {"gain_vs_fixed_ratio": -0.01563, "fallback_rate": 0.5},
                            ],
                        },
                    },
                },
            },
        },
    ],
}


def test_routing_regime_endpoint_rows_hand_computed():
    rows = routing_regime_endpoint_rows(HAND_BUILT_ROUTING_ROBUSTNESS_REPORT)
    assert rows == {
        ROUTING_REGIME_ENDPOINT_HEADING: [
            ["8", "0.900", "kda", "*", "0.411", "1.000", "0.589"],
            ["8", "0.900", "kda-rls-lite", "*", "0.469", "1.000", "0.531"],
        ]
    }


def test_routing_regime_gain_rows_hand_computed():
    rows = routing_regime_gain_rows(HAND_BUILT_ROUTING_ROBUSTNESS_REPORT)
    assert rows == {
        ROUTING_REGIME_GAIN_HEADING: [
            ["8", "0.900", "evidence_along_key", "*", "unavailable", "0.094 @ 0.299 / 0.010 @ 0.010"],
            ["8", "0.900", "gate_entropy", "*", "0.005 @ 0.010 / -0.016 @ 0.500", "0.005 @ 0.010 / -0.016 @ 0.500"],
        ]
    }


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
    routing_robustness_report, rls_lite_ablation_report,
):
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert result["num_checks"] > 0
    mismatches = [c for c in result["checks"] if not c["match"]]
    assert mismatches == [], f"report/sweep mismatches: {mismatches}"
    assert result["all_match"]


def test_checked_in_report_ablation_tables_cross_checked(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """The report's #51/#54 ablation tables are cross-checked cell-by-cell
    against fresh full-grid recomputation: 2 ablations x 2 operating points
    x (1 header + 12 rows x 6 columns) = 292 checks, all matching."""
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    ablation_checks = [
        c for c in result["checks"] if "ablation" in c["table"]
    ]
    assert len(ablation_checks) == 4 * (1 + 12 * 6)
    assert all(c["match"] for c in ablation_checks)
    # Every grid cell is covered, not just the default cell.
    compared_rows = {
        (c["table"], c["row"]) for c in ablation_checks if c["row"] != "(header)"
    }
    assert len(compared_rows) == 4 * 12


def test_checked_in_report_regime_tables_cross_checked(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """The report's #50/#52 regime tables are cross-checked cell-by-cell
    against fresh full-grid recomputation: the failure-rate table (1 header +
    9 cells x 3 variants x 6 columns), the AUROC table (1 + 9 x 6 signals x
    7), the endpoints table (1 + 27 x 7), and the gain table (1 + 9 x 5
    signals x 7) — 1048 checks, all matching."""
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    regime_checks = [
        c for c in result["checks"] if "regime" in c["table"]
    ]
    assert len(regime_checks) == (1 + 27 * 6) + (1 + 54 * 7) + (1 + 27 * 7) + (1 + 45 * 7)
    assert all(c["match"] for c in regime_checks)
    # Every grid cell is covered, not just the default cell.
    compared_rows = {
        (c["table"], c["row"]) for c in regime_checks if c["row"] != "(header)"
    }
    assert len(compared_rows) == 27 + 54 + 27 + 45
    # The default regime rows are among them and carry the marker column.
    default_checks = [
        c for c in regime_checks
        if c["column"] == "default" and c["expected"] == "*"
    ]
    assert len(default_checks) == 3 + 6 + 3 + 5  # rows at the (8, 0.9) cell
    assert all(c["match"] for c in default_checks)


def test_checked_in_report_rls_lite_rows_cross_checked(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """The #54 default cell must contribute the report's kda-rls-lite
    capacity rows at both pre-registered operating points."""
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    rls_checks = [
        c for c in result["checks"]
        if c["table"] == CAPACITY_HEADING and c["row"] == "kda-rls-lite"
    ]
    assert {c["column"] for c in rls_checks} == {"pairs=8", "pairs=16"}
    assert all(c["match"] for c in rls_checks)


def test_result_schema_is_json_serializable(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["num_checks"] == result["num_checks"]
    for check in result["checks"]:
        assert set(check) >= {"table", "row", "column", "expected", "match"}


def test_mismatch_detection_on_corrupted_report(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    corrupted = report_markdown.replace("0.534 [0.513, 0.555]", "0.999 [0.998, 1.000]")
    assert corrupted != report_markdown
    result = check_report_consistency(
        corrupted,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert not result["all_match"]
    bad = [c for c in result["checks"] if not c["match"]]
    # The corrupted string appears in the capacity sweep, the interference
    # sweep (not cross-checked), AND the #51 ablation table's default row at
    # pairs=8 — so exactly two checks fail: the capacity cell and the
    # ablation row.
    assert len(bad) == 2
    assert bad[0]["table"] == CAPACITY_HEADING
    assert bad[0]["row"] == "precision-gated-kda"
    assert bad[0]["column"] == "pairs=8"
    assert bad[0]["actual"] == "0.999 [0.998, 1.000]"
    assert bad[1]["table"] == PRECISION_ABLATION_HEADING_8
    assert bad[1]["column"] == "recall accuracy"
    assert bad[1]["actual"] == "0.999 [0.998, 1.000]"


def test_mismatch_detection_on_corrupted_rls_lite_cell(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """Corrupting the report's RLS-lite pairs=8 value flags every
    cross-checked rendering: the capacity cell and the two ablation-table
    rows whose formatted value coincides with it (the default row and the
    beta_min=0.100 row, which is recall-identical to the default)."""
    corrupted = report_markdown.replace("0.469 [0.430, 0.518]", "0.111 [0.100, 0.122]")
    assert corrupted != report_markdown
    result = check_report_consistency(
        corrupted,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert not result["all_match"]
    bad = [c for c in result["checks"] if not c["match"]]
    assert len(bad) == 3
    assert bad[0]["table"] == CAPACITY_HEADING
    assert bad[0]["row"] == "kda-rls-lite"
    assert bad[0]["column"] == "pairs=8"
    assert bad[0]["actual"] == "0.111 [0.100, 0.122]"
    rls_bad = [c for c in bad if c["table"].startswith("### KDA-RLS-lite ablation")]
    assert len(rls_bad) == 2
    assert all(c["column"] == "recall accuracy" for c in rls_bad)
    assert all(c["actual"] == "0.111 [0.100, 0.122]" for c in rls_bad)


def test_mismatch_detection_on_corrupted_ablation_row(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """A hand edit to a non-default ablation cell (a string that appears
    nowhere else in the report) must flag exactly that cell."""
    corrupted = report_markdown.replace("0.604 [0.573, 0.638]", "0.999 [0.998, 1.000]")
    assert corrupted != report_markdown
    result = check_report_consistency(
        corrupted,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert not result["all_match"]
    bad = [c for c in result["checks"] if not c["match"]]
    assert len(bad) == 1
    assert bad[0]["table"] == PRECISION_ABLATION_HEADING_8
    assert bad[0]["column"] == "recall accuracy"
    assert bad[0]["actual"] == "0.999 [0.998, 1.000]"


def test_mismatch_detection_on_corrupted_regime_cell(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """A hand edit to a #50 regime-table cell (a string that appears nowhere
    else in the report) must flag exactly that cell."""
    corrupted = report_markdown.replace("0.130 [0.088, 0.177]", "0.999 [0.998, 1.000]")
    assert corrupted != report_markdown
    result = check_report_consistency(
        corrupted,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert not result["all_match"]
    bad = [c for c in result["checks"] if not c["match"]]
    assert len(bad) == 1
    assert bad[0]["table"] == CALIBRATION_REGIME_FAILURE_HEADING
    assert bad[0]["row"] == "row 0"
    assert bad[0]["column"] == "failure rate"
    assert bad[0]["actual"] == "0.999 [0.998, 1.000]"


def test_missing_table_is_reported_not_raised(
    ablation_report, calibration_robustness_report, routing_robustness_report,
    rls_lite_ablation_report,
):
    result = check_report_consistency(
        "# empty report\n",
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    assert not result["all_match"]
    assert result["num_mismatches"] == result["num_checks"]
    assert all("not found" in c["note"] for c in result["checks"])


def test_unavailable_signals_cross_checked(
    report_markdown, ablation_report, calibration_robustness_report,
    routing_robustness_report, rls_lite_ablation_report,
):
    """Baseline KDA exposes no precision/evidence signals: the report's
    literal 'unavailable' cells must match the sweeps' unavailable records."""
    result = check_report_consistency(
        report_markdown,
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_robustness_report,
        routing_robustness_report=routing_robustness_report,
        rls_lite_ablation_report=rls_lite_ablation_report,
    )
    unavailable = [
        c for c in result["checks"] if c["expected"] == "unavailable"
    ]
    assert unavailable, "expected some unavailable-signal checks"
    assert all(c["match"] and c["actual"] == "unavailable" for c in unavailable)


def test_deterministic_across_runs(report_markdown):
    first = check_report_consistency(
        report_markdown,
        ablation_report=run_precision_gated_ablation(
            PrecisionGatedAblationConfig()
        ),
        calibration_robustness_report=run_calibration_robustness(
            CalibrationRobustnessConfig()
        ),
        routing_robustness_report=run_routing_robustness(
            RoutingRobustnessConfig()
        ),
        rls_lite_ablation_report=run_rls_lite_ablation(RLSLiteAblationConfig()),
    )
    second = check_report_consistency(
        report_markdown,
        ablation_report=run_precision_gated_ablation(
            PrecisionGatedAblationConfig()
        ),
        calibration_robustness_report=run_calibration_robustness(
            CalibrationRobustnessConfig()
        ),
        routing_robustness_report=run_routing_robustness(
            RoutingRobustnessConfig()
        ),
        rls_lite_ablation_report=run_rls_lite_ablation(RLSLiteAblationConfig()),
    )
    assert first == second


def test_positive_control_tiny_sweeps_match_report(tmp_path):
    """A tiny synthetic end-to-end control: restricted sweeps produce
    well-formed expectations and the checker flags no structural problems
    when the numbers are recomputed into a fresh mini-report."""

    ablation = run_precision_gated_ablation(
        PrecisionGatedAblationConfig(
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

    rls_lite = run_rls_lite_ablation(
        RLSLiteAblationConfig(
            evidence_decay_grid=(1.0,),
            beta_min_grid=(0.0,),
            beta_max_grid=(1.0,),
            operating_points=((4, 0.9),),
            seeds=TINY_SEEDS,
            batch_size=TINY_BATCH,
            bootstrap_resamples=10,
        )
    )
    rls_lite_capacity = rls_lite_capacity_expectations(rls_lite)
    assert ("kda-rls-lite", 4) in rls_lite_capacity

    # Build a mini-report from the recomputed strings: the checker must
    # accept it, proving the expectations are in the report's own format.
    mini = [
        CAPACITY_HEADING,
        "",
        "| variant | pairs=4 |",
        "|---|---|",
        f"| kda | {capacity[('kda', 4)]} |",
        f"| precision-gated-kda | {capacity[('precision-gated-kda', 4)]} |",
        f"| kda-rls-lite | {rls_lite_capacity[('kda-rls-lite', 4)]} |",
        "",
        "### Failure rates",
        "",
        "| variant | failure rate | mean recall cosine |",
        "|---|---|---|",
        "",
        AUROC_HEADING,
        "",
        "| signal | kda |",
        "|---|---|",
        "",
        "### Endpoints",
        "",
        "| variant | compressed accuracy | exact accuracy | headroom |",
        "|---|---|---|",
        "",
        ROUTING_GAINS_HEADING,
        "",
        "| signal | kda |",
        "|---|---|",
    ]
    calibration = run_calibration_robustness(
        CalibrationRobustnessConfig(
            num_pairs_grid=(8,),
            collision_cosine_grid=(0.9,),
            seeds=TINY_SEEDS,
            batch_size=TINY_BATCH,
            bootstrap_resamples=10,
        )
    )
    routing = run_routing_robustness(
        RoutingRobustnessConfig(
            num_pairs_grid=(8,),
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
        rls_lite_ablation_report=rls_lite,
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
