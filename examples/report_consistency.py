"""Cross-check the Phase 4 results report against the extension sweeps.

The checked-in ``docs/PHASE4_RESULTS.md`` (#34) is generated from the
pre-registered #28/#34/#38 default configs. The follow-up sweeps — #50
calibration robustness, #51 precision-gated hyperparameter ablation, #52
routing robustness, and #54 RLS-lite hyperparameter ablation — each contain
a default cell that reproduces the
corresponding base pipeline bit-for-bit, so the report's numbers and the
sweeps' default-cell numbers are two renderings of the same runs and must
agree exactly (as formatted strings). The report also embeds all four
extension grids in full; those tables are cross-checked cell-by-cell against
fresh full-grid runs (the grids are small — seconds to recompute), so a
hand edit to any embedded row is caught, not just the default cell.

This example recomputes the four sweeps on their checked-in full-grid
default configs, parses the checked-in
report, and prints one line per compared cell with ``OK``/``MISMATCH`` —
mismatches are shown, never hidden or corrected. Exit code is 0 only when
every compared cell matches.

Usage:
    python examples/report_consistency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kda.experiments import (
    CalibrationRobustnessConfig,
    PrecisionGatedAblationConfig,
    RLSLiteAblationConfig,
    RoutingRobustnessConfig,
    check_report_consistency,
    run_calibration_robustness,
    run_precision_gated_ablation,
    run_rls_lite_ablation,
    run_routing_robustness,
)

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "PHASE4_RESULTS.md"


def main() -> None:
    print("Recomputing the extension sweeps on their checked-in full-grid")
    print("default configs (#50/#51/#52/#54) ...")
    ablation_report = run_precision_gated_ablation(PrecisionGatedAblationConfig())
    calibration_report = run_calibration_robustness(CalibrationRobustnessConfig())
    routing_report = run_routing_robustness(RoutingRobustnessConfig())
    rls_lite_report = run_rls_lite_ablation(RLSLiteAblationConfig())

    result = check_report_consistency(
        REPORT_PATH.read_text(),
        ablation_report=ablation_report,
        calibration_robustness_report=calibration_report,
        routing_robustness_report=routing_report,
        rls_lite_ablation_report=rls_lite_report,
    )

    print(f"\nCross-checking tables in {REPORT_PATH.name}:")
    current_table = None
    for check in result["checks"]:
        if check["table"] != current_table:
            current_table = check["table"]
            print(f"\n  {current_table}")
        status = "OK      " if check["match"] else "MISMATCH"
        print(
            f"    [{status}] {check['row']} / {check['column']}: "
            f"report={check['actual']!r} sweep={check['expected']!r}"
            + (f" ({check['note']})" if "note" in check else "")
        )

    print(
        f"\n{result['num_checks'] - result['num_mismatches']}/"
        f"{result['num_checks']} cells match; "
        f"all_match={result['all_match']}"
    )
    print("\nFull JSON result:")
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["all_match"] else 1)


if __name__ == "__main__":
    main()
