"""Audit the Phase 4 report tables rendered directly from the #34 configs.

This complements ``examples/report_consistency.py``: that script checks the
report against the #50/#51/#52/#54 extension sweeps, while this script
recomputes the checked-in #34 capacity/interference and overwrite/recency
configs and verifies the tables they render directly (interference, fresh
recall, stale dominance, relative update norm, and key-axis alignment).

Usage:
    python examples/report_table_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kda.experiments.report_table_audit import audit_report_tables

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "PHASE4_RESULTS.md"


def main() -> None:
    print("Recomputing the #34 capacity/interference and overwrite/recency configs ...")
    result = audit_report_tables(REPORT_PATH.read_text())

    print(f"\nAuditing tables in {REPORT_PATH.name}:")
    current_table = None
    for check in result["checks"]:
        if check["table"] != current_table:
            current_table = check["table"]
            print(f"\n  {current_table}")
        status = "OK      " if check["match"] else "MISMATCH"
        print(
            f"    [{status}] {check['row']} / {check['column']}: "
            f"report={check['actual']!r} recomputed={check['expected']!r}"
            + (f" ({check['note']})" if "note" in check else "")
        )

    print(
        f"\n{result['num_checks'] - result['num_mismatches']}/"
        f"{result['num_checks']} cells match; all_match={result['all_match']}"
    )
    print("\nFull JSON result:")
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["all_match"] else 1)


if __name__ == "__main__":
    main()
