"""Run the calibration robustness sweep (extends issue #28) and print it.

Re-runs the #28 calibration question — do internal uncertainty signals
predict recall failure? — across a pre-registered grid of task regimes
(sequence load x key-collision cosine), to check whether the headline
result (native precision/evidence signals predict failure; generic proxies
are flat or anti-correlated) is regime-independent or an artifact of the
single default operating point. The grid, signals, and metrics are frozen
in :class:`CalibrationRobustnessConfig` before any results are collected;
unavailable, flat, and negative cells are printed alongside informative
ones (acceptance criteria).

Usage:
    python examples/calibration_robustness.py
"""

from __future__ import annotations

import json
import math

from kda.experiments import (
    CalibrationRobustnessConfig,
    run_calibration_robustness,
)


def _format_auroc(entry: dict) -> str:
    if not entry["available"]:
        return f"{'unavailable':>22s}"
    value = entry["auroc"]["value"]
    low, high = entry["auroc"]["ci"]
    if math.isnan(value):
        return f"{'nan (single class)':>22s}"
    return f"{value:>8.3f} [{low:5.3f},{high:5.3f}]"


def main() -> None:
    config = CalibrationRobustnessConfig()
    print("Pre-registered robustness config (fixed before collecting results):")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    report = run_calibration_robustness(config)

    for cell in report["cells"]:
        print(
            f"\n=== Cell: num_pairs={cell['num_pairs']}, "
            f"collision_cosine={cell['collision_cosine']} ==="
        )
        for name in config.variants:
            section = cell["variants"][name]
            rate = section["failure_rate"]
            print(
                f"  {name:20s} failure rate {rate['value']:.3f} "
                f"[{rate['ci'][0]:.3f}, {rate['ci'][1]:.3f}] "
                f"({section['num_failures']}/{section['num_items']} items)"
            )
        header = f"  {'signal':24s}" + "".join(
            f"{name:>22s}" for name in config.variants
        )
        print(header)
        for signal in config.signals:
            row = f"  {signal:24s}"
            for name in config.variants:
                row += _format_auroc(cell["variants"][name]["signals"][signal])
            print(row)

    print("\n=== AUROC range across cells (per variant, per signal) ===")
    header = f"{'signal':24s}" + "".join(f"{name:>22s}" for name in config.variants)
    print(header)
    for signal in config.signals:
        row = f"{signal:24s}"
        for name in config.variants:
            values = [
                cell["variants"][name]["signals"][signal]["auroc"]["value"]
                for cell in report["cells"]
                if cell["variants"][name]["signals"][signal]["available"]
            ]
            values = [v for v in values if not math.isnan(v)]
            row += (
                f"{min(values):>8.3f} .. {max(values):>10.3f}  "
                if values
                else f"{'unavailable':>22s}"
            )
        print(row)

    print("\n=== Full JSON report (includes flat/negative results) ===")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
