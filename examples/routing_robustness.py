"""Run the routing robustness sweep (extends issue #38) and print it.

Re-runs the #38 hybrid-routing question — does uncertainty-aware routing to
an exact-attention fallback beat a matched-budget fixed-ratio baseline? —
across a pre-registered grid of task regimes (sequence load x key-collision
cosine), to check whether the headline routing result is regime-independent
or an artifact of the single default operating point. The grid, signals,
threshold quantiles, and baselines are frozen in
:class:`RoutingRobustnessConfig` before any results are collected;
unavailable, flat, and negative cells are printed alongside informative
ones (acceptance criteria).

Usage:
    python examples/routing_robustness.py
"""

from __future__ import annotations

import json

from kda.experiments import (
    RoutingRobustnessConfig,
    run_routing_robustness,
)


def _format_point(entry: dict, index: int) -> str:
    """Adaptive recall and gain over fixed ratio at one operating point."""

    if not entry["available"]:
        return f"{'unavailable':>34s}"
    point = entry["operating_points"][index]
    return (
        f"acc {point['recall_accuracy']:.3f} | "
        f"gain vs fixed {point['gain_vs_fixed_ratio']:+.3f}"
    )


def main() -> None:
    config = RoutingRobustnessConfig()
    print("Pre-registered robustness config (fixed before collecting results):")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    report = run_routing_robustness(config)

    # Focus the tables on the 0.9-quantile operating point (index 3 of the
    # pre-registered threshold grid); the full grid is in the JSON report.
    focus = 3
    print(
        f"\nOperating point shown: threshold at the "
        f"{config.threshold_quantiles[focus]:.2f} quantile of each cell's "
        f"pooled scores (fallback rate is the cost proxy; no latency "
        f"measured)."
    )

    for cell in report["cells"]:
        print(
            f"\n=== Cell: num_pairs={cell['num_pairs']}, "
            f"collision_cosine={cell['collision_cosine']} ==="
        )
        for name in config.variants:
            section = cell["variants"][name]
            print(
                f"  {name:20s} compressed {section['compressed_accuracy']:.3f} | "
                f"exact {section['exact_accuracy']:.3f} | "
                f"headroom {section['exact_headroom']:+.3f}"
            )
        header = f"  {'signal':24s}" + "".join(
            f"{name:>34s}" for name in config.variants
        )
        print(header)
        for signal in config.signals:
            row = f"  {signal:24s}"
            for name in config.variants:
                row += f"{_format_point(cell['variants'][name]['signals'][signal], focus):>34s}"
            print(row)

    print("\n=== Adaptive gain over fixed-ratio baseline across cells ===")
    print(f"(signal x variant, at the {config.threshold_quantiles[focus]:.2f} quantile)")
    header = f"{'signal':24s}" + "".join(f"{name:>24s}" for name in config.variants)
    print(header)
    for signal in config.signals:
        row = f"{signal:24s}"
        for name in config.variants:
            gains = [
                cell["variants"][name]["signals"][signal]["operating_points"][focus][
                    "gain_vs_fixed_ratio"
                ]
                for cell in report["cells"]
                if cell["variants"][name]["signals"][signal]["available"]
            ]
            row += (
                f"{min(gains):>+9.3f} .. {max(gains):>+9.3f}    "
                if gains
                else f"{'unavailable':>24s}"
            )
        print(row)

    print("\n=== Full JSON report (includes flat/negative results) ===")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
