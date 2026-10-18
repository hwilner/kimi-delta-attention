"""Run the Phase 4 adaptive hybrid-routing experiment (issue #38).

Routes queries between the compressed KDA-family memory and an exact
softmax-attention fallback on the #35 key-collision task, scoring each
query's predicted uncertainty from captured traces. For every
(variant, signal, threshold) operating point the report prints the recall
accuracy against the fallback rate, next to a fixed-ratio baseline that
routes the *same number* of queries chosen at random — the adaptive arm only
wins when the signal identifies *which* queries fail. Variants, seeds,
signals, and the threshold grid are pre-registered in
:class:`RoutingExperimentConfig` before results are collected; flat/negative
signals (e.g. gate entropy, which the #28 calibration experiment found
uninformative at these defaults) are printed alongside informative ones.

Usage:
    python examples/hybrid_routing.py
"""

from __future__ import annotations

import json

from kda.experiments import RoutingExperimentConfig, run_routing_experiment


def main() -> None:
    config = RoutingExperimentConfig()
    print("Pre-registered config (fixed before collecting results):")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    report = run_routing_experiment(config)

    for variant_name, section in report["variants"].items():
        print(f"\n=== variant: {variant_name} ===")
        print(
            f"compressed accuracy {section['compressed_accuracy']:.3f} | "
            f"exact accuracy {section['exact_accuracy']:.3f} | "
            f"headroom {section['exact_headroom']:+.3f}"
        )
        header = (
            f"{'signal':22s} {'rate':>5s} {'recall':>7s} {'fixed':>7s} "
            f"{'gain':>6s} {'over':>6s} {'under':>6s}"
        )
        print(header)
        for signal_name in config.signals:
            entry = section["signals"][signal_name]
            if not entry["available"]:
                print(f"{signal_name:22s} unavailable ({entry['reason']})")
                continue
            for point in entry["operating_points"]:
                print(
                    f"{signal_name:22s} "
                    f"{point['fallback_rate']:5.2f} "
                    f"{point['recall_accuracy']:7.3f} "
                    f"{point['fixed_ratio_accuracy']:7.3f} "
                    f"{point['gain_vs_fixed_ratio']:+6.3f} "
                    f"{point['overuse_rate']:6.3f} "
                    f"{point['underuse_rate']:6.3f}"
                )
    print(
        "\nColumns: rate = fraction routed to exact (cost proxy, no latency "
        "measured); recall = accuracy after routing; fixed = matched-budget "
        "fixed-ratio baseline; gain = recall - fixed; over = wasted exact "
        "calls among routed; under = failures left unrouted."
    )


if __name__ == "__main__":
    main()
