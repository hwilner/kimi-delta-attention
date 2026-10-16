"""Run the Phase 4 calibration experiment (issue #28) and print the report.

Tests whether internal uncertainty signals (evidence/precision magnitude,
prediction-error norm, key-collision score, gate entropy, state effective
rank) predict actual recall failure on the #35 key-collision task, comparing
KDA, KDA-RLS-lite, and precision-gated KDA. All metrics and signals are
pre-registered in :class:`CalibrationConfig` before any results are
collected; bootstrap confidence intervals are cluster resamples over
sequences; flat/negative results are printed alongside informative ones
(acceptance criteria).

Usage:
    python examples/calibration_experiment.py
"""

from __future__ import annotations

import json

from kda.experiments import CalibrationConfig, run_calibration_experiment


def main() -> None:
    config = CalibrationConfig()
    print("Pre-registered config (fixed before collecting results):")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    report = run_calibration_experiment(config)

    print("\n=== AUROC of each uncertainty signal (failure prediction) ===")
    header = f"{'signal':24s}" + "".join(f"{name:>22s}" for name in config.variants)
    print(header)
    for signal in config.signals:
        row = f"{signal:24s}"
        for name in config.variants:
            entry = report["variants"][name]["signals"][signal]
            if entry["available"]:
                low, high = entry["auroc"]["ci"]
                row += f"{entry['auroc']['value']:>8.3f} [{low:5.3f},{high:5.3f}]"
            else:
                row += f"{'unavailable':>22s}"
        print(row)

    print("\n=== Per-variant summary ===")
    for name in config.variants:
        section = report["variants"][name]
        rate = section["failure_rate"]
        print(
            f"{name:20s} failure rate {rate['value']:.3f} "
            f"[{rate['ci'][0]:.3f}, {rate['ci'][1]:.3f}] "
            f"({section['num_failures']}/{section['num_items']} items), "
            f"mean recall cosine {section['mean_recall_cosine']:.3f}"
        )

    print("\n=== Full JSON report (includes flat/negative results) ===")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
