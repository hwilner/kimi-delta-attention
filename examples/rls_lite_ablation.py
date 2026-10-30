"""Run the #36 KDA-RLS-lite hyperparameter ablation and print it.

The #36 prototype landed with one default hyperparameter setting, and #34
measured the RLS-lite recall curve only at that default. This example
sweeps the pre-registered (evidence_decay, beta_min, beta_max) grid at the
pre-registered operating points to check whether that result is stable or a
lucky default. The grid, operating points, and metrics are frozen in
:class:`RLSLiteAblationConfig` before any results are collected; collapsed
or flat cells are printed alongside strong ones.

Usage:
    python examples/rls_lite_ablation.py
"""

from __future__ import annotations

import json

from kda.experiments import RLSLiteAblationConfig, run_rls_lite_ablation


def _format_rate(entry: dict) -> str:
    value = entry["value"]
    low, high = entry["ci"]
    return f"{value:>6.3f} [{low:5.3f},{high:5.3f}]"


def main() -> None:
    config = RLSLiteAblationConfig()
    print("Pre-registered ablation config (fixed before collecting results):")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    report = run_rls_lite_ablation(config)

    for point in report["operating_points"]:
        print(
            f"\n=== Operating point: num_pairs={point['num_pairs']}, "
            f"collision_cosine={point['collision_cosine']} ==="
        )
        for name, entry in point["reference"].items():
            print(
                f"  reference {name:20s} recall "
                f"{_format_rate(entry['recall_accuracy'])} "
                f"(mean cosine {entry['mean_recall_cosine']:.3f})"
            )
        header = (
            f"  {'rho':>5s} {'beta_min':>8s} {'beta_max':>8s} "
            f"{'recall accuracy':>20s} {'mean cosine':>12s}"
        )
        print(header)
        for cell in point["cells"]:
            marker = " *default*" if cell["is_default"] else ""
            print(
                f"  {cell['evidence_decay']:>5.2f} {cell['beta_min']:>8.2f} "
                f"{cell['beta_max']:>8.2f} "
                f"{_format_rate(cell['recall_accuracy']):>20s} "
                f"{cell['mean_recall_cosine']:>12.3f}{marker}"
            )

    print("\n=== Recall accuracy range across cells (per operating point) ===")
    for point in report["operating_points"]:
        values = [c["recall_accuracy"]["value"] for c in point["cells"]]
        print(
            f"  num_pairs={point['num_pairs']:>2d}, "
            f"cosine={point['collision_cosine']}: "
            f"{min(values):.3f} .. {max(values):.3f} "
            f"(spread {max(values) - min(values):.3f})"
        )

    print("\nFull report:")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
