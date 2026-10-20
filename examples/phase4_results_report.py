"""Generate the Phase 4 results report (issue #34).

Runs the four pre-registered Phase 4 experiment configs — the
capacity/interference and overwrite/recency sweeps of
:mod:`kda.experiments.report_experiments`, plus the #28 calibration and #38
hybrid-routing experiments — and renders ``docs/PHASE4_RESULTS.md`` with all
tables produced from those checked-in configs (acceptance criterion: every
figure/table is generated from a checked-in config or script). The output is
deterministic: seeded experiments, fixed float formatting, no timestamps.

Usage:
    python examples/phase4_results_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from kda.experiments import (
    CalibrationConfig,
    CapacityConfig,
    OverwriteConfig,
    RoutingExperimentConfig,
    run_calibration_experiment,
    run_capacity_experiment,
    run_overwrite_experiment,
    run_routing_experiment,
)

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "PHASE4_RESULTS.md"


def _f(x: float) -> str:
    """Fixed float formatting used for every number in the report."""

    return f"{x:.3f}"


def _fci(entry: dict) -> str:
    """Format a ``{value, ci}`` metric as ``0.750 [0.688, 0.806]``."""

    low, high = entry["ci"]
    return f"{_f(entry['value'])} [{_f(low)}, {_f(high)}]"


def _config_block(title: str, config_dict: dict) -> list[str]:
    return [
        f"<details><summary>{title}</summary>",
        "",
        "```json",
        json.dumps(config_dict, indent=2, sort_keys=True),
        "```",
        "",
        "</details>",
        "",
    ]


def _capacity_section(report: dict) -> list[str]:
    lines = [
        "## Capacity / interference curves",
        "",
        "Recall accuracy (fraction of queries whose readout from the final "
        "state reaches cosine "
        + _f(report["config"]["failure_threshold"])
        + " with the target) on the key-collision task, with cluster-bootstrap"
        " 95% CIs over sequences. GDN-2/EDA run on the shared alpha/beta gate"
        " path, where their tied/zero-erase configurations reduce to KDA"
        " exactly (the paper ties asserted by the #37 tests) — their rows are"
        " regression checks of those reductions, not independent mechanisms.",
        "",
        "### Capacity sweep (collision cosine "
        + _f(report["config"]["capacity_collision_cosine"])
        + ")",
        "",
    ]
    variants = list(report["capacity_sweep"])
    loads = [p["num_pairs"] for p in report["capacity_sweep"][variants[0]]["points"]]
    lines.append("| variant | " + " | ".join(f"pairs={n}" for n in loads) + " |")
    lines.append("|" + "---|" * (len(loads) + 1))
    for name in variants:
        cells = [_fci(p["recall_accuracy"]) for p in report["capacity_sweep"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Interference sweep (fixed load "
        + str(report["interference_sweep"][variants[0]]["num_pairs"])
        + " pairs)",
        "",
    ]
    cosines = [p["collision_cosine"] for p in report["interference_sweep"][variants[0]]["points"]]
    lines.append("| variant | " + " | ".join(f"cos={_f(c)}" for c in cosines) + " |")
    lines.append("|" + "---|" * (len(cosines) + 1))
    for name in variants:
        cells = [_fci(p["recall_accuracy"]) for p in report["interference_sweep"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _overwrite_section(report: dict) -> list[str]:
    variants = list(report["variants"])
    updates = [p["num_updates"] for p in report["variants"][variants[0]]["points"]]
    lines = [
        "## Overwrite / recency behavior",
        "",
        "Same-key overwrite task: each key is re-written with a fresh value "
        "every round. **Fresh recall accuracy** measures the readout of the "
        "*latest* value per key (cosine "
        + _f(report["config"]["failure_threshold"])
        + " criterion, bootstrap 95% CIs); the **stale-dominance rate** is the"
        " fraction of queries whose readout is closer to an overwritten value"
        " than to the latest one. The EDA arm erases at the current key with"
        " constant gate "
        + _f(report["config"]["eda_erase_gate"])
        + " (the paper's `e_t = k_t` collapse — the only erase-address choice"
        " available on the shared task path; a pre-registered baseline"
        " configuration, not a tuned design).",
        "",
        "### Fresh recall accuracy",
        "",
        "| variant | " + " | ".join(f"updates={u}" for u in updates) + " |",
        "|" + "---|" * (len(updates) + 1),
    ]
    for name in variants:
        cells = [_fci(p["fresh_recall_accuracy"]) for p in report["variants"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Stale-dominance rate",
        "",
        "| variant | " + " | ".join(f"updates={u}" for u in updates) + " |",
        "|" + "---|" * (len(updates) + 1),
    ]
    for name in variants:
        cells = [_fci(p["stale_dominance_rate"]) for p in report["variants"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _locality_section(report: dict) -> list[str]:
    variants = list(report["variants"])
    updates = [p["num_updates"] for p in report["variants"][variants[0]]["points"]]
    lines = [
        "## Edit locality",
        "",
        "Each write's *edit* is `DeltaS_t = S_t - Diag(alpha_t) S_{t-1}`, "
        "computed from replayed state trajectories (captured traces only — no "
        "model reruns). **Relative update norm** `||DeltaS_t||_F / ||S_t||_F` "
        "measures edit magnitude (the first write into an empty state always "
        "scores 1). **Key-axis alignment** `||k_t k_t^T DeltaS_t||_F / "
        "||DeltaS_t||_F` measures how concentrated the edit is along the "
        "write key: exactly 1 for KDA/RLS-lite (`beta k`, `beta_eff k` write "
        "vectors) and for EDA's `e_t = k_t` collapse, smaller when the edit "
        "spreads off-key (the precision-gated gain `g_t`).",
        "",
        "### Relative update norm",
        "",
        "| variant | " + " | ".join(f"updates={u}" for u in updates) + " |",
        "|" + "---|" * (len(updates) + 1),
    ]
    for name in variants:
        cells = [_f(p["edit_relative_update_norm"]) for p in report["variants"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Key-axis alignment",
        "",
        "| variant | " + " | ".join(f"updates={u}" for u in updates) + " |",
        "|" + "---|" * (len(updates) + 1),
    ]
    for name in variants:
        cells = [_f(p["edit_key_axis_alignment"]) for p in report["variants"][name]["points"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _calibration_section(report: dict) -> list[str]:
    variants = list(report["variants"])
    signals = list(report["variants"][variants[0]]["signals"])
    lines = [
        "## Uncertainty calibration (#28 experiment, default config)",
        "",
        "Do internal uncertainty signals predict recall failure? AUROC of "
        "each pre-registered signal against the failure label on the "
        "key-collision task (cluster-bootstrap 95% CIs; 0.5 = chance, < 0.5 ="
        " anti-correlated, reported as-is). Full details and pre-registration"
        " in PR #47.",
        "",
        "### Failure rates",
        "",
        "| variant | failure rate | mean recall cosine |",
        "|---|---|---|---|",
    ]
    for name in variants:
        section = report["variants"][name]
        lines.append(
            f"| {name} | {_fci(section['failure_rate'])} | "
            f"{_f(section['mean_recall_cosine'])} |"
        )
    lines += [
        "",
        "### AUROC of each uncertainty signal",
        "",
        "| signal | " + " | ".join(variants) + " |",
        "|" + "---|" * (len(variants) + 1),
    ]
    for signal in signals:
        cells = []
        for name in variants:
            entry = report["variants"][name]["signals"][signal]
            cells.append(_fci(entry["auroc"]) if entry["available"] else "unavailable")
        lines.append(f"| {signal} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _routing_section(report: dict) -> list[str]:
    variants = list(report["variants"])
    signals = list(report["variants"][variants[0]]["signals"])
    lines = [
        "## Consequence: hybrid routing (#38 experiment, default config)",
        "",
        "Uncertainty-aware routing between the compressed memory and an "
        "exact-attention fallback. Each cell shows the best and worst "
        "operating points over the pre-registered threshold grid, as "
        "`gain @ fallback rate` (gain = recall improvement over a "
        "matched-budget fixed-ratio router at that operating point; "
        "fallback rate = cost proxy, no wall-clock measured). Flat and "
        "negative signals are shown in full, not filtered. Full grid in "
        "PR #48.",
        "",
        "### Endpoints",
        "",
        "| variant | compressed accuracy | exact accuracy | headroom |",
        "|---|---|---|---|",
    ]
    for name in variants:
        section = report["variants"][name]
        lines.append(
            f"| {name} | {_f(section['compressed_accuracy'])} | "
            f"{_f(section['exact_accuracy'])} | {_f(section['exact_headroom'])} |"
        )
    lines += [
        "",
        "### Best / worst adaptive gain over matched fixed-ratio routing",
        "",
        "| signal | " + " | ".join(variants) + " |",
        "|" + "---|" * (len(variants) + 1),
    ]
    for signal in signals:
        cells = []
        for name in variants:
            entry = report["variants"][name]["signals"][signal]
            if not entry["available"]:
                cells.append("unavailable")
                continue
            points = entry["operating_points"]
            best = max(points, key=lambda p: p["gain_vs_fixed_ratio"])
            worst = min(points, key=lambda p: p["gain_vs_fixed_ratio"])
            cells.append(
                f"{_f(best['gain_vs_fixed_ratio'])} @ {_f(best['fallback_rate'])}"
                f" / {_f(worst['gain_vs_fixed_ratio'])} @ {_f(worst['fallback_rate'])}"
            )
        lines.append(f"| {signal} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


_PROSE_HEADER = """# Phase 4 results report — Precision-Gated / Kalman Delta Attention

This report is **generated from checked-in pre-registered configs** by
`examples/phase4_results_report.py` (issue #34, epic #25). Do not edit the
tables by hand; regenerate with:

```
python examples/phase4_results_report.py
```

All experiments are deterministic given the seeds in the embedded configs.
Claims are limited to the tested synthetic regime (acceptance criterion).

## Hypothesis and related-work boundary

**Hypotheses (fixed in the Phase 4 epic #25 before these experiments ran).**

- **H1 — mechanism.** Diagonal precision/evidence gating (a Kalman-flavored,
  per-channel write-strength control) improves recall over fixed-beta KDA
  under key-collision and overwrite stress.
- **H2 — measurement.** The variants' *native* uncertainty signals
  (precision along the key, evidence along the key) predict recall failure
  well enough to route hard queries to an exact fallback, buying recall per
  unit of fallback rate that a fixed-ratio router cannot; generic proxies
  (gate entropy, prediction-error norm, state effective rank) were expected
  to be weaker and are reported regardless of outcome.

**Related-work boundary.** KDA itself comes from *Kimi Linear* (Yang et al.,
arXiv:2510.26692) and is this repository's base architecture. **GDN-2**
(Gated DeltaNet-2, Hatamizadeh, Choi, Kautz, arXiv:2605.22791) and **EDA**
(Erase-then-Delta Attention, Li, Zhang, Luo et al., arXiv:2606.26560) are
published **prior baselines**, implemented in #37 strictly as references —
they are not contributions of this repository, and their sequential reference
recurrences intentionally omit the papers' chunkwise/kernel algorithms.
Phase 4's own contribution is measurement infrastructure (diagnostics #27,
calibration #28, routing #38) plus two small educational prototypes
(KDA-RLS-lite #36, diagonal precision-gated KDA #33). Nothing here is a
trained model or a production kernel.

## Exact recurrence definitions

All variants share the readout `o_t = S_t^T q_t` and differ only in the state
update (`S_bar = Diag(alpha_t) S_{t-1}` is the common decay step):

- **KDA** (base architecture, arXiv:2510.26692):
  `S_t = S_bar + beta_t k_t (v_t - S_bar^T k_t)^T` — scalar delta gate
  `beta_t` ties erase and write strength together.
- **KDA-RLS-lite** (#36 prototype): the KDA update with
  `beta_eff,t = clamp(beta_t / (lambda + e_t), beta_min, beta_max)`, where
  `e_t = sum_c n_{t,c} k_{t,c}^2` is the occupancy of a decaying evidence
  accumulator `n_t = rho alpha_t ⊙ n_{t-1} + k_t^2`. Repeated writes along a
  direction damp later writes there.
- **Precision-gated KDA** (#33 prototype):
  `g_t = (P_t^- ⊙ k_t) / (lambda + sum_c P^-_{t,c} k_{t,c}^2)`,
  `S_t = S_bar + g_t (v_t - S_bar^T k_t)^T`, with a diagonal precision
  time update (`P / forgetting + process_noise`) and an RLS-style downdate
  (`P ⊙ (1 - g ⊙ k)`), clamped to `[p_min, p_max]`.
- **GDN-2** (prior baseline, arXiv:2605.22791):
  `S_t = (I - k_t (b_t ⊙ k_t)^T) S_bar + k_t (w_t ⊙ v_t)^T` with
  channel-wise erase/write gates; tying `b_t = w_t = beta_t` recovers KDA
  (paper Appendix A.5) — the configuration used on the shared gate path.
- **EDA** (prior baseline, arXiv:2606.26560):
  `S_t = (I - beta_t k_t k_t^T)(I - gamma_t e_t e_t^T) S_bar
  + beta_t k_t v_t^T` with an independently addressed L2-normalized erase
  direction; `gamma = 0` recovers KDA, and `e_t = k_t` collapses to a
  stronger same-address correction — the configuration used here, since the
  shared task path has no learned erase addresses.

## Task generators and seeds

All tasks come from the #35 generators (`kda/tasks/generators.py`);
gate distributions mirror the #28 pre-registration
(`alpha = sigmoid(N(0,1) + 2.0)`, `beta = sigmoid(N(0,1) + 1.0)`, drawn
from `seed + 10000` independently of task content). Seeds and sizes are
embedded verbatim in the configs below.

"""

_PROSE_OVERHEAD = """## Computational overhead

Asymptotic per-step costs of the **sequential reference recurrences** in this
repository (no chunkwise WY forms, no fused kernels). No wall-clock latency
is measured anywhere in Phase 4; the hybrid-routing experiments use the
*fallback rate* (fraction of queries answered exactly) as the only cost
proxy.

| mechanism | state memory | per-step update cost | notes |
|---|---|---|---|
| KDA | `O(d_h * d_v)` | `O(d_h * d_v)` | one rank-1 update |
| KDA-RLS-lite (#36) | `O(d_h * d_v)` + `O(d_h)` evidence | `O(d_h * d_v)` + `O(d_h)` | evidence accumulator + scalar damping |
| precision-gated KDA (#33) | `O(d_h * d_v)` + `O(d_h)` precision | `O(d_h * d_v)` + `O(d_h)` | diagonal precision update/downdate |
| GDN-2 (prior baseline) | `O(d_h * d_v)` | `O(d_h * d_v)` | tied config identical to KDA |
| EDA (prior baseline) | `O(d_h * d_v)` | `O(d_h * d_v)` | + one rank-1 erase when enabled |
| exact-attention fallback (#38) | `O(L * (d_h + d_v))` raw cache | `O(L * (d_h + d_v))` per query | the cost the router avoids on easy queries |

The educational prototypes deliberately stay sequential and readable; no
production-performance or kernel-level speedup is claimed.

## Negative results and limitations

Reported per the repository's honest-negatives rule; numbers are in the
tables above.

- **Generic uncertainty proxies are flat or anti-correlated** in this
  regime: `gate_entropy` sits at chance in the calibration table and never
  beats fixed-ratio routing; `prediction_error_norm` is *anti*-correlated
  with failure (AUROC < 0.5, negative routing gains at most operating
  points); `state_effective_rank` is strongly anti-correlated — a recency
  artifact (the state fills with time while failures concentrate on early,
  decayed writes), not a usable signal here.
- **`key_collision_score` is only weakly positive** — directionally useful
  but far below the native precision/evidence signals it is a proxy for.
- **The baselines do not differentiate on the shared gate path**: GDN-2
  (tied) and EDA (`gamma = 0`) coincide with KDA by construction, so the
  capacity/interference tables contain duplicate rows by design (they
  double as regression checks of the paper ties). The EDA overwrite arm uses
  the `e_t = k_t` collapse with a pre-registered constant gate — the paper's
  learned, independently addressed erase is *not* evaluated.
- **Mechanism gains are modest, not transformative**: precision/evidence
  gating buys a few accuracy points at high collision cosines and reduces
  stale-value dominance at these defaults; it does not remove the capacity
  cliff.
- **Scope**: synthetic tasks only; gates are sampled, not learned; no
  training runs; sequential reference recurrences only; ECE uses a
  deterministic min-max score-to-probability mapping (no fitted
  recalibration); bootstrap CIs are percentile intervals; claims are limited
  to the tested regime (acceptance criterion).

## Pre-registered configs (embedded verbatim)

"""


def build_report_markdown(
    *,
    capacity_config: CapacityConfig | None = None,
    overwrite_config: OverwriteConfig | None = None,
    calibration_config: CalibrationConfig | None = None,
    routing_config: RoutingExperimentConfig | None = None,
) -> str:
    """Run the pre-registered Phase 4 experiments and render the report.

    Defaults are the checked-in default configs; the function is
    deterministic (seeded experiments, fixed formatting, no timestamps), so
    regenerating reproduces the checked-in ``docs/PHASE4_RESULTS.md``
    byte-for-byte (tested).
    """

    capacity_config = capacity_config or CapacityConfig()
    overwrite_config = overwrite_config or OverwriteConfig()
    calibration_config = calibration_config or CalibrationConfig()
    routing_config = routing_config or RoutingExperimentConfig()

    capacity_report = run_capacity_experiment(capacity_config)
    overwrite_report = run_overwrite_experiment(overwrite_config)
    calibration_report = run_calibration_experiment(calibration_config)
    routing_report = run_routing_experiment(routing_config)

    parts = [_PROSE_HEADER]
    parts += _capacity_section(capacity_report) + [""]
    parts += _overwrite_section(overwrite_report) + [""]
    parts += _locality_section(overwrite_report) + [""]
    parts += _calibration_section(calibration_report) + [""]
    parts += _routing_section(routing_report) + [""]
    parts.append(_PROSE_OVERHEAD)
    parts += _config_block("Capacity/interference config (issue #34 sweeps)",
                           capacity_report["config"])
    parts += _config_block("Overwrite/recency config (issue #34 sweeps)",
                           overwrite_report["config"])
    parts += _config_block("Calibration config (#28)",
                           calibration_report["config"])
    parts += _config_block("Hybrid-routing config (#38)",
                           routing_report["config"])
    return "\n".join(parts).rstrip("\n") + "\n"


def main() -> None:
    markdown = build_report_markdown()
    REPORT_PATH.write_text(markdown)
    print(f"wrote {REPORT_PATH} ({len(markdown.splitlines())} lines)")


if __name__ == "__main__":
    main()
