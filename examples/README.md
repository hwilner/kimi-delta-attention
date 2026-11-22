# Examples index

Runnable entry points for the repository's research tooling. Every experiment
script is deterministic (seeded), prints a human-readable summary followed by
the full JSON report, and finishes in seconds on a CPU.

## Running

From the repository root:

```bash
pip install -e .                              # once, or set PYTHONPATH per run
PYTHONPATH=. python examples/<name>.py
```

The runtimes below are indicative CPU times for the checked-in default
configs on a small sandbox machine — usage guidance only, not benchmarks.

## Phase 4 track (epic #25)

| Example | Issue / PR | What it shows | Runtime |
|---|---|---|---|
| `diagnostics_report.py` | #27 / #45 | State/gate diagnostics (state norms, spectra, gate entropy, key-collision counts, prediction-error norms) computed from captured traces — no diagnostic ever reruns the model. | seconds |
| `calibration_experiment.py` | #28 / #47 | Do internal uncertainty signals predict recall failure? AUROC / average precision / ECE with cluster-bootstrap CIs for KDA, KDA-RLS-lite, and precision-gated KDA. | seconds |
| `hybrid_routing.py` | #38 / #48 | Uncertainty-aware routing to an exact-attention fallback vs a matched-budget fixed-ratio baseline, with a per-query audit log. | seconds |
| `precision_gated_ablation.py` | #33 / #51 | Hyperparameter ablation of precision-gated KDA (lambda, forgetting, process noise) at two operating points; the default cell reproduces the #34 capacity row exactly. | seconds |
| `rls_lite_ablation.py` | #36 / #54 | Hyperparameter ablation of KDA-RLS-lite (evidence decay, beta clamps); the default cell reproduces the #34 capacity row exactly. | seconds |
| `calibration_robustness.py` | #28 / #50 | The #28 calibration experiment re-run across a 9-cell regime grid (sequence load x collision cosine). | ~10 s |
| `routing_robustness.py` | #38 / #52 | The #38 routing experiment re-run across the same 9-cell regime grid. | seconds |
| `phase4_results_report.py` | #34 / #49 | Regenerates `docs/PHASE4_RESULTS.md` from the checked-in pre-registered configs. Byte-for-byte reproduction is enforced by a test — never edit the report by hand. | ~15 s |
| `report_consistency.py` | #34 / #53 | Cross-checks every cross-checked report cell against a fresh recomputation of the extension sweeps; exits nonzero on any mismatch. | ~10 s |
| `report_table_audit.py` | #34 / #57 | Audits the report tables rendered directly from the #34 configs (interference, overwrite/recency, edit-locality); exits nonzero on any mismatch. | seconds |

Suggested reading order: diagnostics (#27) → calibration (#28) → routing
(#38) → robustness sweeps (#50, #52) and ablations (#51, #54) → the report
(#34) → the checkers (#53, #57).

## Other

| Example | What it shows |
|---|---|
| `basic_usage.py` | Minimal `KimiDeltaAttention` layer usage on synthetic tensors. |
| `memory_walkthrough.py` | #17 — KDA vs naive full attention, decoded step by step: fixed recurrent state vs growing KV cache, with illustrative per-step timings (not a benchmark). Heavily commented companion to `docs/EXTENDED_INTRODUCTION.md`. |

## Conventions

- Synthetic tasks only; no wall-clock, kernel, or production-performance
  claims. The routing cost proxy is the fallback rate, never latency.
- Honest negatives: flat or anti-correlated signals and collapsed ablation
  cells are printed alongside strong ones, never filtered.
- Experiment configs are frozen dataclasses embedded verbatim in their
  reports (pre-registration); scores are quantized at collection time so
  rank-based statistics reproduce across float32 builds.
- GDN-2 and EDA are prior-art baselines (#37), not contributions of this
  repository.
