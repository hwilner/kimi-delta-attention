# Project Status and Task Tracker

This file tracks the state of the project: which tasks are open, in progress,
or completed, and how they relate to each other. It is updated as part of each
merged pull request.

## Completed milestones

- Educational implementation of Kimi Delta Attention (KDA) based on the Kimi
  Linear paper (arXiv:2510.26692): `KDACore`, `KDAChunkwise`, and the
  `KimiDeltaAttention` layer, with unit tests and usage examples.
- Documentation set: `README.md`, `docs/ARCHITECTURE.md`,
  `docs/EXTENDED_INTRODUCTION.md`, and contribution/process docs.

## Epics

| Epic | Title | Status |
|---|---|---|
| #1 | Phase 1: Associative-recall validation suite | Open |
| #2 | Phase 2: State inspection tooling | Open |
| #3 | Phase 3 (novel research): Delta-state surgery | Open |
| #25 | Phase 4: Precision-Gated / Kalman Delta Attention research track | Open |

## Epic: Phase 1 \u2014 Associative-recall validation suite (#1)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #4 | MQAR-style synthetic key\u2192value recall task generators + data versioning | S | \u2014 (ready) |
| #5 | Train small KDA on recall suite + report capacity curve | S | #4 |
| #6 | Minimal additive linear-attention + DeltaNet + Gated DeltaNet baselines for comparison | S | #4 |

## Epic: Phase 2 \u2014 State inspection tooling (#2)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #7 | State-capture utilities: expose/serialize S_t, \u03b1_t, \u03b2_t trajectories during a forward pass + tests | S | \u2014 (ready) |
| #8 | Least-squares readout probe: recover stored values V\u0302 = S\u1d40K, key recovery via pseudoinverse, fidelity metrics | S | #7 |
| #9 | Channel-gating logger + per-channel \u03b1 statistics report | XS | #7 |

## Epic: Phase 3 \u2014 Delta-state surgery (#3)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #10 | Targeted-edit operator: closed-form inverse-delta state update (k_old \u2192 v_new) + unit tests on synthetic states | S | #7 |
| #11 | Edit-specificity experiment: post-edit recall of target vs unedited pairs, interference vs key similarity | S | #10, #8 |
| #12 | Erase operation (\u03b2=1, v=0 / forced channel decay) + crosstalk structure measurement | S | #10 |
| #13 | Per-channel attribution experiment: ablate top-attributed channels, measure selective recall loss vs random ablation | S | #9, #5 |
| #14 | Capacity/interference curve: linear vs DeltaNet vs Gated DeltaNet vs KDA editability comparison | S | #6, #10 |
| #15 | Results report against pre-registered metrics with honest negatives | S | #11, #12, #13, #14 |

## Epic: Phase 4 \u2014 Precision-Gated / Kalman Delta Attention (#25)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #26 | Shared delta-memory variant interface and registry | S | \u2014 (in progress) |
| #35 | Key-collision and overwrite synthetic task generators | S | \u2014 (in progress) |
| #36 | KDA-RLS-lite prototype with evidence-modulated beta | S | #26, #35 (in progress) |
| #33 | Diagonal precision-gated KDA prototype | S | #26 |
| #27 | State/gate diagnostics for memory capacity and uncertainty experiments | S | #7, #26 |
| #37 | GDN-2 and EDA reference baselines for comparison | S | #26 |
| #28 | Calibration experiment: predicted uncertainty vs recall failure | S | #27, #35, #36 |
| #38 | Adaptive hybrid-routing prototype: KDA vs exact attention fallback | S | #27, #28 |
| #34 | Phase 4 results report with pre-registered metrics and honest negatives | S | #28, #33, #36, #38 |

## Standalone (no epic)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #16 | pyproject.toml packaging + pytest config cleanup | XS | \u2014 (ready) |
| #17 | Example script: KDA vs naive attention memory/latency walkthrough with comments | XS | \u2014 (ready) |
| #18 | docs/API_REFERENCE.md for the kda package public API | S | \u2014 (ready) |

## Suggested contribution paths

- **"I want a quick win":** #16, #17, or #9 (after #7) \u2014 XS cards, clear
  acceptance criteria, no research risk.
- **"I want to build ML infrastructure":** #4 \u2192 #5, then #7 \u2192 #8. You end up
  owning the measurement stack everything else depends on.
- **"I want to do the novel research":** start with #7 (state capture) \u2192 #10
  (edit operator) \u2192 #11 (edit specificity). This is the critical path to the
  first delta-state-surgery result.
