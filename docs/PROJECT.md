# Project Board

In-repo mirror of the GitHub issue backlog. The issues are the source of truth;
this file is the readable overview. Rationale for every phase lives in
`docs/ROADMAP.md`; engineering selection rules in `docs/METHODS.md`; how to pick
up a card in `CONTRIBUTING.md`.

## Community implementation and extension welcome

Everyone is welcome to help extend this educational implementation of the
original Kimi Linear work. The cards below cover implementation, tests,
baselines, experiments, examples, API documentation, and packaging; each is a
valid contribution path. If a useful extension is not represented by a card,
open a discussion-style issue to propose it before starting a large change.
Contributors should retain the repository's educational scope and follow the
testing and evidence rules in `CONTRIBUTING.md`.

## Status: Ready to start

| Issue | Title | Size | Epic |
|---|---|---|---|
| #4 | MQAR-style synthetic key→value recall task generators + data versioning | S | Phase 1 (#1) |
| #7 | State-capture utilities: expose/serialize S_t, α_t, β_t trajectories during a forward pass + tests | S | Phase 2 (#2) |
| #16 | pyproject.toml packaging + pytest config cleanup | XS | Standalone |
| #17 | Example script: KDA vs naive attention memory/latency walkthrough with comments | XS | Standalone |
| #18 | docs/API_REFERENCE.md for the kda package public API | S | Standalone |

## Epic: Phase 1 — Associative-recall validation suite (#1)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #4 | MQAR-style synthetic key→value recall task generators + data versioning | S | — (ready) |
| #5 | Train small KDA on recall suite + report capacity curve | S | #4 |
| #6 | Minimal additive linear-attention + DeltaNet + Gated DeltaNet baselines for comparison | S | #4 |

## Epic: Phase 2 — State inspection tooling (#2)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #7 | State-capture utilities: expose/serialize S_t, α_t, β_t trajectories during a forward pass + tests | S | — (ready) |
| #8 | Least-squares readout probe: recover stored values V̂ = SᵀK, key recovery via pseudoinverse, fidelity metrics | S | #7 |
| #9 | Channel-gating logger + per-channel α statistics report | XS | #7 |

## Epic: Phase 3 — Delta-state surgery (novel research) (#3)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #10 | Targeted-edit operator: closed-form inverse-delta state update (k_old → v_new) + unit tests on synthetic states | S | #7 |
| #11 | Edit-specificity experiment: post-edit recall of target vs unedited pairs, interference vs key similarity | S | #10, #8 |
| #12 | Erase operation (β=1, v=0 / forced channel decay) + crosstalk structure measurement | S | #10 |
| #13 | Per-channel attribution experiment: ablate top-attributed channels, measure selective recall loss vs random ablation | S | #9, #5 |
| #14 | Capacity/interference curve: linear vs DeltaNet vs Gated DeltaNet vs KDA editability comparison | S | #6, #10 |
| #15 | Results report against pre-registered metrics with honest negatives | S | #11, #12, #13, #14 |

## Standalone (no epic)

| Issue | Title | Size | Blocked by |
|---|---|---|---|
| #16 | pyproject.toml packaging + pytest config cleanup | XS | — (ready) |
| #17 | Example script: KDA vs naive attention memory/latency walkthrough with comments | XS | — (ready) |
| #18 | docs/API_REFERENCE.md for the kda package public API | S | — (ready) |

## Suggested contribution paths

- **"I want a quick win":** #16, #17, or #9 (after #7) — XS cards, clear
  acceptance criteria, no research risk.
- **"I want to build ML infrastructure":** #4 → #5, then #7 → #8. You end up
  owning the measurement stack everything else depends on.
- **"I want to do the novel research":** start with #7 (state capture) → #10
  (edit operator) → #11 (edit specificity). This is the critical path to the
  first delta-state-surgery result.
- **"I want docs/education work":** #18, #17 — both immediately available.

## Rules for every card

1. **Tests pass.** `pytest tests/` green at merge; new code gets at least one
   synthetic-data test.
2. **Honest negatives.** Flat or negative experimental results are reported,
   never suppressed (see `CONTRIBUTING.md`).
3. **Pre-registered metrics.** Experiment cards evaluate against the metrics
   fixed in `docs/ROADMAP.md` Phase 3 *before* results are collected.

## Live GitHub Project

The public [Kimi Delta Attention — Contributions](https://github.com/users/hwilner/projects/8)
board contains issues #1–#18 and a `Workflow` field with `Blocked`, `Ready`,
`In progress`, and `Done` states mirroring this file. Epics retain their linked
sub-issues, so contributors can navigate between the board and the parent work.
This file remains the clone-readable mirror; GitHub issues are the source of
truth for scope, dependencies, and acceptance criteria.
