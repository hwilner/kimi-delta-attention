# Contributing

Thanks for your interest in contributing to this educational KDA
implementation! This file explains how to pick up work and how we keep the
project honest.

## Finding work: the issue-card system

All planned work lives in GitHub issues with a consistent card format:

- **Title** starts with a size tag: `[XS]` (under an hour), `[S]` (a focused
  session), `[L]` (multi-part, usually an epic).
- **## Task** — what to build, including Size / Work type / Task labels.
- **## Dependencies** — "Blocked by:" entries naming the exact issues that must
  land first. Don't start a blocked card unless you're also doing the blocker.
- **## Acceptance criteria** — the checklist a PR must satisfy.
- **## Boundary** — what is explicitly *out of scope* for that card.

Epics (`[L]`) carry a **## Sub-tasks** checklist linking their children.
`docs/PROJECT.md` mirrors the backlog as an in-repo board with suggested
contribution paths.

## Development setup

```bash
git clone https://github.com/hwilner/kimi-delta-attention.git
cd kimi-delta-attention
pip install -e ".[dev]"
pytest tests/ -v
```

All tests run on synthetic/random data — no dataset downloads needed.

## Branch and PR conventions

- Branch from `main`: `short-desc-issueN` (e.g. `state-capture-issue12`).
- One issue card per PR. Reference the issue in the PR description
  (`Closes #N` when complete).
- Fill out the PR template sections; keep the "Task key" comment intact.
- CI-equivalent bar (run locally before pushing): `pytest tests/` passes, and
  any new module has at least one synthetic-data test.

## Integrity rules

1. **Tests pass** before merge — no commented-out failures.
2. **Honest negatives.** If an experiment fails or a metric is flat, report it
   as-is. A well-documented negative result is a valid contribution here
   (Phase 3 explicitly pre-registers this).
3. **Pre-registered metrics.** For experiment cards, evaluation metrics are
   fixed in the issue *before* results are collected; changing them afterwards
   requires documenting why.
4. **Synthetic scope.** Claims from synthetic benchmarks stay synthetic —
   don't extrapolate to real LLM behavior in docs or issue comments.
5. **Citations.** Any claim about prior work in docs must cite the papers
   listed in `docs/INTRODUCTION.md` / `docs/ROADMAP.md` reference sections.

## Questions

Open a discussion-style issue if a card is ambiguous — refining a backlog item
before starting it is encouraged (every card footer says so).
