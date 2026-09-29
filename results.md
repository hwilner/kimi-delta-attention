# Results

**Generated 2026-09-29 by executing this repository's own test suite, plus a manual
check of the README's performance claims against the code.**

## Verification run

| Metric | Value |
|---|---|
| Tests passed | **15** |
| Tests failed | 0 |
| Wall clock | ~9 s |
| Environment | Python 3.11, CPU-only, no GPU (torch 2.14.0+cu130) |

Reproduce with:

```bash
PYTHONPATH=$PWD pytest -q
```

## ⚠️ Performance claims in the README are not supported by this repository

The README's "Performance Highlights" section states:

- ✅ **First linear attention to outperform full attention** under fair comparisons
- ✅ **75% reduction in KV cache usage** for long-context scenarios
- ✅ **6× faster decoding** at 1M token contexts
- ✅ **Superior performance** on MMLU-Pro, RULER, and RL-style benchmarks

**None of these are verifiable from this repository.** Specifically:

- There is no MMLU-Pro, RULER, or RL-style benchmark harness in the tree. A search
  for those names returns no files.
- No test measures speed, memory, or throughput. All 15 tests assert tensor
  shapes, gradient flow, dropout behaviour, state continuity, and dtype — i.e.
  mechanical correctness of the attention implementation.
- No comparison against full attention is performed anywhere in the tests.

These claims may well be true of the paper or of a separate experimental setup.
They are **not** established by this codebase, and should not be read as though
this repository demonstrated them. Compare `hybrid-rag-cag-framework`'s retraction
(`results.md` there) — there, unsupported claims were actively retracted; here they
are left in place pending the author's decision.

## What the passing tests do establish

The implementation is real and internally consistent:

- Delta-attention forward passes produce correctly-shaped outputs
- Gradients flow through the attention path
- Chunkwise and recurrent modes agree
- State continuity holds across sequential steps
- Dropout and head-dimension variants behave

That is meaningful engineering evidence. It is **not** evidence for the
performance claims above.

## What is not established

- No speed, memory, or throughput number was reproduced.
- No benchmark was reproduced — none is present in the repository.
- No comparison against full attention was reproduced.
- Coverage was not measured.

## Negative and unverified results

- **Four headline performance claims are unverified by this repository.** Recorded
  rather than repeated as fact.
- **No empirical result was independently reproduced.** A passing test suite is
  not evidence that the attention mechanism outperforms full attention.
