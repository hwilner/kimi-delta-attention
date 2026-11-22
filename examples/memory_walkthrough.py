"""
Memory walkthrough: KDA's fixed-size state vs a growing KV cache (issue #17).

This script is the "see it for yourself" companion to
``docs/EXTENDED_INTRODUCTION.md``. That document explains *why* full
attention re-reads the whole bookshelf while KDA keeps a single notebook;
this script lets you watch both strategies decode the same synthetic
sequence, step by step, and prints two things for each context length:

1. **Memory** — how many bytes each strategy must carry forward to decode
   the next token:

   - naive full attention: the **KV cache**, one key vector and one value
     vector per past token, so memory grows *linearly* with the context;
   - KDA: the recurrent **state matrix** ``S`` of shape
     ``(head_dim, value_dim)``, whose size does not depend on the context
     at all — token 10 and token 10,000 cost exactly the same to remember.

2. **Step time** — how long a single next-token step takes on this CPU as
   the context grows. The naive step re-reads all cached keys (its cost
   grows with the context); the KDA step touches only the fixed state
   (its cost stays flat).

Run it from the repository root:

    PYTHONPATH=. python examples/memory_walkthrough.py

Boundary (please read before quoting numbers)
---------------------------------------------
These are *illustrative* measurements on synthetic data with a pure-Python
naive reference implemented below — not a rigorous benchmark and not a
claim about production kernels, fused implementations, or wall-clock
performance of any real system. Absolute timings vary from machine to
machine; only the *scaling shape* (flat vs growing) is the point. Seeding
makes the data and the memory figures deterministic; timings are not
deterministic and are printed as rough averages only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from kda.core import KDACore


@dataclass(frozen=True)
class WalkthroughRow:
    """One context length, both strategies."""

    context_length: int
    kda_state_bytes: int      # bytes of the KDA recurrent state
    kv_cache_bytes: int       # bytes of the naive KV cache
    kda_step_ms: float        # illustrative per-step time, KDA
    naive_step_ms: float      # illustrative per-step time, naive attention


def _make_stream(context_length: int, head_dim: int, value_dim: int, seed: int):
    """Synthetic token stream: one token per step, single head, batch size 1.

    Everything is drawn from a seeded generator so the walkthrough is
    reproducible. Keys are L2-normalized (as ``KDACore`` expects), alpha is
    a per-channel decay in [0, 1], beta a scalar write gate in [0, 1].
    """

    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, context_length, head_dim, generator=g)
    k = torch.nn.functional.normalize(
        torch.randn(1, context_length, head_dim, generator=g), dim=-1
    )
    v = torch.randn(1, context_length, value_dim, generator=g)
    alpha = torch.rand(1, context_length, head_dim, generator=g)
    beta = torch.rand(1, context_length, 1, generator=g)
    return q, k, v, alpha, beta


def _kda_prefill(core: KDACore, k, v, alpha, beta) -> torch.Tensor:
    """Run the KDA recurrence over the context and return the state ``S``.

    ``S`` has shape ``(batch, head_dim, value_dim)`` — this matrix is the
    *entire* memory KDA carries between steps, no matter how long the
    context was. See ``docs/EXTENDED_INTRODUCTION.md`` section 2 ("keep a
    notebook instead").
    """

    # We pass zero queries because prefill only needs the state; KDACore
    # returns outputs too, which we ignore here.
    dummy_q = torch.zeros_like(k)
    _, state = core(dummy_q, k, v, alpha, beta)
    return state


def _kda_step(core: KDACore, state, q_t, k_t, v_t, alpha_t, beta_t):
    """One KDA decode step: update the fixed state, read out with q_t.

    Every tensor involved has a shape that does NOT mention the context
    length — that is the whole point of the walkthrough.
    """

    out, new_state = core(
        q_t.unsqueeze(1), k_t.unsqueeze(1), v_t.unsqueeze(1),
        alpha_t.unsqueeze(1), beta_t.unsqueeze(1),
        initial_state=state,
    )
    return out.squeeze(1), new_state


def _naive_step(q_t, cache_k, cache_v):
    """One naive full-attention decode step against the growing KV cache.

    This is ordinary softmax attention, implemented inline so the cost is
    visible: the new query is compared against **every cached key**
    (``cache_k`` has one row per past token), then the output is a weighted
    sum of **every cached value**. Longer context -> longer cache -> more
    work per step. Nothing here is optimized on purpose.
    """

    scores = (cache_k @ q_t.transpose(0, 1)).squeeze(-1)  # (1, T)
    scores = scores / (q_t.shape[-1] ** 0.5)
    weights = torch.softmax(scores, dim=-1)               # (1, T)
    return weights @ cache_v                              # (1, value_dim)


def _time_step(fn, repeats: int, warmup: int) -> float:
    """Average milliseconds per call — illustrative only, never a benchmark."""

    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / max(repeats, 1) * 1e3


def run_walkthrough(
    context_lengths=(16, 64, 256, 1024),
    head_dim: int = 32,
    value_dim: int = 32,
    repeats: int = 25,
    warmup: int = 5,
    seed: int = 0,
):
    """Decode a synthetic stream with both strategies at each context length.

    Returns a list of ``WalkthroughRow``. The byte counts are exact and
    deterministic; the timings are rough single-machine averages (see the
    module docstring's boundary note).
    """

    core = KDACore(head_dim=head_dim, value_dim=value_dim)
    rows = []
    with torch.no_grad():
        for length in context_lengths:
            q, k, v, alpha, beta = _make_stream(length, head_dim, value_dim, seed)

            # --- KDA: fold the whole context into one fixed-size state. ---
            state = _kda_prefill(core, k, v, alpha, beta)
            kda_bytes = state.numel() * state.element_size()

            # --- Naive: keep every key and value vector around, forever. ---
            cache_k = k.clone()  # (1, T, head_dim)
            cache_v = v.clone()  # (1, T, value_dim)
            kv_bytes = (cache_k.numel() + cache_v.numel()) * k.element_size()

            # Time the *next* step for both, given the same context.
            q_t, k_t = q[:, -1], k[:, -1]
            v_t, alpha_t, beta_t = v[:, -1], alpha[:, -1], beta[:, -1]
            kda_ms = _time_step(
                lambda: _kda_step(core, state, q_t, k_t, v_t, alpha_t, beta_t),
                repeats, warmup,
            )
            naive_ms = _time_step(
                lambda: _naive_step(q_t, cache_k, cache_v),
                repeats, warmup,
            )

            rows.append(
                WalkthroughRow(
                    context_length=length,
                    kda_state_bytes=kda_bytes,
                    kv_cache_bytes=kv_bytes,
                    kda_step_ms=kda_ms,
                    naive_step_ms=naive_ms,
                )
            )
    return rows


def format_report(rows) -> str:
    """Render the walkthrough as a human-readable table."""

    lines = [
        "=" * 74,
        "KDA vs naive full attention: what decoding one more token costs",
        "=" * 74,
        "",
        "Memory carried between steps (bytes, float32, single head):",
        "",
        f"  {'context':>8} | {'KDA state':>12} | {'KV cache':>12} | {'ratio':>8}",
        f"  {'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 8}",
    ]
    for r in rows:
        ratio = r.kv_cache_bytes / r.kda_state_bytes
        lines.append(
            f"  {r.context_length:>8} | {r.kda_state_bytes:>12} | "
            f"{r.kv_cache_bytes:>12} | {ratio:>7.1f}x"
        )
    lines += [
        "",
        "The KDA column never moves: the state is one (head_dim x value_dim)",
        "matrix. The KV cache column grows by two vectors per token, so the",
        "ratio grows linearly with the context.",
        "",
        "Illustrative per-step decode time on this CPU (ms/step, rough):",
        "",
        f"  {'context':>8} | {'KDA':>10} | {'naive':>10}",
        f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}",
    ]
    for r in rows:
        lines.append(
            f"  {r.context_length:>8} | {r.kda_step_ms:>10.3f} | "
            f"{r.naive_step_ms:>10.3f}"
        )
    lines += [
        "",
        "The KDA step touches only the fixed state, so it stays flat; the",
        "naive step re-reads the whole cache, so it grows with the context.",
        "",
        "Reminder: synthetic data, pure-Python naive reference, one machine.",
        "These timings illustrate scaling shape only — they are not a",
        "benchmark and say nothing about production kernels or latency.",
        "See docs/EXTENDED_INTRODUCTION.md for the intuition behind the",
        "fixed-size notebook.",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = run_walkthrough()
    print(format_report(rows))


if __name__ == "__main__":
    main()
