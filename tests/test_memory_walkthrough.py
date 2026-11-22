"""Synthetic tests for examples/memory_walkthrough.py (issue #17).

The walkthrough is an educational script, so the tests pin the parts that
must be true by construction rather than timings: the KDA state size is
constant across context lengths, the KV cache size grows linearly, the
naive reference matches PyTorch's own causal attention, data generation is
seeded, and the rendered report carries the honest-boundary disclaimer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_walkthrough():
    spec = importlib.util.spec_from_file_location(
        "memory_walkthrough", REPO_ROOT / "examples" / "memory_walkthrough.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_rows(module):
    return module.run_walkthrough(
        context_lengths=(4, 8, 16), head_dim=4, value_dim=4,
        repeats=1, warmup=0, seed=0,
    )


def test_kda_state_bytes_constant_across_lengths():
    module = _load_walkthrough()
    rows = _tiny_rows(module)
    assert len(rows) == 3
    assert {r.kda_state_bytes for r in rows} == {4 * 4 * 4}  # d_h*d_v*float32


def test_kv_cache_bytes_grow_linearly_with_context():
    module = _load_walkthrough()
    rows = _tiny_rows(module)
    for r in rows:
        # Two float32 vectors (key + value) per cached token.
        assert r.kv_cache_bytes == r.context_length * (4 + 4) * 4
    assert rows[0].kv_cache_bytes < rows[1].kv_cache_bytes < rows[2].kv_cache_bytes


def test_stream_generation_is_seeded_and_normalized():
    module = _load_walkthrough()
    a = module._make_stream(8, 4, 4, seed=7)
    b = module._make_stream(8, 4, 4, seed=7)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)
    keys = a[1]
    assert torch.allclose(keys.norm(dim=-1), torch.ones_like(keys[..., 0]))
    alpha, beta = a[3], a[4]
    assert (alpha >= 0).all() and (alpha <= 1).all()
    assert (beta >= 0).all() and (beta <= 1).all()


def test_naive_reference_matches_torch_causal_attention():
    """The inline naive step must be real softmax attention, not a stub."""

    module = _load_walkthrough()
    torch.manual_seed(0)
    length, head_dim, value_dim = 6, 4, 4
    q, k, v, _, _ = module._make_stream(length, head_dim, value_dim, seed=1)
    q_t = q[:, -1]  # decode step: the last token's query

    got = module._naive_step(q_t, k, v)

    # One-shot causal attention over the full sequence; the last row must
    # equal the decode-step output (the query attends to tokens 1..T).
    scores = q @ k.transpose(1, 2) / (head_dim ** 0.5)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    expected_last = (torch.softmax(scores, dim=-1) @ v)[:, -1]
    assert torch.allclose(got, expected_last, atol=1e-6)


def test_kda_step_keeps_state_shape_regardless_of_context():
    module = _load_walkthrough()
    core = module.KDACore(head_dim=4, value_dim=4)
    for length in (3, 12):
        q, k, v, alpha, beta = module._make_stream(length, 4, 4, seed=0)
        state = module._kda_prefill(core, k, v, alpha, beta)
        assert state.shape == (1, 4, 4)
        out, new_state = module._kda_step(
            core, state, q[:, -1], k[:, -1], v[:, -1], alpha[:, -1], beta[:, -1]
        )
        assert out.shape == (1, 4)
        assert new_state.shape == (1, 4, 4)
        assert torch.isfinite(out).all() and torch.isfinite(new_state).all()


def test_report_renders_and_keeps_the_boundary_disclaimer():
    module = _load_walkthrough()
    text = module.format_report(_tiny_rows(module))
    for needle in ("context", "KDA state", "KV cache", "not a", "benchmark"):
        assert needle in text
