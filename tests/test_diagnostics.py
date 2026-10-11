"""Tests for the state/gate diagnostics module (issue #27)."""

import json

import pytest
import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.instrumentation import (
    gate_statistics,
    key_collision_statistics,
    prediction_error_metrics,
    replay_state_trajectory,
    state_metrics,
    summarize_trajectory,
)
from kda.variants import create_variant


def _synthetic_inputs(batch_size=2, seq_len=8, head_dim=16, value_dim=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(
        torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1
    )
    keys = F.normalize(
        torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1
    )
    values = torch.randn(batch_size, seq_len, value_dim, generator=generator)
    gates = {
        "alpha": torch.sigmoid(
            torch.randn(batch_size, seq_len, head_dim, generator=generator)
        ),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, 1, generator=generator)
        ),
    }
    return queries, keys, values, gates


# ---------------------------------------------------------------------------
# state_metrics
# ---------------------------------------------------------------------------


def test_state_metrics_shapes():
    states = torch.randn(2, 8, 16, 12)
    metrics = state_metrics(states)

    assert metrics.frobenius_norm.shape == (2, 8)
    assert metrics.spectral_norm.shape == (2, 8)
    assert metrics.condition_number.shape == (2, 8)
    assert metrics.effective_rank.shape == (2, 8)
    assert metrics.singular_values.shape == (2, 8, 12)  # min(d_h, d_v)


def test_state_metrics_known_diagonal():
    # diag(3, 4): singular values [4, 3], Frobenius 5, spectral 4, cond 4/3.
    state = torch.diag(torch.tensor([3.0, 4.0])).reshape(1, 1, 2, 2)
    metrics = state_metrics(state)

    assert metrics.frobenius_norm[0, 0].item() == pytest.approx(5.0)
    assert metrics.spectral_norm[0, 0].item() == pytest.approx(4.0)
    assert metrics.condition_number[0, 0].item() == pytest.approx(4.0 / 3.0)
    assert torch.allclose(
        metrics.singular_values[0, 0], torch.tensor([4.0, 3.0])
    )


def test_state_metrics_effective_rank_extremes():
    # Identity: uniform spectrum -> maximum effective rank = matrix order.
    identity = torch.eye(4).reshape(1, 1, 4, 4)
    # Rank-1: all mass in one singular value -> effective rank 1.
    rank_one = torch.zeros(1, 1, 4, 4)
    rank_one[0, 0, 0, 0] = 2.0

    identity_metrics = state_metrics(identity)
    rank_one_metrics = state_metrics(rank_one)

    assert identity_metrics.effective_rank[0, 0].item() == pytest.approx(4.0)
    assert rank_one_metrics.effective_rank[0, 0].item() == pytest.approx(1.0)
    # Rank-1 4x4 matrix is rank-deficient -> infinite condition number.
    assert torch.isinf(rank_one_metrics.condition_number[0, 0])


def test_state_metrics_zero_state():
    states = torch.zeros(2, 3, 8, 6)
    metrics = state_metrics(states)

    assert torch.all(metrics.frobenius_norm == 0)
    assert torch.all(metrics.spectral_norm == 0)
    assert torch.all(metrics.effective_rank == 0)
    # All-zero states are rank-deficient: condition number is inf by convention.
    assert torch.all(torch.isinf(metrics.condition_number))


def test_state_metrics_deterministic():
    generator = torch.Generator().manual_seed(0)
    states = torch.randn(2, 4, 8, 8, generator=generator)
    first = state_metrics(states)
    second = state_metrics(states)

    assert torch.equal(first.singular_values, second.singular_values)
    assert torch.equal(first.effective_rank, second.effective_rank)


def test_state_metrics_rejects_vector_input():
    with pytest.raises(ValueError, match="at least 2 dims"):
        state_metrics(torch.randn(4))


# ---------------------------------------------------------------------------
# gate_statistics
# ---------------------------------------------------------------------------


def test_gate_statistics_constant_tensor():
    # 0.5 is exactly representable in float32, so the std is exactly 0.
    stats = gate_statistics(torch.full((2, 8, 16), 0.5))

    assert stats.mean == pytest.approx(0.5)
    assert stats.std == pytest.approx(0.0)
    assert stats.min == pytest.approx(0.5)
    assert stats.max == pytest.approx(0.5)
    # All mass in a single bin -> zero histogram entropy.
    assert stats.entropy == pytest.approx(0.0)
    assert stats.num_elements == 2 * 8 * 16


def test_gate_statistics_uniform_beats_concentrated_entropy():
    uniform = torch.linspace(0.0, 1.0, steps=10_000)
    concentrated = torch.full((10_000,), 0.5) + 0.001 * torch.sin(
        torch.arange(10_000, dtype=torch.float32)
    )

    uniform_entropy = gate_statistics(uniform, bins=50).entropy
    concentrated_entropy = gate_statistics(concentrated, bins=50).entropy

    # Uniform over [0, 1] approaches the maximum log(50); the concentrated
    # gate occupies only a few bins.
    assert uniform_entropy > concentrated_entropy + 1.0


def test_gate_statistics_deterministic_and_range_clamped():
    # Values outside value_range are clamped into the edge bins (documented).
    gate = torch.tensor([-1.0, 0.25, 0.5, 0.75, 2.0])
    first = gate_statistics(gate, bins=4)
    second = gate_statistics(gate, bins=4)

    assert first == second
    # -1 and 2 land in the edge bins together with 0.25/0.75-side mass; the
    # distribution is spread over all 4 bins at most.
    assert 0.0 <= first.entropy <= torch.log(torch.tensor(4.0)).item() + 1e-6


def test_gate_statistics_rejects_empty_and_bad_bins():
    with pytest.raises(ValueError, match="non-empty"):
        gate_statistics(torch.empty(0))
    with pytest.raises(ValueError, match="bins"):
        gate_statistics(torch.rand(4), bins=0)
    with pytest.raises(ValueError, match="value_range"):
        gate_statistics(torch.rand(4), value_range=(1.0, 0.0))


# ---------------------------------------------------------------------------
# prediction_error_metrics
# ---------------------------------------------------------------------------


def test_prediction_error_metrics_known_values():
    errors = torch.zeros(1, 3, 4)
    errors[0, 0, 0] = 3.0  # norm 3
    errors[0, 1, 1] = 4.0  # norm 4
    errors[0, 2, 0] = 6.0  # norm 6 (final step)

    metrics = prediction_error_metrics(errors)

    assert metrics.l2_norm.shape == (1, 3)
    assert torch.allclose(metrics.l2_norm[0], torch.tensor([3.0, 4.0, 6.0]))
    assert metrics.mean == pytest.approx((3.0 + 4.0 + 6.0) / 3)
    assert metrics.max == pytest.approx(6.0)
    assert metrics.final == pytest.approx(6.0)


def test_prediction_error_metrics_rejects_vector_input():
    with pytest.raises(ValueError, match="at least 2 dims"):
        prediction_error_metrics(torch.randn(4))


# ---------------------------------------------------------------------------
# key_collision_statistics
# ---------------------------------------------------------------------------


def test_key_collision_identical_keys_collide():
    key = F.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    keys = key.repeat(1, 4, 1)  # one batch, four identical keys

    stats = key_collision_statistics(keys, threshold=0.9)

    assert stats.total_pairs == 6
    assert stats.collision_counts.tolist() == [6]
    assert stats.collision_fraction.tolist() == [1.0]
    assert stats.max_similarity[0].item() == pytest.approx(1.0)


def test_key_collision_orthogonal_keys_do_not_collide():
    keys = torch.eye(3).unsqueeze(0)  # one batch, three orthogonal keys

    stats = key_collision_statistics(keys, threshold=0.9)

    assert stats.collision_counts.tolist() == [0]
    assert stats.collision_fraction.tolist() == [0.0]
    assert stats.max_similarity[0].item() == pytest.approx(0.0, abs=1e-6)


def test_key_collision_threshold_sensitivity_and_matrix():
    generator = torch.Generator().manual_seed(0)
    keys = F.normalize(torch.randn(2, 16, 8, generator=generator), dim=-1)

    strict = key_collision_statistics(keys, threshold=0.99)
    loose = key_collision_statistics(keys, threshold=-1.0, return_matrix=True)

    # Every pair collides at threshold -1; none may be lost at 0.99.
    assert loose.collision_counts.tolist() == [strict.total_pairs] * 2
    assert torch.all(strict.collision_counts <= loose.collision_counts)
    assert loose.similarity_matrix.shape == (2, 16, 16)
    # Cosine self-similarity of normalized keys is 1 on the diagonal.
    diagonal = loose.similarity_matrix.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-6)


def test_key_collision_single_key_sequence():
    stats = key_collision_statistics(torch.randn(2, 1, 8))

    assert stats.total_pairs == 0
    assert stats.collision_counts.tolist() == [0, 0]
    assert stats.max_similarity.tolist() == [-1.0, -1.0]


def test_key_collision_rejects_bad_shapes_and_threshold():
    with pytest.raises(ValueError, match="batch, L, d_h"):
        key_collision_statistics(torch.randn(4, 8))
    with pytest.raises(ValueError, match="threshold"):
        key_collision_statistics(torch.randn(1, 2, 4), threshold=2.0)


# ---------------------------------------------------------------------------
# replay_state_trajectory
# ---------------------------------------------------------------------------


def test_replay_matches_precision_gated_final_state():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    states = replay_state_trajectory(
        keys, values, gates["alpha"], result.trace["gain"]
    )

    assert states.shape == (2, 8, 16, 12)
    assert torch.allclose(states[:, -1], result.final_state, atol=1e-5)


def test_replay_matches_kda_core_final_state():
    queries, keys, values, gates = _synthetic_inputs()
    core = KDACore(head_dim=16, value_dim=12)
    _, final_state = core(
        queries=queries,
        keys=keys,
        values=values,
        alpha=gates["alpha"],
        beta=gates["beta"],
    )

    write_vectors = gates["beta"] * keys  # w_t = beta_t * k_t
    states = replay_state_trajectory(keys, values, gates["alpha"], write_vectors)

    assert torch.allclose(states[:, -1], final_state, atol=1e-5)


def test_replay_rejects_mismatched_shapes():
    keys = torch.randn(2, 4, 8)
    values = torch.randn(2, 4, 6)
    alpha = torch.randn(2, 4, 8)
    write = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match="matching keys"):
        replay_state_trajectory(keys, values[:1], alpha, write)
    with pytest.raises(ValueError, match="last dim"):
        replay_state_trajectory(keys, values, alpha[..., :4], write)
    with pytest.raises(ValueError, match="initial_state"):
        replay_state_trajectory(
            keys, values, alpha, write, initial_state=torch.zeros(2, 8, 7)
        )


# ---------------------------------------------------------------------------
# summarize_trajectory
# ---------------------------------------------------------------------------


def test_summarize_trajectory_full_report_is_json_serializable():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)
    states = replay_state_trajectory(
        keys, values, gates["alpha"], result.trace["gain"]
    )

    report = summarize_trajectory(states=states, trace=result.trace, keys=keys)

    assert set(report) == {"state", "gates", "prediction_error", "key_collisions"}
    # Precision-gated traces carry beta only as a pass-through (no beta_eff),
    # so it is summarized under "beta_raw" for cross-variant comparability.
    assert set(report["gates"]) == {"alpha", "beta_raw", "gain"}
    assert report["state"]["effective_rank_final"] > 0
    assert report["prediction_error"]["final"] >= 0
    assert report["key_collisions"]["total_pairs"] == 8 * 7 // 2
    # JSON-serializable: plain floats/ints only.
    json.dumps(report)


def test_summarize_trajectory_fallback_trace_uses_beta_raw():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    report = summarize_trajectory(trace=result.trace)

    # RLS-lite exposes beta_eff; the raw beta must not be duplicated.
    assert "beta_eff" in report["gates"]
    assert "beta_raw" not in report["gates"]
    # Fallback traces (None placeholders) are skipped silently.
    assert "prediction_error" not in report


def test_summarize_trajectory_rejects_unbatched_states():
    with pytest.raises(ValueError, match="batch, L, d_h, d_v"):
        summarize_trajectory(states=torch.randn(8, 16, 12))
