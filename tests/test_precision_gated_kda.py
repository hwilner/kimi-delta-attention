"""Tests for the diagonal precision-gated KDA variant (issue #33)."""

import pytest
import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.variants import available_variants, create_variant
from kda.variants.precision_gated_kda import PrecisionGatedKDAVariant


def _synthetic_inputs(batch_size=2, seq_len=8, head_dim=16, value_dim=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1)
    keys = F.normalize(torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1)
    values = torch.randn(batch_size, seq_len, value_dim, generator=generator)
    gates = {
        "alpha": torch.sigmoid(torch.randn(batch_size, seq_len, head_dim, generator=generator)),
        "beta": torch.sigmoid(torch.randn(batch_size, seq_len, 1, generator=generator)),
    }
    return queries, keys, values, gates


def test_precision_gated_is_registered():
    assert "precision-gated-kda" in available_variants()


def test_output_shapes_and_trace():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=12)

    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert result.outputs.shape == (2, 8, 12)
    assert result.final_state.shape == (2, 16, 12)
    # Precision/gain/error trajectories are exposed for diagnostics (#27).
    assert result.trace["precision"].shape == (2, 8, 16)
    assert result.trace["gain"].shape == (2, 8, 16)
    assert result.trace["prediction_error"].shape == (2, 8, 12)
    assert result.trace["alpha"].shape == (2, 8, 16)
    assert (result.trace["precision"] > 0).all()


def test_disabled_variant_matches_kda_baseline():
    """With precision gating off, the variant must reproduce KDA numerically."""
    queries, keys, values, gates = _synthetic_inputs()

    variant = create_variant(
        "precision-gated-kda", head_dim=16, value_dim=12, precision_enabled=False
    )
    result = variant(queries, keys, values, gates=gates)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries, keys, values, gates["alpha"], gates["beta"]
    )

    assert torch.allclose(result.outputs, expected_outputs, atol=1e-6)
    assert torch.allclose(result.final_state, expected_state, atol=1e-6)


def test_single_step_recurrence_matches_manual_computation():
    """One hand-computed step: time update, gain, rank-1 write, downdate."""
    head_dim, value_dim = 3, 2
    k = F.normalize(torch.tensor([[1.0, 1.0, 1.0]]), dim=-1).view(1, 1, head_dim)
    q = k.clone()
    v = torch.tensor([[[2.0, -1.0]]])
    alpha = torch.full((1, 1, head_dim), 0.5)
    initial_state = torch.arange(6, dtype=torch.float32).view(1, 3, 2) / 10.0

    variant = create_variant(
        "precision-gated-kda",
        head_dim=head_dim,
        value_dim=value_dim,
        precision_init=1.0,
        lambda_=1.0,
        forgetting=1.0,
        process_noise=0.0,
    )
    result = variant(
        q,
        k,
        v,
        gates={"alpha": alpha},
        initial_state=initial_state,
        return_trace=True,
    )

    # Manual computation.
    k_flat = k.view(head_dim)
    precision_prior = torch.full((head_dim,), 1.0)  # P^-_1 = P_0/1 + 0
    denom = 1.0 + (precision_prior * k_flat * k_flat).sum() + variant.eps
    gain = precision_prior * k_flat / denom
    state_bar = 0.5 * initial_state[0]
    error = v.view(value_dim) - state_bar.T @ k_flat
    state = state_bar + torch.outer(gain, error)
    precision = precision_prior * (1.0 - gain * k_flat)
    expected_output = state.T @ q.view(head_dim)

    assert torch.allclose(result.final_state[0], state, atol=1e-6)
    assert torch.allclose(result.outputs[0, 0], expected_output, atol=1e-6)
    assert torch.allclose(result.trace["gain"][0, 0], gain, atol=1e-6)
    assert torch.allclose(result.trace["precision"][0, 0], precision, atol=1e-6)
    assert torch.allclose(result.trace["prediction_error"][0, 0], error, atol=1e-6)


def test_precision_is_consumed_along_written_direction():
    """Without forgetting/noise, the downdate shrinks precision where |k_c| > 0."""
    head_dim, value_dim = 8, 4
    seq_len = 3
    key = F.normalize(torch.tensor([1.0, 2.0, 0.5, 0.0, 1.0, 0.0, 0.25, 0.75]), dim=0)
    queries = key.view(1, 1, head_dim).repeat(1, seq_len, 1)
    keys = queries.clone()
    values = torch.randn(1, seq_len, value_dim)
    gates = {"alpha": torch.ones(1, seq_len, head_dim)}

    variant = create_variant(
        "precision-gated-kda",
        head_dim=head_dim,
        value_dim=value_dim,
        precision_init=1.0,
        forgetting=1.0,
        process_noise=0.0,
    )
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    precision = result.trace["precision"].squeeze(0)  # (L, d_h)
    active = key.abs() > 0
    # Channels touched by the key lose precision monotonically...
    assert (precision[1:, active] < precision[:-1, active]).all()
    # ...while untouched channels keep their precision.
    assert torch.allclose(
        precision[:, ~active], torch.ones_like(precision[:, ~active])
    )


def test_repeated_writes_reduce_prediction_error():
    """Rewriting the same association must drive the delta-rule error down."""
    head_dim, value_dim = 8, 4
    seq_len = 6
    key = F.normalize(torch.randn(1, 1, head_dim, generator=torch.Generator().manual_seed(7)), dim=-1)
    value = torch.randn(1, 1, value_dim, generator=torch.Generator().manual_seed(8))
    queries = key.repeat(1, seq_len, 1)
    keys = key.repeat(1, seq_len, 1)
    values = value.repeat(1, seq_len, 1)
    gates = {"alpha": torch.ones(1, seq_len, head_dim)}

    variant = create_variant("precision-gated-kda", head_dim=head_dim, value_dim=value_dim)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    error_norm = result.trace["prediction_error"].squeeze(0).norm(dim=-1)  # (L,)
    # e_{t+1} = (1 - kᵀ g_t) e_t with 0 < kᵀ g_t < 1: strictly decreasing.
    assert (error_norm[1:] < error_norm[:-1]).all()
    # And the association is nearly learned by the end of the sequence.
    assert error_norm[-1] < 0.2 * error_norm[0]


def test_precision_stays_within_clamps():
    """Aggressive forgetting + process noise must not escape [p_min, p_max]."""
    queries, keys, values, gates = _synthetic_inputs(seq_len=12)

    variant = create_variant(
        "precision-gated-kda",
        head_dim=16,
        value_dim=12,
        precision_init=1.0,
        forgetting=0.5,
        process_noise=0.25,
        p_min=0.05,
        p_max=4.0,
    )
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    precision = result.trace["precision"]
    assert (precision >= 0.05 - 1e-6).all()
    assert (precision <= 4.0 + 1e-6).all()
    assert torch.isfinite(precision).all()


def test_forgetting_and_process_noise_inflate_precision():
    """With lambda_ huge the gain vanishes, so P_t ≈ P_{t-1}/γ + q."""
    head_dim, value_dim = 4, 2
    seq_len = 4
    queries = F.normalize(torch.randn(1, seq_len, head_dim), dim=-1)
    keys = F.normalize(torch.randn(1, seq_len, head_dim), dim=-1)
    values = torch.randn(1, seq_len, value_dim)
    gates = {"alpha": torch.ones(1, seq_len, head_dim)}
    forgetting, process_noise = 0.5, 0.1

    variant = create_variant(
        "precision-gated-kda",
        head_dim=head_dim,
        value_dim=value_dim,
        precision_init=1.0,
        lambda_=1e9,  # gain ≈ 0 → downdate is a no-op
        forgetting=forgetting,
        process_noise=process_noise,
        p_max=1e9,
    )
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    expected = torch.ones(1, head_dim)
    for t in range(seq_len):
        expected = expected / forgetting + process_noise
        assert torch.allclose(
            result.trace["precision"][:, t, :], expected, rtol=1e-4, atol=1e-4
        )


def test_stable_under_random_synthetic_data():
    """Long random sequence: outputs, state, and precision stay finite."""
    queries, keys, values, gates = _synthetic_inputs(
        batch_size=3, seq_len=64, head_dim=32, value_dim=24, seed=123
    )

    variant = create_variant(
        "precision-gated-kda",
        head_dim=32,
        value_dim=24,
        forgetting=0.9,
        process_noise=0.01,
    )
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert torch.isfinite(result.outputs).all()
    assert torch.isfinite(result.final_state).all()
    assert torch.isfinite(result.trace["precision"]).all()
    assert torch.isfinite(result.trace["gain"]).all()


def test_parameter_validation():
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, forgetting=0.0)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, forgetting=1.5)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, lambda_=0.0)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, process_noise=-0.1)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, p_min=-1.0)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, p_min=2.0, p_max=1.0)
    with pytest.raises(ValueError):
        PrecisionGatedKDAVariant(head_dim=8, value_dim=8, precision_init=10.0, p_max=1.0)


def test_requires_alpha_gate_and_beta_for_fallback():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=12)

    with pytest.raises(ValueError):
        variant(queries, keys, values, gates=None)
    with pytest.raises(ValueError):
        variant(queries, keys, values, gates={"beta": gates["beta"]})

    # Enabled mode does not need beta; disabled fallback does.
    variant(queries, keys, values, gates={"alpha": gates["alpha"]})
    disabled = create_variant(
        "precision-gated-kda", head_dim=16, value_dim=12, precision_enabled=False
    )
    with pytest.raises(ValueError):
        disabled(queries, keys, values, gates={"alpha": gates["alpha"]})


def test_gradient_flows_through_gain_and_error():
    queries, keys, values, gates = _synthetic_inputs(seq_len=4)
    keys = keys.requires_grad_(True)
    values = values.requires_grad_(True)

    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates)
    result.outputs.sum().backward()

    assert keys.grad is not None
    assert values.grad is not None
    assert torch.isfinite(keys.grad).all()
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0
