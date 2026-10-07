"""Tests for the KDA-RLS-lite evidence-modulated variant (issue #36)."""

import pytest
import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.variants import available_variants, create_variant
from kda.variants.kda_rls_lite import KDARLSLiteVariant


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


def test_rls_lite_is_registered():
    assert "kda-rls-lite" in available_variants()


def test_output_shapes_and_trace():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)

    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert result.outputs.shape == (2, 8, 12)
    assert result.final_state.shape == (2, 16, 12)
    # Gate/evidence trajectories are exposed for diagnostics (#27 consumer).
    assert result.trace["beta_eff"].shape == (2, 8, 1)
    assert result.trace["beta_raw"].shape == (2, 8, 1)
    assert result.trace["evidence"].shape == (2, 8, 16)
    assert result.trace["evidence_along_key"].shape == (2, 8, 1)
    assert (result.trace["evidence"] >= 0).all()


def test_disabled_variant_matches_kda_baseline():
    """With evidence gating off, KDA-RLS-lite must reproduce KDA numerically."""
    queries, keys, values, gates = _synthetic_inputs()

    variant = create_variant(
        "kda-rls-lite", head_dim=16, value_dim=12, evidence_enabled=False
    )
    result = variant(queries, keys, values, gates=gates)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries, keys, values, gates["alpha"], gates["beta"]
    )

    assert torch.allclose(result.outputs, expected_outputs, atol=1e-6)
    assert torch.allclose(result.final_state, expected_state, atol=1e-6)


def test_evidence_accumulator_matches_closed_form():
    """n_t = alpha ⊙ n_{t-1} + k_t^2 with rho=1, checked step by step."""
    queries, keys, values, gates = _synthetic_inputs(seq_len=5)
    # Pin alpha to a known value to make the closed form easy to verify.
    gates["alpha"] = torch.full_like(gates["alpha"], 0.5)

    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    expected = torch.zeros(2, 16)
    for t in range(5):
        k_t = keys[:, t, :]
        expected = 0.5 * expected + k_t * k_t
        assert torch.allclose(result.trace["evidence"][:, t, :], expected, atol=1e-6)


def test_beta_eff_respects_bounds():
    queries, keys, values, gates = _synthetic_inputs(seq_len=6)

    variant = create_variant(
        "kda-rls-lite",
        head_dim=16,
        value_dim=12,
        beta_min=0.1,
        beta_max=0.4,
    )
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    beta_eff = result.trace["beta_eff"]
    assert (beta_eff >= 0.1 - 1e-6).all()
    assert (beta_eff <= 0.4 + 1e-6).all()


def test_evidence_damps_but_never_amplifies_beta():
    """With lambda_=1 and non-negative evidence, beta_eff <= beta_raw."""
    queries, keys, values, gates = _synthetic_inputs()

    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert (result.trace["beta_eff"] <= result.trace["beta_raw"] + 1e-6).all()


def test_repeat_key_writes_increase_occupancy():
    """Writing the same key repeatedly must grow evidence along that key."""
    head_dim, value_dim = 8, 4
    key = F.normalize(torch.tensor([1.0, 2.0, 0.5, 0.0, 1.0, 0.0, 0.25, 0.75]), dim=0)
    seq_len = 4
    queries = key.view(1, 1, head_dim).repeat(1, seq_len, 1)
    keys = queries.clone()
    values = torch.randn(1, seq_len, value_dim)
    gates = {
        "alpha": torch.ones(1, seq_len, head_dim),
        "beta": torch.full((1, seq_len, 1), 0.9),
    }

    variant = create_variant("kda-rls-lite", head_dim=head_dim, value_dim=value_dim)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    occupancy = result.trace["evidence_along_key"].squeeze(0).squeeze(-1)
    # Occupancy along the repeated key grows monotonically...
    assert (occupancy[1:] > occupancy[:-1]).all()
    # ...and the effective write gate is progressively damped.
    beta_eff = result.trace["beta_eff"].squeeze(0).squeeze(-1)
    assert (beta_eff[1:] < beta_eff[:-1]).all()


def test_single_step_recurrence_matches_manual_computation():
    """One hand-computed step: decay, evidence-modulated erase, then write."""
    head_dim, value_dim = 3, 2
    k = F.normalize(torch.tensor([[1.0, 1.0, 1.0]]), dim=-1).view(1, 1, head_dim)
    q = k.clone()
    v = torch.tensor([[[2.0, -1.0]]])
    alpha = torch.full((1, 1, head_dim), 0.5)
    beta = torch.full((1, 1, 1), 0.8)
    initial_state = torch.arange(6, dtype=torch.float32).view(1, 3, 2) / 10.0

    variant = create_variant(
        "kda-rls-lite",
        head_dim=head_dim,
        value_dim=value_dim,
        lambda_=1.0,
        beta_min=0.0,
        beta_max=1.0,
    )
    result = variant(
        q,
        k,
        v,
        gates={"alpha": alpha, "beta": beta},
        initial_state=initial_state,
        return_trace=True,
    )

    # Manual computation.
    k_flat = k.view(head_dim)
    evidence = k_flat * k_flat  # n_1 = k^2 (initial evidence is zero)
    e = (evidence * k_flat * k_flat).sum()
    beta_eff = (0.8 / (1.0 + e + variant.eps)).clamp(0.0, 1.0)
    state = 0.5 * initial_state[0]
    state = state - beta_eff * torch.outer(k_flat, k_flat @ state)
    state = state + beta_eff * torch.outer(k_flat, v.view(value_dim))
    expected_output = state.T @ q.view(head_dim)

    assert torch.allclose(result.final_state[0], state, atol=1e-6)
    assert torch.allclose(result.outputs[0, 0], expected_output, atol=1e-6)
    assert torch.allclose(
        result.trace["beta_eff"][0, 0, 0], beta_eff, atol=1e-6
    )


def test_parameter_validation():
    with pytest.raises(ValueError):
        KDARLSLiteVariant(head_dim=8, value_dim=8, evidence_decay=0.0)
    with pytest.raises(ValueError):
        KDARLSLiteVariant(head_dim=8, value_dim=8, lambda_=-0.1)
    with pytest.raises(ValueError):
        KDARLSLiteVariant(head_dim=8, value_dim=8, beta_min=0.9, beta_max=0.1)


def test_requires_alpha_and_beta_gates():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)

    with pytest.raises(ValueError):
        variant(queries, keys, values, gates=None)
    with pytest.raises(ValueError):
        variant(queries, keys, values, gates={"alpha": gates["alpha"]})


def test_gradient_flows_through_beta_eff():
    queries, keys, values, gates = _synthetic_inputs(seq_len=4)
    gates["beta"] = gates["beta"].requires_grad_(True)

    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates)
    result.outputs.sum().backward()

    assert gates["beta"].grad is not None
    assert torch.isfinite(gates["beta"].grad).all()
    assert gates["beta"].grad.abs().sum() > 0
