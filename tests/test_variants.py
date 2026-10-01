"""Tests for the shared delta-memory variant interface."""

import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.variants import (
    DeltaMemoryOutput,
    VariantNotFoundError,
    available_variants,
    create_variant,
)


def _synthetic_inputs(batch_size=2, seq_len=8, head_dim=16, value_dim=12):
    queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
    keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
    values = torch.randn(batch_size, seq_len, value_dim)
    gates = {
        "alpha": torch.sigmoid(torch.randn(batch_size, seq_len, head_dim)),
        "beta": torch.sigmoid(torch.randn(batch_size, seq_len, 1)),
    }
    return queries, keys, values, gates


def test_kda_variant_is_registered():
    assert "kda" in available_variants()


def test_kda_variant_matches_core_output():
    queries, keys, values, gates = _synthetic_inputs()

    variant = create_variant("kda", head_dim=16, value_dim=12)
    wrapped = variant(queries, keys, values, gates=gates, return_trace=True)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries,
        keys,
        values,
        gates["alpha"],
        gates["beta"],
    )

    assert isinstance(wrapped, DeltaMemoryOutput)
    assert torch.allclose(wrapped.outputs, expected_outputs)
    assert torch.allclose(wrapped.final_state, expected_state)
    assert "alpha" in wrapped.trace
    assert "beta" in wrapped.trace


def test_variant_supports_initial_state():
    queries, keys, values, gates = _synthetic_inputs(seq_len=4)
    initial_state = torch.randn(2, 16, 12)

    variant = create_variant("kda", head_dim=16, value_dim=12)
    result = variant(
        queries,
        keys,
        values,
        gates=gates,
        initial_state=initial_state,
    )

    assert result.outputs.shape == (2, 4, 12)
    assert result.final_state.shape == initial_state.shape


def test_unknown_variant_raises_helpful_error():
    try:
        create_variant("missing-variant", head_dim=8, value_dim=8)
    except VariantNotFoundError as exc:
        assert "Available" in str(exc)
    else:
        raise AssertionError("expected VariantNotFoundError")


def test_variant_validates_common_shapes():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("kda", head_dim=16, value_dim=12)

    try:
        variant(queries, keys, values[..., :4], gates=gates)
    except ValueError as exc:
        assert "value dim" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched value_dim")
