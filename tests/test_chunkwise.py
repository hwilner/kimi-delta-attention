"""Tests for the chunkwise DPLR formulation and low-precision state utilities.

All tests use synthetic random data, consistent with the repository testing
charter. The chunkwise implementation is validated for exact numerical parity
against the sequential ``KDACore`` recurrence (the ground truth), across chunk
boundaries, non-multiple sequence lengths, and gradient flow.
"""

import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.chunkwise import KDADPLRChunkwise, chunkwise_dplr_forward
from kda.precision import BlockScaledFP8State, stochastic_round


def _make_inputs(batch, seq_len, head_dim, value_dim, seed):
    """Generate a synthetic KDA input batch.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        head_dim: Query/key dimension.
        value_dim: Value dimension.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (queries, keys, values, alpha, beta) tensors.
    """
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(batch, seq_len, head_dim, generator=generator), dim=-1)
    keys = F.normalize(torch.randn(batch, seq_len, head_dim, generator=generator), dim=-1)
    values = torch.randn(batch, seq_len, value_dim, generator=generator)
    # Keep alpha away from 0 so cumulative products stay well-conditioned
    # in the float32 reference implementation.
    alpha = 0.5 + 0.5 * torch.sigmoid(
        torch.randn(batch, seq_len, head_dim, generator=generator)
    )
    beta = torch.sigmoid(torch.randn(batch, seq_len, 1, generator=generator))
    return queries, keys, values, alpha, beta


class TestChunkwiseParity:
    """Numerical parity between chunkwise DPLR and sequential KDACore."""

    def _check_parity(self, batch, seq_len, head_dim, value_dim, chunk_size, seed):
        queries, keys, values, alpha, beta = _make_inputs(
            batch, seq_len, head_dim, value_dim, seed
        )
        sequential = KDACore(head_dim, value_dim)
        parallel = KDADPLRChunkwise(head_dim, value_dim, chunk_size=chunk_size)

        out_seq, state_seq = sequential(queries, keys, values, alpha, beta)
        out_par, state_par = parallel(queries, keys, values, alpha, beta)

        assert out_par.shape == out_seq.shape
        assert state_par.shape == state_seq.shape
        assert torch.allclose(out_par, out_seq, atol=1e-4, rtol=1e-4), (
            f"max output deviation: {(out_par - out_seq).abs().max().item():.2e}"
        )
        assert torch.allclose(state_par, state_seq, atol=1e-4, rtol=1e-4), (
            f"max state deviation: {(state_par - state_seq).abs().max().item():.2e}"
        )

    def test_single_chunk(self):
        """Parity when the sequence fits in exactly one chunk."""
        self._check_parity(2, 64, 16, 16, chunk_size=64, seed=0)

    def test_multiple_chunks(self):
        """Parity across several chunk boundaries (inter-chunk hand-off)."""
        self._check_parity(2, 192, 16, 16, chunk_size=64, seed=1)

    def test_ragged_sequence_length(self):
        """Parity when sequence length is not a multiple of the chunk size."""
        self._check_parity(2, 100, 16, 16, chunk_size=64, seed=2)

    def test_sub_chunk_sequence(self):
        """Parity for sequences shorter than one chunk."""
        self._check_parity(2, 17, 16, 16, chunk_size=64, seed=3)

    def test_unequal_head_and_value_dims(self):
        """Parity with rectangular state matrices (d_h != d_v)."""
        self._check_parity(2, 128, 32, 16, chunk_size=32, seed=4)

    def test_initial_state_handoff(self):
        """Parity with a non-zero initial state (streaming inference)."""
        batch, seq_len, head_dim, value_dim = 2, 128, 16, 16
        queries, keys, values, alpha, beta = _make_inputs(
            batch, seq_len, head_dim, value_dim, seed=5
        )
        initial = torch.randn(batch, head_dim, value_dim)
        out_seq, state_seq = KDACore(head_dim, value_dim)(
            queries, keys, values, alpha, beta, initial_state=initial
        )
        out_par, state_par = chunkwise_dplr_forward(
            queries, keys, values, alpha, beta, initial_state=initial, chunk_size=64
        )
        assert torch.allclose(out_par, out_seq, atol=1e-4, rtol=1e-4)
        assert torch.allclose(state_par, state_seq, atol=1e-4, rtol=1e-4)

    def test_gradient_flow(self):
        """Chunkwise form supports backpropagation with matching gradients."""
        torch.manual_seed(6)
        batch, seq_len, head_dim, value_dim = 1, 96, 16, 16
        queries, keys, values, alpha, beta = _make_inputs(
            batch, seq_len, head_dim, value_dim, seed=6
        )
        q1 = queries.clone().requires_grad_(True)
        q2 = queries.clone().requires_grad_(True)

        out_seq, _ = KDACore(head_dim, value_dim)(q1, keys, values, alpha, beta)
        out_par, _ = chunkwise_dplr_forward(q2, keys, values, alpha, beta)
        out_seq.sum().backward()
        out_par.sum().backward()

        assert q1.grad is not None and q2.grad is not None
        assert torch.allclose(q1.grad, q2.grad, atol=1e-4, rtol=1e-4)


class TestStochasticRounding:
    """Property tests for stochastic rounding (Phase 1 numerics)."""

    def test_unbiased_in_expectation(self):
        """Stochastic rounding error averages to zero over many draws."""
        torch.manual_seed(0)
        x = torch.full((4096,), 1.3)
        draws = torch.stack(
            [stochastic_round(x, mantissa_bits=3) for _ in range(200)]
        )
        mean_estimate = draws.mean(dim=0)
        assert (mean_estimate - x).abs().max() < 0.02

    def test_representable_values_unchanged(self):
        """Exactly representable values round to themselves."""
        x = torch.tensor([1.0, 2.0, 0.5, 1.5, -1.25, 3.0])
        rounded = stochastic_round(x, mantissa_bits=3)
        assert torch.equal(rounded, x)

    def test_bounds(self):
        """Rounded values stay within one ULP of the input."""
        torch.manual_seed(1)
        x = torch.randn(1000).abs() + 0.1
        rounded = stochastic_round(x, mantissa_bits=3)
        ulp = torch.pow(
            torch.tensor(2.0), torch.floor(torch.log2(x)) - 3
        )
        assert (rounded - x).abs().max() <= ulp.max() + 1e-7


class TestBlockScaledFP8:
    """Tests for the block-scaled FP8 state cache."""

    def test_roundtrip_error_bounded(self):
        """Quantize-dequantize error is small relative to block magnitudes."""
        torch.manual_seed(2)
        state = torch.randn(2, 64, 64)
        cache = BlockScaledFP8State(block_size=32, stochastic=False)
        codes, scales = cache.quantize(state)
        recovered = cache.dequantize(codes, scales)
        rel_error = (recovered - state).abs().max() / state.abs().max()
        assert rel_error < 0.15  # e4m3 has ~2^-4 relative resolution

    def test_handles_non_multiple_dimensions(self):
        """Quantization pads states whose dims are not multiples of block size."""
        torch.manual_seed(3)
        state = torch.randn(1, 48, 40)
        cache = BlockScaledFP8State(block_size=32)
        codes, _ = cache.quantize(state)
        assert codes.shape == state.shape

    def test_rejects_half_precision_input(self):
        """Half-precision input is rejected to avoid silent double rounding."""
        cache = BlockScaledFP8State()
        try:
            cache.quantize(torch.randn(8, 8, dtype=torch.float16))
        except TypeError:
            return
        raise AssertionError("expected TypeError for float16 input")
