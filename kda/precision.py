"""Low-precision state representations for long-horizon KDA recurrence.

Implements the numerical-stabilization utilities specified in Phase 1 of the
repository roadmap: block-scaled FP8 state caching and stochastic rounding.

Over context windows exceeding ~32k tokens, repeated multiplications by the
delta-rule transition ``(I - beta_t k_t k_t^T)`` accumulate rounding error in
16-bit half precision, causing catastrophic cancellation in the recurrent
state. Two complementary mitigations are provided:

* :func:`stochastic_round` -- unbiased rounding to a target mantissa width,
  which converts systematic rounding drift into zero-mean noise and prevents
  correlated error growth across steps.
* :class:`BlockScaledFP8State` -- an FP8 (e4m3) state cache with per-block
  power-of-two scaling, giving FP16-comparable dynamic range at half the
  memory footprint.

These are pure-PyTorch *emulations* (quantized values are stored dequantized
in float32) intended as correctness oracles and CPU fallbacks for fused
kernels, mirroring the educational charter of this repository.
"""

from typing import Tuple

import torch

__all__ = ["stochastic_round", "BlockScaledFP8State"]

# e4m3 format: 4 exponent bits, 3 mantissa bits, max normal value 448.
_FP8_E4M3_MAX = 448.0
_FP8_MANTISSA_BITS = 3
_FP8_EXPONENT_BIAS = 7


def stochastic_round(x: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    """Round a tensor to a reduced mantissa width with stochastic rounding.

    For each element, the two nearest representable values at the target
    precision are computed and one is chosen with probability proportional to
    proximity, making the rounding error zero-mean in expectation.

    Args:
        x: Input tensor (float32 recommended).
        mantissa_bits: Number of explicit mantissa bits of the target format
            (e.g. 3 for e4m3 FP8, 10 for FP16).

    Returns:
        Tensor of the same shape and dtype, with values rounded stochastically
        to the target mantissa width.
    """
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("stochastic_round expects float32/float64 input")

    abs_x = x.abs()
    zero_mask = abs_x == 0
    abs_safe = torch.where(zero_mask, torch.ones_like(abs_x), abs_x)

    # Exponent of the nearest power of two at or below |x|.
    exponent = torch.floor(torch.log2(abs_safe))
    # ULP at the target precision for each element.
    ulp = torch.pow(torch.tensor(2.0, dtype=x.dtype, device=x.device),
                    exponent - mantissa_bits)

    lower = torch.floor(abs_safe / ulp) * ulp
    upper = lower + ulp
    prob_upper = (abs_safe - lower) / ulp
    draw = torch.rand_like(prob_upper)
    rounded = torch.where(draw < prob_upper, upper, lower)

    rounded = torch.where(zero_mask, torch.zeros_like(rounded), rounded)
    return torch.copysign(rounded, x)


class BlockScaledFP8State:
    """Block-scaled FP8 (e4m3) cache for the KDA recurrent state matrix.

    The state ``S in R^{d_h x d_v}`` is partitioned into square blocks; each
    block is quantized to e4m3 with an independent power-of-two scale, keeping
    quantization error proportional to the local magnitude instead of the
    global state norm.

    Attributes:
        block_size: Edge length of each scaling block.
        stochastic: Whether to use stochastic rounding during quantization.
    """

    def __init__(self, block_size: int = 32, stochastic: bool = True):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.stochastic = stochastic

    def quantize(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a state matrix to block-scaled FP8.

        Args:
            state: State tensor ``(..., d_h, d_v)`` in float32.

        Returns:
            A tuple ``(codes, scales)`` where ``codes`` holds dequantized FP8
            values in float32 and ``scales`` has one power-of-two scale per
            block. ``dequantize(*quantize(x))`` reconstructs the quantized x.
        """
        if state.dtype != torch.float32:
            raise TypeError("BlockScaledFP8State expects float32 input")

        *batch_dims, rows, cols = state.shape
        pad_r = (-rows) % self.block_size
        pad_c = (-cols) % self.block_size
        padded = torch.nn.functional.pad(state, (0, pad_c, 0, pad_r))

        blocks = padded.reshape(
            *batch_dims,
            padded.shape[-2] // self.block_size, self.block_size,
            padded.shape[-1] // self.block_size, self.block_size,
        )
        block_max = blocks.abs().amax(dim=(-3, -1), keepdim=True)
        safe_max = block_max.clamp_min(1e-12)

        # Power-of-two scale so the block max maps inside e4m3 range. Ceil
        # (not floor) guarantees normalized values never exceed 448.
        scales = torch.pow(
            2.0, torch.ceil(torch.log2(safe_max / _FP8_E4M3_MAX))
        )
        normalized = blocks / scales

        # Emulate e4m3 mantissa rounding.
        if self.stochastic:
            quantized = stochastic_round(normalized, _FP8_MANTISSA_BITS)
        else:
            ulp_scale = torch.pow(
                torch.tensor(2.0, device=state.device),
                torch.floor(torch.log2(normalized.abs().clamp_min(1e-12)))
                - _FP8_MANTISSA_BITS,
            )
            quantized = torch.round(normalized / ulp_scale) * ulp_scale
        quantized = quantized.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)

        codes = (quantized * scales).reshape(padded.shape)
        if pad_r or pad_c:
            codes = codes[..., :rows, :cols]
        return codes, scales

    def dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Reconstruct a state matrix from block-scaled FP8 codes.

        Args:
            codes: Tensor returned by :meth:`quantize` (already dequantized
                float32 values in this emulation).
            scales: Per-block scales returned by :meth:`quantize` (kept for
                interface compatibility with fused kernels).

        Returns:
            The reconstructed state tensor (identical to ``codes`` in this
            reference implementation).
        """
        return codes
