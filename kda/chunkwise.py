"""Chunkwise Diagonal-Plus-Low-Rank (DPLR) parallel form of Kimi Delta Attention.

This module implements the chunkwise associative-scan formulation of the KDA
recurrence described in the architectural roadmap for this repository
(Phase 1: Systems & Numerical Modernization).

The sequential recurrence is

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

Each transition operator ``A_t = (I - beta_t k_t k_t^T) Diag(alpha_t)`` is a
*diagonal-plus-rank-1* (DPLR) matrix. Rather than evaluating the recurrence
token-by-token (a strictly memory-bandwidth-bound loop over HBM), sequences
are partitioned into non-overlapping chunks of length ``C`` (default 64).
Inside a chunk the recurrence is solved in closed form via a WY
representation:

1.  Absorb the diagonal decay into log-space cumulative products
    ``log Gamma_t = cumsum(log alpha)``. Every decay ratio that appears in
    the algorithm has the form ``Gamma_t / Gamma_s`` with ``s <= t`` and is
    therefore bounded in ``(0, 1]``; these are computed as
    ``exp(log Gamma_t - log Gamma_s)`` so no quantity ever overflows or
    underflows regardless of how aggressive the per-channel decay is.
2.  In decay-rescaled coordinates the recurrence is a plain delta rule whose
    intra-chunk write coefficients ``u_t`` satisfy the triangular system

        (I + tril(A, -1)) U = beta * (V - K_bar S_0),
        A_{ts} = beta_t * k_t^T (Gamma_t / Gamma_s) k_s   (s < t),

    solved with one batched triangular inversion per chunk.
3.  Intra-chunk outputs and the inter-chunk state hand-off are pure matmuls:

        O     = (Gamma Q) S_0 + tril(B) U,
        B_{ts} = q_t^T (Gamma_t / Gamma_s) k_s            (s <= t),
        S_end = Gamma_C S_0 + [k_s * (Gamma_C / Gamma_s)]^T U.

All steps are exact algebraic rearrangements of the sequential recurrence, so
outputs match ``KDACore`` to floating-point associativity error (~1e-6 in
float32). This pure-PyTorch reference is the correctness oracle for the fused
Triton kernel tracked on the project roadmap; it parallelizes all intra-chunk
work into batched matmuls and one ``C x C`` triangular solve per chunk.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KDADPLRChunkwise", "chunkwise_dplr_forward"]

_LOG_ALPHA_FLOOR = 1e-12


def _invert_unit_lower_triangular(matrix: torch.Tensor) -> torch.Tensor:
    """Invert a batched unit lower-triangular matrix by forward substitution.

    Args:
        matrix: Tensor of shape ``(B, C, C)`` with unit diagonal and zeros
            above the diagonal.

    Returns:
        Tensor of shape ``(B, C, C)``, the inverse of ``matrix``.
    """
    batch, size, _ = matrix.shape
    inverse = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    inverse = inverse.unsqueeze(0).expand(batch, size, size).clone()
    for row in range(1, size):
        # inv[row] = e_row - sum_{s<row} M[row, s] * inv[s]
        correction = torch.einsum("bs,bsc->bc", matrix[:, row, :row], inverse[:, :row])
        inverse[:, row] = inverse[:, row] - correction
    return inverse


def chunkwise_dplr_forward(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the KDA recurrence in chunkwise DPLR parallel form.

    Args:
        queries: Query vectors of shape ``(B, L, d_h)`` (L2-normalized).
        keys: Key vectors of shape ``(B, L, d_h)`` (L2-normalized).
        values: Value vectors of shape ``(B, L, d_v)``.
        alpha: Per-channel decay of shape ``(B, L, d_h)``, entries in (0, 1].
        beta: Scalar write gate of shape ``(B, L, 1)``, entries in [0, 1].
        initial_state: Optional initial state ``(B, d_h, d_v)``.
        chunk_size: Chunk length ``C``; 64 matches the roadmap target.

    Returns:
        A tuple ``(outputs, final_state)`` with shapes ``(B, L, d_v)`` and
        ``(B, d_h, d_v)``, algebraically identical to the sequential
        ``KDACore`` evaluation.
    """
    batch, seq_len, head_dim = queries.shape
    value_dim = values.shape[-1]
    device, dtype = queries.device, queries.dtype

    if initial_state is None:
        state = torch.zeros(batch, head_dim, value_dim, device=device, dtype=dtype)
    else:
        state = initial_state

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - seq_len
    if pad_len > 0:
        # Padded positions use alpha=1, beta=0 so they are exact no-ops.
        queries = F.pad(queries, (0, 0, 0, pad_len))
        keys = F.pad(keys, (0, 0, 0, pad_len))
        values = F.pad(values, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, pad_len), value=1.0)
        beta = F.pad(beta, (0, 0, 0, pad_len), value=0.0)

    q = queries.view(batch, num_chunks, chunk_size, head_dim)
    k = keys.view(batch, num_chunks, chunk_size, head_dim)
    v = values.view(batch, num_chunks, chunk_size, value_dim)
    a = alpha.view(batch, num_chunks, chunk_size, head_dim)
    b = beta.view(batch, num_chunks, chunk_size, 1)

    outputs = torch.empty(
        batch, num_chunks, chunk_size, value_dim, device=device, dtype=dtype
    )

    for c in range(num_chunks):
        qc, kc, vc, ac, bc = q[:, c], k[:, c], v[:, c], a[:, c], b[:, c]

        # Log-space cumulative decay: log Gamma_t (B, C, d_h).
        log_gamma = torch.cumsum(torch.log(ac.clamp_min(_LOG_ALPHA_FLOOR)), dim=1)

        # Decay-ratio tensor E[t, s] = Gamma_t / Gamma_s (entries <= 1 for
        # s <= t; entries for s > t are masked out by tril below).
        ratio = torch.exp(log_gamma.unsqueeze(2) - log_gamma.unsqueeze(1))

        # Triangular coupling A_{ts} = beta_t k_t^T (Gamma_t/Gamma_s) k_s.
        coupling = bc * torch.einsum("bth,bsh,btsh->bts", kc, kc, ratio)
        unit_lower = torch.eye(chunk_size, device=device, dtype=dtype) + torch.tril(
            coupling, diagonal=-1
        )
        transform = _invert_unit_lower_triangular(unit_lower)  # (B, C, C)

        # Right-hand side: beta * (V - K_bar S_0), K_bar = Gamma k.
        k_bar = torch.exp(log_gamma) * kc
        rhs = bc * (vc - torch.einsum("bch,bhv->bcv", k_bar, state))
        writes = torch.einsum("bts,bsv->btv", transform, rhs)  # U (B, C, d_v)

        # Intra-chunk outputs: O = (Gamma Q) S_0 + tril(B) U with
        # B_{ts} = q_t^T (Gamma_t/Gamma_s) k_s for s <= t.
        intra = torch.tril(torch.einsum("bth,bsh,btsh->bts", qc, kc, ratio))
        q_tilde = torch.exp(log_gamma) * qc
        outputs[:, c] = torch.einsum("bch,bhv->bcv", q_tilde, state) + torch.einsum(
            "bts,bsv->btv", intra, writes
        )

        # Inter-chunk hand-off:
        # S_end = Gamma_C S_0 + [k_s * (Gamma_C/Gamma_s)]^T U.
        log_gamma_end = log_gamma[:, -1:]  # (B, 1, d_h)
        end_rescaled_keys = kc * torch.exp(log_gamma_end - log_gamma)
        state = torch.exp(log_gamma_end.squeeze(1)).unsqueeze(-1) * state
        state = state + torch.einsum("bch,bcv->bhv", end_rescaled_keys, writes)

    outputs = outputs.view(batch, num_chunks * chunk_size, value_dim)
    if pad_len > 0:
        outputs = outputs[:, :-pad_len]
    return outputs, state


class KDADPLRChunkwise(nn.Module):
    """Chunkwise DPLR KDA layer with the same interface as ``KDACore``.

    A drop-in parallel replacement for the sequential recurrence: identical
    inputs, identical outputs (up to float associativity), but the intra-chunk
    computation is expressed as batched matrix multiplications and one
    ``C x C`` triangular solve per chunk instead of ``C`` sequential state
    materializations.

    Args:
        head_dim: Dimension of queries/keys (``d_h``).
        value_dim: Dimension of values (``d_v``).
        chunk_size: Chunk length ``C`` (default 64).
    """

    def __init__(self, head_dim: int, value_dim: int, chunk_size: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.chunk_size = chunk_size

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the chunkwise DPLR forward pass.

        Args:
            queries: Query vectors ``(B, L, d_h)``.
            keys: Key vectors ``(B, L, d_h)``.
            values: Value vectors ``(B, L, d_v)``.
            alpha: Per-channel decay ``(B, L, d_h)``.
            beta: Scalar write gate ``(B, L, 1)``.
            initial_state: Optional initial state ``(B, d_h, d_v)``.

        Returns:
            Tuple of outputs ``(B, L, d_v)`` and final state ``(B, d_h, d_v)``.
        """
        return chunkwise_dplr_forward(
            queries,
            keys,
            values,
            alpha,
            beta,
            initial_state=initial_state,
            chunk_size=self.chunk_size,
        )
