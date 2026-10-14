"""Gated DeltaNet-2 reference baseline (Phase 4, issue #37).

This module is a **reference baseline**, not a novel contribution of this
repository. It implements the published Gated Delta Rule-2 recurrence from

    Hatamizadeh, Choi, Kautz. "Gated DeltaNet-2: Decoupling Erase and Write
    in Linear Attention." arXiv:2605.22791 (2026).

GDN-2 keeps KDA's channel-wise decay ``Diag(α_t)`` but replaces the single
scalar delta gate ``β_t`` (which ties erasing and writing together in both
Gated DeltaNet and KDA) with two channel-wise gates:

    e_t = b_t ⊙ k_t          (key-side gated erase direction)
    z_t = w_t ⊙ v_t          (value-side gated write target)
    S_t = (I − k_t e_tᵀ) Diag(α_t) S_{t−1} + k_t z_tᵀ
    o_t = S_tᵀ q_t

where ``b_t ∈ [0,1]^{d_h}`` is the erase gate on the key axis and
``w_t ∈ [0,1]^{d_v}`` is the write gate on the value axis. Tying
``b_t = β_t·1`` and ``w_t = β_t·1`` recovers the KDA update exactly; tying
``α_t`` to a scalar as well recovers Gated DeltaNet (the reductions stated in
the paper's Appendix A.5).

Approximations relative to the paper (documented per issue #37 acceptance
criteria):

- Sequential reference recurrence only. The paper's chunkwise WY algorithm
  and fused kernels are *not* implemented; the sequential form above is the
  defining recurrence those algorithms compute. No production-performance
  claims are made or implied.
- Gate-generation machinery from the paper (sigmoid projections for the
  gates, the log-space decay parameterization) is out of scope: like every
  variant in this repository, gates are supplied externally through the
  shared ``gates`` mapping, so the baseline plugs into the same
  task/evaluation path as KDA.

Interface notes:

- ``gates["alpha"]`` (``batch, seq_len, head_dim``) is required.
- ``gates["erase"]`` (``batch, seq_len, head_dim``) and ``gates["write"]``
  (``batch, seq_len, value_dim``) supply the channel-wise gates. When both
  are absent, ``gates["beta"]`` (``batch, seq_len, 1``) is broadcast to both
  axes, which is exactly the KDA tie — so the baseline runs unchanged on the
  shared alpha/beta gate path. Supplying exactly one of ``erase``/``write``
  is rejected as ambiguous.
- Applied gate trajectories are exposed via ``return_trace=True`` for the
  diagnostics utilities of issue #27.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from .base import DeltaMemoryOutput, DeltaMemoryVariant, register_variant


@register_variant("gdn-2")
class GatedDeltaNet2Variant(DeltaMemoryVariant):
    """Reference implementation of Gated DeltaNet-2 (arXiv:2605.22791).

    Args:
        head_dim: Dimension of queries/keys (``d_h``).
        value_dim: Dimension of values (``d_v``).
        eps: Small constant for numerical stability (kept for interface
            symmetry with the other variants; the recurrence itself does not
            use it).
    """

    def __init__(self, head_dim: int, value_dim: int, eps: float = 1e-6) -> None:
        super().__init__(head_dim=head_dim, value_dim=value_dim)
        self.eps = eps

    def _resolve_gates(
        self,
        gates: Optional[Mapping[str, torch.Tensor]],
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return ``(alpha, erase_gate, write_gate, tied_from_beta)``."""
        if gates is None or "alpha" not in gates:
            raise ValueError(
                "GatedDeltaNet2Variant requires gates containing 'alpha'"
            )
        alpha = gates["alpha"]
        if alpha.shape != (batch_size, seq_len, self.head_dim):
            raise ValueError(
                f"alpha must have shape {(batch_size, seq_len, self.head_dim)}, "
                f"got {tuple(alpha.shape)}"
            )

        has_erase = "erase" in gates
        has_write = "write" in gates
        if has_erase != has_write:
            raise ValueError(
                "gates must supply both 'erase' and 'write' together, or "
                "neither (in which case 'beta' is broadcast to both axes)"
            )

        if has_erase:
            erase = gates["erase"]
            write = gates["write"]
            if erase.shape != (batch_size, seq_len, self.head_dim):
                raise ValueError(
                    f"erase gate must have shape "
                    f"{(batch_size, seq_len, self.head_dim)}, "
                    f"got {tuple(erase.shape)}"
                )
            if write.shape != (batch_size, seq_len, self.value_dim):
                raise ValueError(
                    f"write gate must have shape "
                    f"{(batch_size, seq_len, self.value_dim)}, "
                    f"got {tuple(write.shape)}"
                )
            return alpha, erase, write, False

        if "beta" not in gates:
            raise ValueError(
                "GatedDeltaNet2Variant requires gates containing 'beta' when "
                "channel-wise 'erase'/'write' gates are not supplied"
            )
        beta = gates["beta"]
        if beta.shape != (batch_size, seq_len, 1):
            raise ValueError(
                f"beta must have shape {(batch_size, seq_len, 1)}, "
                f"got {tuple(beta.shape)}"
            )
        # KDA tie: the same scalar gate drives both axes.
        erase = beta.expand(batch_size, seq_len, self.head_dim)
        write = beta.expand(batch_size, seq_len, self.value_dim)
        return alpha, erase, write, True

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: Optional[Mapping[str, torch.Tensor]] = None,
        initial_state: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> DeltaMemoryOutput:
        self._validate_inputs(queries, keys, values)
        batch_size, seq_len, _ = queries.shape
        alpha, erase, write, tied_from_beta = self._resolve_gates(
            gates, batch_size, seq_len
        )

        if initial_state is None:
            state = torch.zeros(
                batch_size,
                self.head_dim,
                self.value_dim,
                device=queries.device,
                dtype=queries.dtype,
            )
        else:
            state = initial_state

        outputs = []
        erase_steps = []
        write_steps = []

        # Gated Delta Rule-2 (arXiv:2605.22791, Eq. 29), applied step by step.
        for t in range(seq_len):
            q_t = queries[:, t, :]          # (B, d_h)
            k_t = keys[:, t, :]             # (B, d_h)
            v_t = values[:, t, :]           # (B, d_v)
            alpha_t = alpha[:, t, :]        # (B, d_h)
            b_t = erase[:, t, :]            # (B, d_h)  key-side erase gate
            w_t = write[:, t, :]            # (B, d_v)  value-side write gate

            # Decay first: S_bar = Diag(alpha_t) S_{t-1}
            state = alpha_t.unsqueeze(-1) * state

            # Erase: subtract k_t (b_t ⊙ k_t)ᵀ S_bar
            e_t = b_t * k_t                                       # (B, d_h)
            erased = torch.bmm(e_t.unsqueeze(1), state)           # (B, 1, d_v)
            state = state - torch.bmm(k_t.unsqueeze(-1), erased)

            # Write: add k_t (w_t ⊙ v_t)ᵀ
            z_t = w_t * v_t                                       # (B, d_v)
            state = state + torch.bmm(k_t.unsqueeze(-1), z_t.unsqueeze(1))

            # Readout: o_t = S_tᵀ q_t
            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)

            if return_trace:
                erase_steps.append(b_t.clone() if tied_from_beta else b_t)
                write_steps.append(w_t.clone() if tied_from_beta else w_t)

        outputs = torch.stack(outputs, dim=1)  # (B, L, d_v)

        trace = {}
        if return_trace:
            trace = {
                "alpha": alpha,
                # (B, L, d_h): erase gate applied on the key axis.
                "erase_gate": torch.stack(erase_steps, dim=1),
                # (B, L, d_v): write gate applied on the value axis.
                "write_gate": torch.stack(write_steps, dim=1),
                # True when the channel-wise gates were tied from beta.
                "tied_from_beta": tied_from_beta,
            }

        return DeltaMemoryOutput(outputs=outputs, final_state=state, trace=trace)
