"""Erase-then-Delta Attention (EDA) reference baseline (Phase 4, issue #37).

This module is a **reference baseline**, not a novel contribution of this
repository. It implements the published EDA recurrence from

    Li, Zhang, Luo, Lin, Wang, Qiu, Mao, Chen, Yuan, Sun, Jiang, Zhang, Men,
    Hu, Cheng, Zheng, Liu, Zhou. "Erase-then-Delta Attention: Decoupling
    Erase and Write Addresses in Delta-Rule Linear Attention."
    arXiv:2606.26560 (2026).

EDA targets *address-level* coupling: in GDN/KDA (and in GDN-2, which
decouples only gate strengths) the active delta correction is anchored to
the current write key ``k_t``, so stale memory stored at a different address
can only fade through address-agnostic decay. EDA inserts an independently
addressed erase operator before the standard delta write (Eq. 8 of the
paper, factors applied right to left):

    S_t = (I − β_t k_t k_tᵀ) (I − γ_t e_t e_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
    o_t = S_tᵀ q_t

- ``Diag(α_t)``: channel-wise decay (as in KDA), applied first.
- ``(I − γ_t e_t e_tᵀ)``: rank-1 erase along the learned cleanup address
  ``e_t`` (L2-normalized, scalar gate ``γ_t``). Querying the erased memory at
  ``e_t`` is suppressed by ``1 − γ_t``; queries orthogonal to ``e_t`` are
  unchanged by the erase step (paper Eq. 12).
- ``(I − β_t k_t k_tᵀ) ... + β_t k_t v_tᵀ``: the usual delta-style
  corrective write at the current key, unchanged.

Setting ``γ_t = 0`` (or omitting the erase gates) recovers the KDA update
exactly. Setting ``e_t = k_t`` collapses to a stronger same-address
correction — for normalized keys the combined erase+delta factor equals
``I − (β_t + γ_t − β_t γ_t) k_t k_tᵀ``.

Approximations relative to the paper (documented per issue #37 acceptance
criteria):

- Sequential reference recurrence only. The paper's doubled-sequence
  chunkwise kernel trick (its Eq. 17) is *not* implemented; the sequential
  form above is the defining recurrence. No production-performance claims
  are made or implied.
- The paper's erase-address low-rank projection and bounded "safe gate"
  decay parameterization are out of scope: like every variant in this
  repository, gates and addresses are supplied externally through the shared
  ``gates`` mapping, so the baseline plugs into the same task/evaluation
  path as KDA.

Interface notes:

- ``gates["alpha"]`` (``batch, seq_len, head_dim``) and ``gates["beta"]``
  (``batch, seq_len, 1``) are required — the same gate pair the KDA path
  already produces.
- ``gates["erase_address"]`` (``batch, seq_len, head_dim``) and
  ``gates["erase_gate"]`` (``batch, seq_len, 1``) are an optional pair. The
  erase address is L2-normalized internally (the paper uses normalized
  addresses). Supplying exactly one of the pair is rejected as ambiguous.
- Applied gate/address trajectories are exposed via ``return_trace=True``
  for the diagnostics utilities of issue #27.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from .base import DeltaMemoryOutput, DeltaMemoryVariant, register_variant


@register_variant("eda")
class EraseThenDeltaVariant(DeltaMemoryVariant):
    """Reference implementation of Erase-then-Delta Attention (arXiv:2606.26560).

    Args:
        head_dim: Dimension of queries/keys (``d_h``).
        value_dim: Dimension of values (``d_v``).
        eps: Small constant guarding the erase-address normalization.
    """

    def __init__(self, head_dim: int, value_dim: int, eps: float = 1e-6) -> None:
        super().__init__(head_dim=head_dim, value_dim=value_dim)
        self.eps = eps

    def _resolve_gates(
        self,
        gates: Optional[Mapping[str, torch.Tensor]],
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(alpha, beta, normalized_erase_address, erase_gate)``."""
        if gates is None or "alpha" not in gates or "beta" not in gates:
            raise ValueError(
                "EraseThenDeltaVariant requires gates containing 'alpha' and 'beta'"
            )
        alpha = gates["alpha"]
        beta = gates["beta"]
        if alpha.shape != (batch_size, seq_len, self.head_dim):
            raise ValueError(
                f"alpha must have shape {(batch_size, seq_len, self.head_dim)}, "
                f"got {tuple(alpha.shape)}"
            )
        if beta.shape != (batch_size, seq_len, 1):
            raise ValueError(
                f"beta must have shape {(batch_size, seq_len, 1)}, "
                f"got {tuple(beta.shape)}"
            )

        has_address = "erase_address" in gates
        has_gate = "erase_gate" in gates
        if has_address != has_gate:
            raise ValueError(
                "gates must supply both 'erase_address' and 'erase_gate' "
                "together, or neither (which recovers baseline KDA)"
            )
        if not has_address:
            return alpha, beta, None, None

        address = gates["erase_address"]
        erase_gate = gates["erase_gate"]
        if address.shape != (batch_size, seq_len, self.head_dim):
            raise ValueError(
                f"erase_address must have shape "
                f"{(batch_size, seq_len, self.head_dim)}, "
                f"got {tuple(address.shape)}"
            )
        if erase_gate.shape != (batch_size, seq_len, 1):
            raise ValueError(
                f"erase_gate must have shape {(batch_size, seq_len, 1)}, "
                f"got {tuple(erase_gate.shape)}"
            )
        # The paper uses L2-normalized erase addresses.
        address = F.normalize(address, p=2, dim=-1, eps=self.eps)
        return alpha, beta, address, erase_gate

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
        alpha, beta, erase_address, erase_gate = self._resolve_gates(
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
        erased_response_steps = []

        # EDA update (arXiv:2606.26560, Eq. 8), applied step by step.
        for t in range(seq_len):
            q_t = queries[:, t, :]          # (B, d_h)
            k_t = keys[:, t, :]             # (B, d_h)
            v_t = values[:, t, :]           # (B, d_v)
            alpha_t = alpha[:, t, :]        # (B, d_h)
            beta_t = beta[:, t, :]          # (B, 1)

            # 1) Decay: S_hat = Diag(alpha_t) S_{t-1}
            state = alpha_t.unsqueeze(-1) * state

            # 2) Erase at the independent address: (I − γ_t e_t e_tᵀ) S_hat
            if erase_address is not None:
                e_t = erase_address[:, t, :]                       # (B, d_h)
                gamma_t = erase_gate[:, t, :]                      # (B, 1)
                erased = torch.bmm(e_t.unsqueeze(1), state)        # (B, 1, d_v)
                state = state - gamma_t.unsqueeze(-1) * torch.bmm(
                    e_t.unsqueeze(-1), erased
                )
                if return_trace:
                    erased_response_steps.append(erased.squeeze(1))

            # 3) Delta corrective write at k_t (identical to KDACore):
            #    S_t = (I − β_t k_t k_tᵀ) S_erased + β_t k_t v_tᵀ
            k_state = torch.bmm(
                k_t.unsqueeze(-1), torch.bmm(k_t.unsqueeze(1), state)
            )
            state = state - beta_t.unsqueeze(-1) * k_state
            kv_outer = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
            state = state + beta_t.unsqueeze(-1) * kv_outer

            # Readout: o_t = S_tᵀ q_t
            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)

        outputs = torch.stack(outputs, dim=1)  # (B, L, d_v)

        trace = {}
        if return_trace:
            trace = {
                "alpha": alpha,
                "beta": beta,
                # (B, L, d_h) or None: L2-normalized erase address applied.
                "erase_address": erase_address,
                # (B, L, 1) or None: erase gate applied (None → exact KDA).
                "erase_gate": erase_gate,
                # (B, L, d_v) or None: memory response removed by each erase
                # step (e_tᵀ S_hat), useful for the #27 diagnostics.
                "erased_response": (
                    torch.stack(erased_response_steps, dim=1)
                    if erase_address is not None
                    else None
                ),
            }

        return DeltaMemoryOutput(outputs=outputs, final_state=state, trace=trace)
