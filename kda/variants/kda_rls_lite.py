"""KDA-RLS-lite: evidence-modulated beta gating (Phase 4, issue #36).

This is the cheapest uncertainty-aware KDA variant. It keeps the standard
key-aligned delta update of :class:`~kda.core.KDACore` untouched and derives
the effective write gate from a per-channel *evidence accumulator* inspired by
recursive least squares (RLS):

    n_t   = rho * alpha_t \u2299 n_{t-1} + k_t^2          (evidence accumulation)
    e_t   = sum_c n_{t,c} * k_{t,c}^2                (occupancy along k_t)
    \u03b2_eff = clamp(\u03b2_t / (\u03bb + e_t), \u03b2_min, \u03b2_max)     (evidence modulation)

Intuition: ``n_t`` counts (decayed) squared key mass written per channel \u2014
how much evidence the state has already received in each direction. ``e_t``
is the fraction of that evidence aligned with the current key ``k_t`` (for
L2-normalized keys it is a Rayleigh-quotient-style occupancy score). When the
state already holds a lot of evidence in the write direction, the write gate
is damped; novel directions (low ``e_t``) leave ``\u03b2`` nearly unchanged.

Scope notes (per issue #36):
- This is a research prototype, not a tuned mechanism; no production
  performance claims are made.
- Setting ``evidence_enabled=False`` recovers baseline KDA numerically
  (the recurrence below mirrors :class:`KDACore` operation-for-operation).
- Evidence and gate trajectories are exposed via ``return_trace=True`` for
  the diagnostics work in issue #27.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from kda.core import KDACore

from .base import DeltaMemoryOutput, DeltaMemoryVariant, register_variant


@register_variant("kda-rls-lite")
class KDARLSLiteVariant(DeltaMemoryVariant):
    """KDA with an RLS-lite evidence accumulator modulating the write gate.

    Args:
        head_dim: Dimension of queries/keys (d_h).
        value_dim: Dimension of values (d_v).
        eps: Small constant for numerical stability.
        evidence_enabled: Master switch. When ``False``, the variant is
            numerically identical to :class:`KDACore`.
        evidence_decay: Extra decay ``rho`` applied to the evidence buffer on
            top of ``alpha_t``. ``1.0`` reproduces the accumulator proposed in
            issue #36 (``n_t = alpha_t \u2299 n_{t-1} + k_t^2``); smaller values
            forget evidence faster. Must lie in ``(0, 1]``.
        lambda_: Non-negative offset ``\u03bb`` in the ``\u03b2_eff`` denominator.
            Larger values damp writes more conservatively; ``lambda_=1`` with
            zero evidence leaves ``\u03b2`` unchanged.
        beta_min: Lower clamp for the effective write gate.
        beta_max: Upper clamp for the effective write gate.
    """

    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        eps: float = 1e-6,
        evidence_enabled: bool = True,
        evidence_decay: float = 1.0,
        lambda_: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
    ) -> None:
        super().__init__(head_dim=head_dim, value_dim=value_dim)
        if not 0.0 < evidence_decay <= 1.0:
            raise ValueError(
                f"evidence_decay must be in (0, 1], got {evidence_decay}"
            )
        if lambda_ < 0.0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        if not 0.0 <= beta_min <= beta_max:
            raise ValueError(
                f"require 0 <= beta_min <= beta_max, got "
                f"beta_min={beta_min}, beta_max={beta_max}"
            )
        self.eps = eps
        self.evidence_enabled = evidence_enabled
        self.evidence_decay = evidence_decay
        self.lambda_ = lambda_
        self.beta_min = beta_min
        self.beta_max = beta_max
        # Reused for the disabled fast path so equivalence is by construction.
        self._core = KDACore(head_dim=head_dim, value_dim=value_dim, eps=eps)

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
        if gates is None:
            raise ValueError(
                "KDARLSLiteVariant requires gates containing 'alpha' and 'beta'"
            )
        if "alpha" not in gates or "beta" not in gates:
            raise ValueError(
                "KDARLSLiteVariant requires gates containing 'alpha' and 'beta'"
            )

        alpha = gates["alpha"]
        beta = gates["beta"]

        if not self.evidence_enabled:
            # Exact baseline KDA; traces still expose beta for diagnostics.
            outputs, final_state = self._core(
                queries=queries,
                keys=keys,
                values=values,
                alpha=alpha,
                beta=beta,
                initial_state=initial_state,
            )
            trace = (
                {
                    "alpha": alpha,
                    "beta_raw": beta,
                    "beta_eff": beta,
                    "evidence": None,
                    "evidence_along_key": None,
                }
                if return_trace
                else {}
            )
            return DeltaMemoryOutput(
                outputs=outputs, final_state=final_state, trace=trace
            )

        batch_size, seq_len, _ = queries.shape
        device = queries.device

        if initial_state is None:
            state = torch.zeros(
                batch_size,
                self.head_dim,
                self.value_dim,
                device=device,
                dtype=queries.dtype,
            )
        else:
            state = initial_state

        evidence = torch.zeros(
            batch_size, self.head_dim, device=device, dtype=queries.dtype
        )

        outputs = []
        evidence_steps = []
        evidence_along_key_steps = []
        beta_eff_steps = []

        # Recurrence mirrors KDACore operation-for-operation; the only change
        # is that beta_t is replaced by the evidence-modulated beta_eff.
        for t in range(seq_len):
            q_t = queries[:, t, :]        # (B, d_h)
            k_t = keys[:, t, :]           # (B, d_h)
            v_t = values[:, t, :]         # (B, d_v)
            alpha_t = alpha[:, t, :]      # (B, d_h)
            beta_t = beta[:, t, :]        # (B, 1)

            # --- Evidence update (RLS-lite) -------------------------------
            # n_t = rho * alpha_t \u2299 n_{t-1} + k_t^2
            evidence = self.evidence_decay * alpha_t * evidence + k_t * k_t
            # Occupancy of past evidence along the current write direction.
            e_t = (evidence * k_t * k_t).sum(dim=-1, keepdim=True)  # (B, 1)

            # beta_eff = clamp(beta / (lambda + e_t), beta_min, beta_max)
            beta_eff = beta_t / (self.lambda_ + e_t + self.eps)
            beta_eff = beta_eff.clamp(min=self.beta_min, max=self.beta_max)

            # --- KDA state update with beta_eff ---------------------------
            state = alpha_t.unsqueeze(-1) * state
            k_state = torch.bmm(
                k_t.unsqueeze(-1), torch.bmm(k_t.unsqueeze(1), state)
            )
            state = state - beta_eff.unsqueeze(-1) * k_state
            kv_outer = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
            state = state + beta_eff.unsqueeze(-1) * kv_outer

            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)

            if return_trace:
                evidence_steps.append(evidence.clone())
                evidence_along_key_steps.append(e_t.clone())
                beta_eff_steps.append(beta_eff.clone())

        outputs = torch.stack(outputs, dim=1)  # (B, L, d_v)

        trace = {}
        if return_trace:
            trace = {
                "alpha": alpha,
                "beta_raw": beta,
                # (B, L, 1): effective write gate actually applied.
                "beta_eff": torch.stack(beta_eff_steps, dim=1),
                # (B, L, d_h): per-channel evidence accumulator trajectory.
                "evidence": torch.stack(evidence_steps, dim=1),
                # (B, L, 1): evidence aligned with each step's key.
                "evidence_along_key": torch.stack(evidence_along_key_steps, dim=1),
            }

        return DeltaMemoryOutput(outputs=outputs, final_state=state, trace=trace)
