"""Diagonal precision-gated KDA (Phase 4, issue #33).

This variant replaces KDA's fixed scalar write gate ``β_t`` with a
per-channel gain derived from a *diagonal precision* state, inspired by
recursive least squares (RLS) and Kalman filtering. The update is the
delta rule with a Kalman-style gain instead of ``β_t k_t``.

Notation per step ``t`` (batch dims omitted, ``P`` is the diagonal
precision vector ``P_t ∈ R^{d_h}``, stored as its per-channel diagonal):

    S_bar = Diag(alpha_t) S_{t-1}                        (decayed state)
    v_hat = S_barᵀ k_t                                   (prediction)
    e_t   = v_t - v_hat                                  (prediction error)
    g_t   = (P_t^- ⊙ k_t) / (λ + Σ_c P^-_t,c k_t,c²)     (write gain)
    S_t   = S_bar + g_t e_tᵀ                             (state update)

with the diagonal precision split into a *time update* and a
*measurement update*:

    P_t^- = P_{t-1} / forgetting + process_noise         (time update)
    P_t   = P_t^- ⊙ (1 - g_t ⊙ k_t)                      (measurement downdate)

Convention note: ``P`` follows the RLS convention — it is an
inverse-evidence / error-covariance-like diagonal, so the gain formula
matches the issue specification exactly. Writing along ``k_t`` *consumes*
precision in that direction (the downdate above), while forgetting
(``forgetting < 1``) and process noise (``process_noise > 0``) *inflate*
``P`` so older associations are re-learned faster. All precision values
are clamped to ``[p_min, p_max]`` every step for numerical stability.

Connection to baseline KDA: the standard KDA update can be written as
``S_t = S_bar + (β_t k_t) e_tᵀ``, i.e. KDA is the special case where the
gain is pinned to ``β_t k_t``. Setting ``precision_enabled=False``
delegates to :class:`~kda.core.KDACore` for exact baseline equivalence
(the fallback required by issue #33).

Computational overhead (acceptance criterion, documented before any
kernel work is considered): on top of :class:`KDACore`'s
``O(d_h · d_v)`` per-step state operations, this variant adds only
``O(d_h)`` elementwise work per step (precision time update, gain
normalization over the key dimension, precision downdate) and one extra
``(batch, d_h)`` buffer for the precision state. No kernel-level or
wall-clock measurements have been taken; this is an asymptotic note only,
not a performance claim.

Scope notes (per issue #33):
- Research prototype, not a tuned mechanism; no production performance
  claims are made.
- Precision, gain, and prediction-error trajectories are exposed via
  ``return_trace=True`` for the diagnostics work in issue #27.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from kda.core import KDACore

from .base import DeltaMemoryOutput, DeltaMemoryVariant, register_variant


@register_variant("precision-gated-kda")
class PrecisionGatedKDAVariant(DeltaMemoryVariant):
    """KDA with a diagonal-precision (Kalman/RLS-style) write gain.

    Args:
        head_dim: Dimension of queries/keys (d_h).
        value_dim: Dimension of values (d_v).
        eps: Small constant added to the gain denominator for numerical
            stability.
        precision_enabled: Master switch. When ``False``, the variant
            delegates to :class:`KDACore` and is numerically identical to
            baseline KDA (requires ``beta`` in ``gates``).
        precision_init: Initial per-channel precision ``P_0``. Must lie in
            ``[p_min, p_max]``.
        lambda_: Positive offset ``λ`` in the gain denominator (RLS-style
            regularization / observation-noise analogue). Larger values
            shrink the gain.
        forgetting: Forgetting factor ``γ ∈ (0, 1]`` for the precision
            time update ``P^- = P / γ``. ``1.0`` disables forgetting;
            smaller values inflate precision faster (older evidence is
            discounted more aggressively).
        process_noise: Non-negative additive term ``q`` in the precision
            time update, modeling state drift between writes.
        p_min: Lower clamp for the precision diagonal. Must be positive
            so the precision state cannot collapse to exactly zero (which
            would permanently freeze the gain at zero).
        p_max: Upper clamp for the precision diagonal, keeping the gain
            bounded when ``forgetting < 1`` or ``process_noise > 0``.
    """

    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        eps: float = 1e-6,
        precision_enabled: bool = True,
        precision_init: float = 1.0,
        lambda_: float = 1.0,
        forgetting: float = 1.0,
        process_noise: float = 0.0,
        p_min: float = 1e-3,
        p_max: float = 1e3,
    ) -> None:
        super().__init__(head_dim=head_dim, value_dim=value_dim)
        if not 0.0 < forgetting <= 1.0:
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}")
        if lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")
        if process_noise < 0.0:
            raise ValueError(
                f"process_noise must be non-negative, got {process_noise}"
            )
        if not 0.0 < p_min <= p_max:
            raise ValueError(
                f"require 0 < p_min <= p_max, got p_min={p_min}, p_max={p_max}"
            )
        if not p_min <= precision_init <= p_max:
            raise ValueError(
                f"precision_init must lie in [p_min, p_max], got "
                f"precision_init={precision_init}, p_min={p_min}, p_max={p_max}"
            )
        self.eps = eps
        self.precision_enabled = precision_enabled
        self.precision_init = precision_init
        self.lambda_ = lambda_
        self.forgetting = forgetting
        self.process_noise = process_noise
        self.p_min = p_min
        self.p_max = p_max
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
        if gates is None or "alpha" not in gates:
            raise ValueError(
                "PrecisionGatedKDAVariant requires gates containing 'alpha'"
            )

        alpha = gates["alpha"]

        if not self.precision_enabled:
            # Exact baseline KDA; the fixed scalar gate beta is required.
            if "beta" not in gates:
                raise ValueError(
                    "PrecisionGatedKDAVariant with precision_enabled=False "
                    "requires gates containing 'beta' (baseline KDA fallback)"
                )
            beta = gates["beta"]
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
                    "precision": None,
                    "gain": None,
                    "prediction_error": None,
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

        # Diagonal precision state P_t, one scalar per key channel.
        precision = torch.full(
            (batch_size, self.head_dim),
            self.precision_init,
            device=device,
            dtype=queries.dtype,
        )

        outputs = []
        precision_steps = []
        gain_steps = []
        prediction_error_steps = []

        # Recurrence follows KDACore's structure; the erase/write pair driven
        # by beta_t is replaced by a single rank-1 correction g_t e_tᵀ whose
        # gain comes from the diagonal precision state.
        for t in range(seq_len):
            q_t = queries[:, t, :]        # (B, d_h)
            k_t = keys[:, t, :]           # (B, d_h)
            v_t = values[:, t, :]         # (B, d_v)
            alpha_t = alpha[:, t, :]      # (B, d_h)

            # --- Precision time update -----------------------------------
            # P^- = P / forgetting + process_noise, clamped for stability.
            precision_prior = precision / self.forgetting + self.process_noise
            precision_prior = precision_prior.clamp(min=self.p_min, max=self.p_max)

            # --- Delta-rule prediction and error -------------------------
            state = alpha_t.unsqueeze(-1) * state            # S_bar
            v_hat = torch.bmm(state.transpose(1, 2), k_t.unsqueeze(-1)).squeeze(-1)
            error = v_t - v_hat                              # (B, d_v)

            # --- Precision-weighted gain ---------------------------------
            # g_t = (P^- ⊙ k_t) / (λ + Σ_c P^-_c k_t,c²)
            weighted_key = precision_prior * k_t             # (B, d_h)
            denom = (
                self.lambda_
                + (weighted_key * k_t).sum(dim=-1, keepdim=True)
                + self.eps
            )                                                # (B, 1)
            gain = weighted_key / denom                      # (B, d_h)

            # --- State update: S_t = S_bar + g_t e_tᵀ --------------------
            state = state + torch.bmm(gain.unsqueeze(-1), error.unsqueeze(1))

            # --- Precision measurement downdate ---------------------------
            # P_t = P^- ⊙ (1 - g_t ⊙ k_t); since g_c k_c ∈ [0, 1) the
            # precision along the written direction is consumed, not grown.
            precision = precision_prior * (1.0 - gain * k_t)
            precision = precision.clamp(min=self.p_min, max=self.p_max)

            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)

            if return_trace:
                precision_steps.append(precision.clone())
                gain_steps.append(gain.clone())
                prediction_error_steps.append(error.clone())

        outputs = torch.stack(outputs, dim=1)  # (B, L, d_v)

        trace = {}
        if return_trace:
            trace = {
                "alpha": alpha,
                # Pass-through for diagnostics comparability; beta is not
                # used by this variant when precision gating is enabled.
                "beta_raw": gates.get("beta"),
                # (B, L, d_h): post-downdate diagonal precision trajectory.
                "precision": torch.stack(precision_steps, dim=1),
                # (B, L, d_h): per-channel write gain actually applied.
                "gain": torch.stack(gain_steps, dim=1),
                # (B, L, d_v): delta-rule prediction error v_t - v_hat.
                "prediction_error": torch.stack(prediction_error_steps, dim=1),
            }

        return DeltaMemoryOutput(outputs=outputs, final_state=state, trace=trace)
