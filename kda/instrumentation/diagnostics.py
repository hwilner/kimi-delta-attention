"""State/gate diagnostics for recurrent-memory experiments (Phase 4, issue #27).

This module implements the reusable diagnostics required by issue #27 for the
memory-capacity and uncertainty experiments (#28 calibration, #38 hybrid
routing). Every function is a pure tensor-in/tensor-out transformation: all
metrics are computed from *captured* trajectories (variant ``trace`` dicts,
stored states, keys) and never require rerunning the model (acceptance
criterion). All functions are deterministic given the same input tensors.

Metrics and their conventions:

- **State norms** (:func:`state_metrics`): per-step Frobenius norm, spectral
  norm (largest singular value), singular-value spectrum, condition number,
  and effective rank of the recurrent state ``S_t ∈ R^{d_h × d_v}``.
  - *Condition number* is ``σ_max / σ_min``; rank-deficient states
    (``σ_min == 0``) report ``inf``. This includes the all-zero initial state.
  - *Effective rank* is the Roy–Vetterli effective rank
    ``exp(H(p))`` with ``p_i = σ_i / Σ_j σ_j`` (entropy of the normalized
    spectrum). It is a smooth, approximate measure of how many directions the
    state actually uses; an all-zero state has effective rank 0 by convention.
  - *Cost note*: the singular-value spectrum costs
    ``O(min(d_h, d_v) · d_h · d_v)`` per state matrix, batched over all
    leading dimensions. This is the most expensive diagnostic here and is
    computed eagerly by :func:`state_metrics`; call it on a subsampled set of
    steps for long trajectories.
- **Gate statistics** (:func:`gate_statistics`): population mean/std/min/max
  plus a *histogram-based* entropy (approximate — the gate distribution is
  binned over a fixed range, default ``[0, 1]`` since the gates in this
  repository are sigmoid outputs; values outside the range are clamped into
  the edge bins). Entropy is reported in nats.
- **Prediction-error norms** (:func:`prediction_error_metrics`): per-step L2
  norm of the delta-rule prediction error ``e_t = v_t - v̂_t`` exposed by the
  Phase 4 variants, plus scalar summaries.
- **Key-collision statistics** (:func:`key_collision_statistics`): pairwise
  cosine similarity between the keys of a sequence (cost ``O(L² · d_h)`` per
  batch element, documented here as the second-most-expensive diagnostic);
  pairs with similarity ``>= threshold`` count as collisions.
- **State-trajectory replay** (:func:`replay_state_trajectory`): reconstructs
  the per-step state trajectory of any KDA-family variant from captured
  tensors using the shared delta rule
  ``S_t = Diag(α_t) S_{t-1} + w_t (v_t - (Diag(α_t) S_{t-1})ᵀ k_t)ᵀ``, where
  the write vector ``w_t`` is ``β_t k_t`` for KDA / KDA-RLS-lite (with
  ``β_eff``) and the Kalman-style gain ``g_t`` for precision-gated KDA. The
  replay is exact (same operations, same order) and lets experiments obtain
  state trajectories from lightweight traces without modifying variant code.
  It recomputes the linear recurrence only — no model parameters are rerun.
- **Summary reports** (:func:`summarize_trajectory`): combines the above into
  a plain-dict report of floats for later experiments and example utilities.

Scope notes (per issue #27): research instrumentation, deterministic and
shape-tested; approximate metrics (histogram entropy, effective rank) and
expensive metrics (SVD-based spectra, ``O(L²)`` collision matrices) are
documented at their definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "GateStatistics",
    "KeyCollisionStatistics",
    "PredictionErrorMetrics",
    "StateMetrics",
    "gate_statistics",
    "key_collision_statistics",
    "prediction_error_metrics",
    "replay_state_trajectory",
    "state_metrics",
    "summarize_trajectory",
]

_INF = float("inf")


@dataclass
class StateMetrics:
    """Per-step spectral diagnostics for a state trajectory.

    Attributes:
        frobenius_norm: ``‖S_t‖_F`` with shape ``(...,)``.
        spectral_norm: Largest singular value ``σ_max(S_t)``, shape ``(...,)``.
        condition_number: ``σ_max / σ_min``, shape ``(...,)``; ``inf`` for
            rank-deficient states (including the all-zero state).
        effective_rank: Roy–Vetterli effective rank, shape ``(...,)``;
            approximate (entropy of the normalized spectrum), 0 for the
            all-zero state.
        singular_values: Full spectrum in descending order, shape
            ``(..., min(d_h, d_v))``.
    """

    frobenius_norm: torch.Tensor
    spectral_norm: torch.Tensor
    condition_number: torch.Tensor
    effective_rank: torch.Tensor
    singular_values: torch.Tensor


@dataclass
class GateStatistics:
    """Summary statistics for one gate tensor.

    Attributes:
        mean: Population mean over all elements.
        std: Population standard deviation (``correction=0``).
        min: Minimum element.
        max: Maximum element.
        entropy: Histogram-based entropy in nats (approximate; see
            :func:`gate_statistics`).
        num_elements: Number of elements summarized.
    """

    mean: float
    std: float
    min: float
    max: float
    entropy: float
    num_elements: int


@dataclass
class PredictionErrorMetrics:
    """Norms of the delta-rule prediction-error trajectory.

    Attributes:
        l2_norm: Per-step error norm ``‖e_t‖_2`` with shape ``(..., L)``.
        mean: Mean of ``l2_norm`` over all steps and batch elements.
        max: Maximum of ``l2_norm``.
        final: Mean of the final-step norm over batch elements.
    """

    l2_norm: torch.Tensor
    mean: float
    max: float
    final: float


@dataclass
class KeyCollisionStatistics:
    """Pairwise key-collision diagnostics for one or more sequences.

    Attributes:
        collision_counts: Number of unordered key pairs ``(i < j)`` with
            cosine similarity ``>= threshold``, shape ``(batch,)``.
        collision_fraction: ``collision_counts / total_pairs``, shape
            ``(batch,)``; 0 when the sequence has fewer than two keys.
        max_similarity: Maximum off-diagonal cosine similarity, shape
            ``(batch,)``; -1 when the sequence has fewer than two keys.
        threshold: Similarity threshold used.
        total_pairs: Number of unordered pairs compared, ``L·(L-1)/2``.
        similarity_matrix: Full cosine-similarity matrix ``(batch, L, L)``;
            only present when requested via ``return_matrix=True`` (cost
            ``O(L²)`` memory per batch element).
    """

    collision_counts: torch.Tensor
    collision_fraction: torch.Tensor
    max_similarity: torch.Tensor
    threshold: float
    total_pairs: int
    similarity_matrix: Optional[torch.Tensor] = None


def state_metrics(states: torch.Tensor) -> StateMetrics:
    """Compute spectral diagnostics for a trajectory of recurrent states.

    Args:
        states: Tensor of shape ``(..., d_h, d_v)``; each trailing
            ``(d_h, d_v)`` matrix is one recurrent state (e.g. the output of
            :func:`replay_state_trajectory` with shape ``(B, L, d_h, d_v)``).

    Returns:
        A :class:`StateMetrics` with per-state norms, condition number,
        effective rank, and the full singular-value spectrum. See the module
        docstring for conventions (``inf`` condition number for rank-deficient
        states) and the cost note — this diagnostic runs a batched SVD.
    """
    if states.dim() < 2:
        raise ValueError(
            f"states must have at least 2 dims (..., d_h, d_v), got {states.shape}"
        )
    frobenius = torch.linalg.vector_norm(states, dim=(-2, -1))
    # Batched SVD: the expensive part of this module (see module docstring).
    svals = torch.linalg.svdvals(states)
    spectral = svals[..., 0]
    smallest = svals[..., -1]
    condition = torch.where(
        smallest > 0,
        spectral / smallest.clamp_min(torch.finfo(svals.dtype).tiny),
        torch.full_like(spectral, _INF),
    )
    total = svals.sum(dim=-1, keepdim=True)
    probs = svals / total.clamp_min(torch.finfo(svals.dtype).tiny)
    # p·log(p) with p clamped so zero singular values contribute exactly 0.
    entropy = -(probs * torch.log(probs.clamp_min(torch.finfo(svals.dtype).tiny))).sum(
        dim=-1
    )
    effective_rank = torch.where(
        total.squeeze(-1) > 0, torch.exp(entropy), torch.zeros_like(entropy)
    )
    return StateMetrics(
        frobenius_norm=frobenius,
        spectral_norm=spectral,
        condition_number=condition,
        effective_rank=effective_rank,
        singular_values=svals,
    )


def gate_statistics(
    gate: torch.Tensor,
    *,
    bins: int = 50,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> GateStatistics:
    """Compute summary statistics for a single gate tensor.

    Args:
        gate: Tensor of any shape ``(...,)``; all elements are summarized.
        bins: Number of histogram bins for the entropy estimate.
        value_range: Fixed histogram range. Gates in this repository are
            sigmoid outputs in ``(0, 1)``; elements outside the range are
            clamped into the edge bins (documented approximation).

    Returns:
        A :class:`GateStatistics`. The entropy is a histogram-based
        approximation in nats: the (clamped) gate values are binned over
        ``value_range`` and ``-Σ p_b log p_b`` is taken over non-empty bins.
        A constant gate has entropy 0; a gate uniform over the range has
        entropy ``log(bins)``.
    """
    if gate.dim() < 1 or gate.numel() == 0:
        raise ValueError(f"gate must be a non-empty tensor, got {gate.shape}")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    lo, hi = value_range
    if not lo < hi:
        raise ValueError(f"value_range must satisfy lo < hi, got {value_range}")
    flat = gate.detach().reshape(-1).float()
    counts = torch.histc(flat.clamp(lo, hi), bins=bins, min=lo, max=hi)
    probs = counts / counts.sum()
    nonzero = probs > 0
    entropy = float(-(probs[nonzero] * probs[nonzero].log()).sum())
    return GateStatistics(
        mean=float(flat.mean()),
        std=float(flat.std(correction=0)),
        min=float(flat.min()),
        max=float(flat.max()),
        entropy=entropy,
        num_elements=flat.numel(),
    )


def prediction_error_metrics(errors: torch.Tensor) -> PredictionErrorMetrics:
    """Compute norms of a captured prediction-error trajectory.

    Args:
        errors: Tensor of shape ``(..., L, d_v)``, e.g. the
            ``"prediction_error"`` entry of a Phase 4 variant trace.

    Returns:
        A :class:`PredictionErrorMetrics` with per-step L2 norms of shape
        ``(..., L)`` and scalar summaries.
    """
    if errors.dim() < 2:
        raise ValueError(
            f"errors must have at least 2 dims (..., L, d_v), got {errors.shape}"
        )
    l2 = torch.linalg.vector_norm(errors, dim=-1)
    return PredictionErrorMetrics(
        l2_norm=l2,
        mean=float(l2.mean()),
        max=float(l2.max()),
        final=float(l2[..., -1].mean()),
    )


def key_collision_statistics(
    keys: torch.Tensor,
    *,
    threshold: float = 0.9,
    return_matrix: bool = False,
) -> KeyCollisionStatistics:
    """Compute key-collision statistics for a batch of key sequences.

    A *collision* is an unordered pair of distinct time steps ``(i, j)`` whose
    keys have cosine similarity ``>= threshold`` — such pairs write to nearly
    the same state direction and interfere under the delta rule.

    Args:
        keys: Tensor of shape ``(batch, L, d_h)``.
        threshold: Cosine-similarity threshold in ``[-1, 1]``.
        return_matrix: Also return the full ``(batch, L, L)`` cosine
            similarity matrix (``O(L²)`` memory per batch element).

    Returns:
        A :class:`KeyCollisionStatistics`. Cost is ``O(L² · d_h)`` per batch
        element (documented in the module docstring).
    """
    if keys.dim() != 3:
        raise ValueError(
            f"keys must have shape (batch, L, d_h), got {keys.shape}"
        )
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
    batch, seq_len, _ = keys.shape
    normalized = F.normalize(keys.detach().float(), dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    # Strictly upper-triangular mask: each unordered pair once, no diagonal.
    pair_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=keys.device),
        diagonal=1,
    )
    total_pairs = seq_len * (seq_len - 1) // 2
    if total_pairs == 0:
        counts = torch.zeros(batch, dtype=torch.long, device=keys.device)
        fraction = torch.zeros(batch, device=keys.device)
        max_similarity = torch.full((batch,), -1.0, device=keys.device)
    else:
        pair_similarities = similarity[:, pair_mask]  # (batch, total_pairs)
        counts = (pair_similarities >= threshold).sum(dim=-1)
        fraction = counts.float() / total_pairs
        max_similarity = pair_similarities.max(dim=-1).values
    return KeyCollisionStatistics(
        collision_counts=counts,
        collision_fraction=fraction,
        max_similarity=max_similarity,
        threshold=threshold,
        total_pairs=total_pairs,
        similarity_matrix=similarity if return_matrix else None,
    )


def replay_state_trajectory(
    keys: torch.Tensor,
    values: torch.Tensor,
    alpha: torch.Tensor,
    write_vectors: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reconstruct the per-step state trajectory of a KDA-family variant.

    Replays the shared delta rule from captured tensors (no model parameters
    are involved):

        ``S_bar = Diag(α_t) S_{t-1}``
        ``e_t   = v_t - S_barᵀ k_t``
        ``S_t   = S_bar + w_t e_tᵀ``

    where ``w_t`` is the per-step write vector: ``β_t k_t`` for KDA (use
    ``beta_eff ⊙ k_t`` for KDA-RLS-lite) and the gain ``g_t`` for
    precision-gated KDA. The replay uses the same operations in the same
    order as the variant implementations, so the final replayed state matches
    ``DeltaMemoryOutput.final_state`` up to floating-point associativity.

    Args:
        keys: Tensor ``(batch, L, d_h)``.
        values: Tensor ``(batch, L, d_v)``.
        alpha: Per-channel forget gate, tensor ``(batch, L, d_h)``.
        write_vectors: Per-channel write vector ``w_t``, tensor
            ``(batch, L, d_h)``.
        initial_state: Optional tensor ``(batch, d_h, d_v)``; zeros if omitted.

    Returns:
        Post-update states ``S_1 … S_L`` stacked along a new time dim, shape
        ``(batch, L, d_h, d_v)``. Cost is ``O(L · d_h · d_v)`` per batch
        element — the same as one forward pass of the recurrence itself.
    """
    if keys.dim() != 3:
        raise ValueError(f"keys must have shape (batch, L, d_h), got {keys.shape}")
    batch, seq_len, head_dim = keys.shape
    for name, tensor, last_dim in (
        ("values", values, None),
        ("alpha", alpha, head_dim),
        ("write_vectors", write_vectors, head_dim),
    ):
        if tensor.dim() != 3 or tensor.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"{name} must have shape (batch, L, -1) matching keys "
                f"{(batch, seq_len)}, got {tensor.shape}"
            )
        if last_dim is not None and tensor.shape[-1] != last_dim:
            raise ValueError(
                f"{name} must have last dim {last_dim}, got {tensor.shape}"
            )
    value_dim = values.shape[-1]
    if initial_state is None:
        state = keys.new_zeros(batch, head_dim, value_dim)
    else:
        if initial_state.shape != (batch, head_dim, value_dim):
            raise ValueError(
                f"initial_state must have shape {(batch, head_dim, value_dim)}, "
                f"got {initial_state.shape}"
            )
        state = initial_state.clone()
    states = []
    for t in range(seq_len):
        s_bar = alpha[:, t].unsqueeze(-1) * state
        v_hat = torch.bmm(s_bar.transpose(1, 2), keys[:, t].unsqueeze(-1)).squeeze(-1)
        error = values[:, t] - v_hat
        state = s_bar + torch.bmm(
            write_vectors[:, t].unsqueeze(-1), error.unsqueeze(1)
        )
        states.append(state.clone())
    return torch.stack(states, dim=1)


def summarize_trajectory(
    *,
    states: Optional[torch.Tensor] = None,
    trace: Optional[Mapping[str, object]] = None,
    keys: Optional[torch.Tensor] = None,
    collision_threshold: float = 0.9,
    entropy_bins: int = 50,
) -> dict:
    """Build a plain-dict diagnostics report from captured trajectories.

    This is the small report utility used by later experiments (#28, #38): it
    combines whichever diagnostics the caller has tensors for into a
    JSON-serializable mapping of floats (per-step tensors are reduced to
    batch-mean summaries; use the individual functions for full trajectories).

    Args:
        states: Optional state trajectory ``(batch, L, d_h, d_v)``, e.g. from
            :func:`replay_state_trajectory`.
        trace: Optional variant trace mapping. Recognized entries:
            ``"alpha"`` (always summarized if present), ``"beta_eff"`` (or
            ``"beta_raw"`` when no effective beta exists), ``"gain"``, and
            ``"prediction_error"``. Non-tensor entries (e.g. the ``None``
            placeholders of fallback traces) are skipped.
        keys: Optional key sequence ``(batch, L, d_h)`` for collision stats.
        collision_threshold: Threshold forwarded to
            :func:`key_collision_statistics`.
        entropy_bins: Histogram bins forwarded to :func:`gate_statistics`.

    Returns:
        A dict with up to four sections — ``"state"``, ``"gates"``,
        ``"prediction_error"``, ``"key_collisions"`` — containing only plain
        Python floats/ints, suitable for logging or JSON serialization.
    """
    report: dict = {}
    if states is not None:
        if states.dim() != 4:
            raise ValueError(
                f"states must have shape (batch, L, d_h, d_v), got {states.shape}"
            )
        metrics = state_metrics(states)
        # Batch-mean trajectories, then summarize over time.
        frob = metrics.frobenius_norm.mean(dim=0)
        spec = metrics.spectral_norm.mean(dim=0)
        erank = metrics.effective_rank.mean(dim=0)
        finite_cond = metrics.condition_number[
            torch.isfinite(metrics.condition_number)
        ]
        report["state"] = {
            "frobenius_norm_final": float(frob[-1]),
            "frobenius_norm_max": float(frob.max()),
            "spectral_norm_final": float(spec[-1]),
            "effective_rank_final": float(erank[-1]),
            "effective_rank_max": float(erank.max()),
            # Condition numbers are often inf early (rank-deficient states);
            # report the max over finite entries only, or inf if none exist.
            "condition_number_max_finite": (
                float(finite_cond.max()) if finite_cond.numel() > 0 else _INF
            ),
        }
    if trace is not None:
        gates_section: dict = {}
        for name in ("alpha", "beta_eff", "beta_raw", "gain"):
            tensor = trace.get(name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if name == "beta_raw" and isinstance(trace.get("beta_eff"), torch.Tensor):
                continue  # beta_eff supersedes the raw gate in the report.
            stats = gate_statistics(tensor, bins=entropy_bins)
            gates_section[name] = {
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "entropy": stats.entropy,
            }
        if gates_section:
            report["gates"] = gates_section
        error_tensor = trace.get("prediction_error")
        if isinstance(error_tensor, torch.Tensor):
            error_metrics = prediction_error_metrics(error_tensor)
            report["prediction_error"] = {
                "mean": error_metrics.mean,
                "max": error_metrics.max,
                "final": error_metrics.final,
            }
    if keys is not None:
        collisions = key_collision_statistics(keys, threshold=collision_threshold)
        report["key_collisions"] = {
            "threshold": collision_threshold,
            "total_pairs": collisions.total_pairs,
            "collision_count_mean": float(collisions.collision_counts.float().mean()),
            "collision_fraction_mean": float(collisions.collision_fraction.mean()),
            "max_similarity_mean": float(collisions.max_similarity.mean()),
            "max_similarity_max": float(collisions.max_similarity.max()),
        }
    return report
