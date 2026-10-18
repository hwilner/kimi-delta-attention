"""Adaptive hybrid routing between compressed KDA memory and exact attention.

This module implements the Phase 4 hybrid-routing prototype (issue #38): run
the compressed delta-memory write pass once, score every query's predicted
uncertainty from the captured traces (no model reruns — the signals reuse the
#27 instrumentation exactly as the #28 calibration experiment does), and route
the highest-uncertainty queries to a small **exact-attention fallback** that
re-reads the raw key/value cache with softmax attention.

Routing inputs (issue #38 list, plus the RLS-lite evidence signal):

- ``precision_along_key`` / ``evidence_along_key`` — native uncertainty
  signals of the precision-gated (#33) and RLS-lite (#36) variants;
- ``prediction_error_norm`` — replayed delta-rule prediction error;
- ``key_collision_score`` — max cosine of the query to any other stored key;
- ``gate_entropy`` — per-step entropy of the alpha gate.

Signal choice is configurable via :class:`RoutingPolicy`; every routing
decision is returned with its score and threshold in an audit log
(acceptance criterion: "configurable and auditable"). Note that the #28
calibration experiment found the *native* signals (precision/evidence)
informative and the *generic proxies* (gate entropy, effective rank) flat or
anti-correlated at default settings — the router supports all of them so the
experiment in :mod:`kda.experiments.routing` can document that contrast
honestly instead of hard-coding the conclusion.

Scope notes (per issue #38): research prototype only; no production
deployment or latency claims. The exact fallback's cost is reported as a
*fallback rate* (fraction of queries answered exactly); wall-clock latency is
never measured here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from kda.instrumentation import (
    gate_statistics,
    key_collision_statistics,
    replay_state_trajectory,
)
from kda.variants import create_variant

__all__ = [
    "ROUTING_SIGNALS",
    "HybridRoutingResult",
    "RoutingPolicy",
    "compute_routing_signals",
    "exact_attention_readout",
    "fixed_ratio_decisions",
    "route_hybrid_readout",
]

_EPS = 1e-8

# Routing-signal registry. ``sign`` converts the raw signal into an
# uncertainty score (``score = sign * raw``; larger = more uncertain = more
# worth routing to the exact fallback), matching the #28 convention.
# ``aligned=True`` marks write-step signals that require queries to align with
# the write steps (the read-what-you-wrote pattern of the #35 tasks); only
# ``key_collision_score`` is defined for arbitrary queries.
ROUTING_SIGNALS: dict[str, dict[str, object]] = {
    "evidence_along_key": {
        "sign": -1.0,
        "variants": ("kda-rls-lite",),
        "aligned": True,
        "description": "RLS-lite occupancy along the write key (negated).",
    },
    "precision_along_key": {
        "sign": 1.0,
        "variants": ("precision-gated-kda",),
        "aligned": True,
        "description": "Post-downdate diagonal precision along the write key.",
    },
    "prediction_error_norm": {
        "sign": 1.0,
        "variants": None,  # all variants (replayed from captured tensors)
        "aligned": True,
        "description": "L2 norm of the replayed delta-rule prediction error.",
    },
    "key_collision_score": {
        "sign": 1.0,
        "variants": None,
        "aligned": False,
        "description": "Max cosine similarity to any other stored key.",
    },
    "gate_entropy": {
        "sign": 1.0,
        "variants": None,
        "aligned": True,
        "description": "Histogram entropy of alpha_t across channels (nats).",
    },
}


def exact_attention_readout(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Exact softmax-attention readout over the raw key/value cache.

    ``out_i = softmax(q_i K^T / temperature) V`` — full (non-causal) attention
    over every write in the sequence, which is the natural *exact* oracle for
    read-after-write recall. This is deliberately the naive quadratic
    formulation: it is the fallback oracle, not a contribution, and its cost
    is exactly what the router tries to avoid paying on easy queries.

    Temperature note (pre-registered, not tuned per result): the task
    convention L2-normalizes keys, so ``q_i K^T`` are cosines in ``[-1, 1]``
    and a softmax at temperature 1 is too blunt to isolate a single stored
    key — an intrinsic property of softmax attention on the unit sphere. The
    default ``temperature=0.1`` sharpens the oracle enough to act as a recall
    fallback; the value is fixed here and in
    :class:`kda.experiments.RoutingExperimentConfig` *before* any results are
    collected.

    Args:
        queries: ``(B, N, d_h)`` read queries.
        keys: ``(B, L, d_h)`` stored keys (the write cache).
        values: ``(B, L, d_v)`` stored values.
        temperature: Softmax temperature for the cosine logits; must be
            positive.

    Returns:
        ``(B, N, d_v)`` exact-attention readouts.
    """

    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries, keys, and values must be rank-3 tensors")
    if keys.shape[:2] != values.shape[:2]:
        raise ValueError("keys and values must agree on batch and sequence dims")
    if queries.shape[0] != keys.shape[0] or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must agree on batch and head dims")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = torch.matmul(queries, keys.transpose(1, 2)) / temperature
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, values)


@dataclass(frozen=True)
class RoutingPolicy:
    """Configurable routing rule (acceptance criterion: auditable policy).

    A query is routed to the exact fallback when its uncertainty *score*
    (``sign * raw_signal``, larger = more uncertain) is at least
    ``threshold``; an optional ``max_fallback_rate`` caps how many queries per
    sequence may be routed, keeping the highest-scoring candidates.

    Attributes:
        signal: Signal name from :data:`ROUTING_SIGNALS`.
        threshold: Score at or above which a query is routed to the exact
            fallback.
        max_fallback_rate: Optional per-sequence budget in ``[0, 1]``; when
            set, at most ``floor(rate * seq_len)`` queries are routed (the
            top-scoring ones above the threshold).
        name: Free-form label recorded in the audit log.
    """

    signal: str
    threshold: float
    max_fallback_rate: Optional[float] = None
    name: str = "adaptive"

    def __post_init__(self) -> None:
        if self.signal not in ROUTING_SIGNALS:
            raise ValueError(
                f"unknown routing signal '{self.signal}'; "
                f"available: {list(ROUTING_SIGNALS)}"
            )
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")
        if self.max_fallback_rate is not None:
            if not 0.0 <= self.max_fallback_rate <= 1.0:
                raise ValueError(
                    f"max_fallback_rate must lie in [0, 1], "
                    f"got {self.max_fallback_rate}"
                )

    def apply(self, scores: torch.Tensor) -> torch.Tensor:
        """Return a boolean ``(B, N)`` mask of queries routed to exact.

        Args:
            scores: ``(B, N)`` uncertainty scores (larger = more uncertain).
        """

        if scores.ndim != 2:
            raise ValueError(f"scores must be a (batch, num_queries) tensor, got {scores.shape}")
        mask = scores >= self.threshold
        if self.max_fallback_rate is None:
            return mask
        budget = int(math.floor(self.max_fallback_rate * scores.shape[1]))
        if budget <= 0:
            return torch.zeros_like(mask)
        # Rank scores descending per sequence; candidates beyond the budget
        # (rank >= budget) are dropped even if above the threshold.
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1,
            order,
            torch.arange(scores.shape[1], device=scores.device).expand_as(order),
        )
        return mask & (ranks < budget)


def fixed_ratio_decisions(
    num_queries: int, ratio: float, *, seed: int = 0
) -> np.ndarray:
    """Deterministic fixed-ratio baseline router (issue #38 comparison arm).

    Routes exactly ``round(ratio * num_queries)`` queries chosen uniformly at
    random (seeded), ignoring all uncertainty signals. Comparing the adaptive
    policy against this baseline *at the same fallback rate* isolates the
    value of the signal: any recall gap between the two at a matched budget
    is attributable to *which* queries are routed, not *how many*.

    Args:
        num_queries: Number of queries to partition.
        ratio: Fraction routed to the exact fallback, in ``[0, 1]``.
        seed: Seed of the selection generator (deterministic).
    """

    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must lie in [0, 1], got {ratio}")
    count = int(round(ratio * num_queries))
    mask = np.zeros(num_queries, dtype=bool)
    if count:
        rng = np.random.default_rng(seed)
        mask[rng.permutation(num_queries)[:count]] = True
    return mask


def _write_vectors(trace: dict, keys: torch.Tensor) -> torch.Tensor:
    """Per-step write vector ``w_t`` for state replay (see #27 conventions)."""

    gain = trace.get("gain")
    if isinstance(gain, torch.Tensor):
        return gain  # precision-gated KDA: w_t = g_t
    for name in ("beta_eff", "beta_raw", "beta"):
        beta = trace.get(name)
        if isinstance(beta, torch.Tensor):
            return beta * keys  # KDA / KDA-RLS-lite: w_t = beta_t * k_t
    raise ValueError("trace exposes no write vector (gain / beta_eff / beta_raw / beta)")


def _prediction_error_norms(
    keys: torch.Tensor,
    values: torch.Tensor,
    alpha: torch.Tensor,
    states: torch.Tensor,
) -> torch.Tensor:
    """Per-step replayed prediction-error norms (identical to #28's version)."""

    initial = torch.zeros_like(states[:, :1])
    previous = torch.cat([initial, states[:, :-1]], dim=1)  # (B, L, d_h, d_v)
    s_bar = alpha.unsqueeze(-1) * previous
    v_hat = torch.einsum("blhv,blh->blv", s_bar, keys)
    return torch.linalg.vector_norm(values - v_hat, dim=-1)  # (B, L)


def _gate_entropy(alpha: torch.Tensor, *, bins: int = 20) -> torch.Tensor:
    """Histogram entropy (nats) of the alpha gate per (batch, step)."""

    batch, seq_len, _ = alpha.shape
    entropy = torch.zeros(batch, seq_len)
    for b in range(batch):
        for t in range(seq_len):
            entropy[b, t] = gate_statistics(alpha[b, t], bins=bins).entropy
    return entropy


def compute_routing_signals(
    variant_name: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    gates: Mapping[str, torch.Tensor],
    *,
    queries: Optional[torch.Tensor] = None,
    entropy_bins: int = 20,
) -> tuple[dict[str, torch.Tensor], object]:
    """Compute every routing signal from one captured write pass.

    Runs ``variant_name`` once over ``(keys, values, gates)`` with
    ``return_trace=True`` and derives all :data:`ROUTING_SIGNALS` raw values
    from the trace and replayed trajectory (no model reruns — same discipline
    as the #28 calibration experiment). Signals the variant does not expose
    are returned as all-NaN tensors, never imputed.

    Args:
        variant_name: Registered delta-memory variant name.
        keys: ``(B, L, d_h)`` write keys (L2-normalized).
        values: ``(B, L, d_v)`` write values.
        gates: Gate mapping with ``alpha`` ``(B, L, d_h)`` and ``beta``
            ``(B, L, 1)``.
        queries: Optional ``(B, N, d_h)`` read queries; defaults to ``keys``.
            Non-aligned queries are only supported for non-aligned signals
            (see :data:`ROUTING_SIGNALS`).
        entropy_bins: Histogram bins for the gate-entropy signal.

    Returns:
        ``(signals, result)`` where ``signals`` maps names to raw ``(B, N)``
        tensors (NaN = unavailable) and ``result`` is the variant's
        :class:`DeltaMemoryOutput` (its ``final_state`` is the compressed
        readout source).
    """

    batch, seq_len, key_dim = keys.shape
    if queries is None:
        queries = keys
    variant = create_variant(variant_name, head_dim=key_dim, value_dim=values.shape[-1])
    with torch.no_grad():
        result = variant(keys, keys, values, gates=gates, return_trace=True)

        write_vectors = _write_vectors(result.trace, keys)
        states = replay_state_trajectory(keys, values, gates["alpha"], write_vectors)
        error_norms = _prediction_error_norms(keys, values, gates["alpha"], states)

        aligned = queries.shape == keys.shape and torch.allclose(queries, keys)
        collision_stats = key_collision_statistics(
            torch.cat([keys, queries], dim=1) if not aligned else keys,
            return_matrix=True,
        )
        similarity = collision_stats.similarity_matrix
        if aligned:
            # Max cosine to any *other* key in the sequence.
            similarity = similarity.masked_fill(
                torch.eye(seq_len, dtype=torch.bool), float("-inf")
            )
            collision = similarity.max(dim=-1).values
        else:
            # Max cosine of each query to any stored key.
            collision = similarity[:, seq_len:, :seq_len].max(dim=-1).values

        nan = torch.full((batch, queries.shape[1]), float("nan"))
        evidence = result.trace.get("evidence_along_key")
        evidence_raw = evidence.squeeze(-1) if isinstance(evidence, torch.Tensor) else nan
        precision = result.trace.get("precision")
        precision_raw = (
            (precision * keys * keys).sum(dim=-1) if isinstance(precision, torch.Tensor) else nan
        )

    raw = {
        "evidence_along_key": evidence_raw,
        "precision_along_key": precision_raw,
        "prediction_error_norm": error_norms,
        "key_collision_score": collision,
        "gate_entropy": _gate_entropy(gates["alpha"], bins=entropy_bins),
    }
    return raw, result


@dataclass
class HybridRoutingResult:
    """Output of :func:`route_hybrid_readout` (auditable by construction).

    Attributes:
        outputs: ``(B, N, d_v)`` per-query readouts; routed queries hold the
            exact-attention readout, the rest the compressed ``S_L^T q``.
        routed: ``(B, N)`` boolean mask of queries sent to the exact fallback.
        scores: ``(B, N)`` uncertainty scores the policy was applied to
            (NaN when the signal is unavailable for the variant).
        fallback_rate: Fraction of queries routed (the cost proxy — no
            wall-clock latency is measured anywhere in this prototype).
        audit_log: One JSON-serializable dict per query with the batch/query
            index, raw score, threshold, policy name, and routing decision.
        compressed_recall: ``(B, N, d_v)`` compressed readouts (pre-routing).
        exact_recall: ``(B, N, d_v)`` exact readouts (the fallback oracle).
        final_state: ``(B, d_h, d_v)`` compressed memory state.
    """

    outputs: torch.Tensor
    routed: torch.Tensor
    scores: torch.Tensor
    fallback_rate: float
    audit_log: list[dict]
    compressed_recall: torch.Tensor
    exact_recall: torch.Tensor
    final_state: torch.Tensor


def route_hybrid_readout(
    variant_name: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    gates: Mapping[str, torch.Tensor],
    policy: RoutingPolicy,
    *,
    queries: Optional[torch.Tensor] = None,
    decisions: Optional[torch.Tensor] = None,
    entropy_bins: int = 20,
    fallback_temperature: float = 0.1,
) -> HybridRoutingResult:
    """Route queries between the compressed KDA state and exact attention.

    Runs the compressed write pass once, computes the configured uncertainty
    signal from captured traces, applies ``policy``, and substitutes the exact
    softmax readout for routed queries. Passing ``decisions`` (a boolean
    ``(B, N)`` mask) overrides the policy application — used to evaluate the
    fixed-ratio baseline under the *same* captured pass, so adaptive and
    baseline arms differ only in *which* queries are routed.

    Args:
        variant_name: Registered variant providing the compressed memory.
        keys: ``(B, L, d_h)`` write keys (L2-normalized).
        values: ``(B, L, d_v)`` write values.
        gates: Gate mapping (``alpha``, ``beta``).
        policy: The adaptive routing rule.
        queries: Optional read queries; defaults to ``keys``.
        decisions: Optional precomputed routing mask (baseline arm).
        entropy_bins: Histogram bins for the gate-entropy signal.
        fallback_temperature: Softmax temperature of the exact fallback
            (see :func:`exact_attention_readout`).
    """

    if queries is None:
        queries = keys
    spec = ROUTING_SIGNALS[policy.signal]
    aligned = queries.shape == keys.shape and torch.allclose(queries, keys)
    if spec["aligned"] and not aligned:
        raise ValueError(
            f"signal '{policy.signal}' is defined per write step and requires "
            "queries aligned with the write keys; only non-aligned signals "
            "('key_collision_score') support arbitrary queries in this prototype"
        )
    available = spec["variants"] is None or variant_name in spec["variants"]
    if not available:
        raise ValueError(
            f"variant '{variant_name}' exposes no '{policy.signal}' signal; "
            f"available for: {spec['variants']}"
        )

    raw, result = compute_routing_signals(
        variant_name, keys, values, gates, queries=queries, entropy_bins=entropy_bins
    )
    scores = float(spec["sign"]) * raw[policy.signal]
    if not torch.isfinite(scores).all():
        raise ValueError(f"signal '{policy.signal}' is unavailable (NaN) for this run")

    with torch.no_grad():
        compressed = torch.einsum("bhv,bnh->bnv", result.final_state, queries)
        exact = exact_attention_readout(
            queries, keys, values, temperature=fallback_temperature
        )
    if decisions is None:
        routed = policy.apply(scores)
    else:
        routed = decisions.to(dtype=torch.bool)
        if routed.shape != scores.shape:
            raise ValueError(
                f"decisions must match scores shape {scores.shape}, got {routed.shape}"
            )
    outputs = torch.where(routed.unsqueeze(-1), exact, compressed)

    audit_log = []
    for b in range(scores.shape[0]):
        for n in range(scores.shape[1]):
            audit_log.append(
                {
                    "batch": b,
                    "query": n,
                    "policy": policy.name,
                    "signal": policy.signal,
                    "score": float(scores[b, n]),
                    "threshold": policy.threshold,
                    "routed_to_exact": bool(routed[b, n]),
                }
            )
    return HybridRoutingResult(
        outputs=outputs,
        routed=routed,
        scores=scores,
        fallback_rate=float(routed.to(torch.float64).mean()),
        audit_log=audit_log,
        compressed_recall=compressed,
        exact_recall=exact,
        final_state=result.final_state,
    )
