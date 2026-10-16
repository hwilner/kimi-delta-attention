"""Calibration experiment: predicted uncertainty vs recall failure (issue #28).

This module implements the Phase 4 calibration experiment: *do the internal
uncertainty signals exposed by the delta-memory variants predict actual
retrieval failure?* It reuses the #35 key-collision task generator, the #26
variant registry, and the #27 diagnostics (state replay, spectra, gate
entropy, collision matrices) so that **no diagnostic ever reruns the model** —
every signal is computed from captured traces and tensors.

Design (all choices pre-registered in :class:`CalibrationConfig` before any
results are collected, per the issue's acceptance criteria):

- **Task.** :func:`kda.tasks.generate_key_collision_batch` with
  ``shuffle_queries=False`` produces ``2 * num_pairs`` key/value writes per
  sequence; each base key is paired with a colliding key at a controlled
  cosine. After the write pass, each key is queried against the *final*
  state, ``recall_i = S_L^T k_i`` (the variants' own readout operation applied
  to the captured final state — not a model rerun).
- **Failure label.** A query is a *recall failure* when
  ``cosine(recall_i, v_i) < failure_threshold`` (pre-registered; zero recall
  vectors score cosine 0 and therefore count as failures).
- **Uncertainty signals** (issue list; one scalar per query, aligned with the
  query's write step ``t = i``):

  1. ``evidence_along_key`` — KDA-RLS-lite (#36) occupancy score ``e_t``;
     raw sign is *negative* (more aligned evidence = less uncertainty).
  2. ``precision_along_key`` — precision-gated KDA (#33) post-downdate
     diagonal precision along the key, ``sum_c P_{t,c} k_{t,c}^2``; positive
     sign (``P`` is error-covariance-like in the RLS convention, so a larger
     value marks a less-certain direction).
  3. ``prediction_error_norm`` — per-step ``||v_t - (Diag(alpha_t)
     S_{t-1})^T k_t||_2``, replayed uniformly for every variant from captured
     tensors via :func:`kda.instrumentation.replay_state_trajectory`; positive
     sign.
  4. ``key_collision_score`` — maximum cosine similarity of the key to any
     *other* key in the same sequence (from
     :func:`kda.instrumentation.key_collision_statistics`); positive sign.
  5. ``gate_entropy`` — histogram entropy (nats) of the per-channel
     ``alpha_t`` gate at the write step (via
     :func:`kda.instrumentation.gate_statistics`); positive sign.
  6. ``state_effective_rank`` — Roy–Vetterli effective rank of the post-write
     state ``S_t`` (approximate, see #27); positive sign (a fuller state
     implies more interference).

  Signals a variant does not natively expose (e.g. evidence for baseline KDA)
  are reported as unavailable rather than imputed.
- **Metrics** (pre-registered list in the config): AUROC and average precision
  of the uncertainty score against the failure label, expected calibration
  error (ECE) after a deterministic min-max score-to-probability mapping
  (documented approximation — no recalibration is fitted, keeping the
  pre-registration clean), a calibration curve (probability bins), and recall
  accuracy per equal-mass uncertainty bin.
- **Uncertainty of the metrics.** Cluster bootstrap over sequences (whole
  sequences are resampled with replacement, respecting within-sequence
  correlation), percentile confidence intervals at the pre-registered level.
- **Honest negatives.** Flat (AUROC ~ 0.5) or negative (AUROC < 0.5) signals
  are reported in the same table as informative ones — the report schema has
  no per-signal filtering, and the example prints all of them.

Scope notes: research instrumentation on synthetic tasks only; no production
or performance claims. GDN-2/EDA baselines (#37) are prior art and are not
part of this comparison, which the issue fixes to KDA, KDA-RLS-lite, and
precision-gated KDA.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from kda.instrumentation import (
    gate_statistics,
    key_collision_statistics,
    replay_state_trajectory,
    state_metrics,
)
from kda.tasks import KeyCollisionTaskConfig, generate_key_collision_batch
from kda.variants import available_variants, create_variant

__all__ = [
    "CalibrationConfig",
    "CalibrationSamples",
    "SIGNAL_SPECS",
    "average_precision",
    "binary_auroc",
    "binned_recall",
    "bootstrap_ci",
    "calibration_curve",
    "collect_samples_for_batch",
    "collect_variant_samples",
    "expected_calibration_error",
    "run_calibration_experiment",
]

_EPS = 1e-8

# Pre-registered uncertainty signals (issue #28). ``sign`` converts the raw
# signal into an uncertainty score: ``score = sign * raw``, so a *larger* score
# always means *more* predicted uncertainty (higher failure risk).
SIGNAL_SPECS: dict[str, dict[str, object]] = {
    "evidence_along_key": {
        "sign": -1.0,
        "variants": ("kda-rls-lite",),
        "description": "RLS-lite occupancy e_t along the write key (negated).",
    },
    "precision_along_key": {
        "sign": 1.0,
        "variants": ("precision-gated-kda",),
        "description": "Post-downdate diagonal precision along the write key.",
    },
    "prediction_error_norm": {
        "sign": 1.0,
        "variants": None,  # all variants (replayed from captured tensors)
        "description": "L2 norm of the replayed delta-rule prediction error.",
    },
    "key_collision_score": {
        "sign": 1.0,
        "variants": None,
        "description": "Max cosine similarity to any other key in the sequence.",
    },
    "gate_entropy": {
        "sign": 1.0,
        "variants": None,
        "description": "Histogram entropy of alpha_t across channels (nats).",
    },
    "state_effective_rank": {
        "sign": 1.0,
        "variants": None,
        "description": "Effective rank of the post-write state (approximate).",
    },
}

_VALID_METRICS = ("auroc", "average_precision", "ece", "calibration_bins")


@dataclass(frozen=True)
class CalibrationConfig:
    """Pre-registered configuration for the #28 calibration experiment.

    The config is frozen and embedded verbatim in the report so the metric
    and signal choices are fixed *before* results are collected (acceptance
    criterion). All runs are deterministic given the seeds below.

    Attributes:
        variants: Registered variant names to compare (issue fixes these to
            KDA, KDA-RLS-lite, and precision-gated KDA).
        seeds: Task-generator seeds; each seed yields one batch.
        batch_size: Sequences per seed.
        num_pairs: Colliding key pairs per sequence (sequence length
            ``L = 2 * num_pairs``).
        key_dim: Key/query dimension (d_h).
        value_dim: Value dimension (d_v).
        collision_cosine: Cosine between each base key and its colliding key.
        gate_seed_offset: Gates are drawn from ``seed + gate_seed_offset`` so
            task content and gates are sampled independently.
        alpha_bias: Pre-registered bias of the alpha gate distribution
            (``alpha = sigmoid(N(0, 1) + alpha_bias)``); the positive default
            keeps decay mild enough that the task is not dominated by
            recency.
        beta_bias: Pre-registered bias of the beta gate distribution
            (``beta = sigmoid(N(0, 1) + beta_bias)``); the positive default
            keeps writes strong enough that failures are informative rather
            than saturated.
        failure_threshold: Cosine recall threshold below which a query counts
            as a recall failure.
        signals: Uncertainty signals to evaluate (subset of
            :data:`SIGNAL_SPECS`).
        metrics: Metrics to compute (subset of ``("auroc",
            "average_precision", "ece", "calibration_bins")``).
        ece_bins: Uniform probability bins for ECE / the calibration curve.
        recall_bins: Equal-mass uncertainty bins for recall-vs-uncertainty.
        entropy_bins: Histogram bins for the per-step gate-entropy signal.
        bootstrap_resamples: Cluster-bootstrap resamples (over sequences).
        ci_level: Confidence level for the percentile intervals.
        bootstrap_seed: Seed for the bootstrap resampling generator.
    """

    variants: tuple[str, ...] = ("kda", "kda-rls-lite", "precision-gated-kda")
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 8
    num_pairs: int = 8
    key_dim: int = 32
    value_dim: int = 32
    collision_cosine: float = 0.9
    gate_seed_offset: int = 10000
    alpha_bias: float = 2.0
    beta_bias: float = 1.0
    failure_threshold: float = 0.5
    signals: tuple[str, ...] = tuple(SIGNAL_SPECS)
    metrics: tuple[str, ...] = ("auroc", "average_precision", "ece", "calibration_bins")
    ece_bins: int = 10
    recall_bins: int = 10
    entropy_bins: int = 20
    bootstrap_resamples: int = 200
    ci_level: float = 0.95
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        unknown_variants = [v for v in self.variants if v not in available_variants()]
        if unknown_variants:
            raise ValueError(
                f"unknown variants in config: {unknown_variants}; "
                f"registered: {list(available_variants())}"
            )
        unknown_signals = [s for s in self.signals if s not in SIGNAL_SPECS]
        if unknown_signals:
            raise ValueError(
                f"unknown signals in config: {unknown_signals}; "
                f"available: {list(SIGNAL_SPECS)}"
            )
        unknown_metrics = [m for m in self.metrics if m not in _VALID_METRICS]
        if unknown_metrics:
            raise ValueError(
                f"unknown metrics in config: {unknown_metrics}; "
                f"valid: {list(_VALID_METRICS)}"
            )
        if not self.seeds:
            raise ValueError("seeds must be a non-empty tuple")
        if self.batch_size <= 0 or self.num_pairs <= 0:
            raise ValueError("batch_size and num_pairs must be positive")
        if self.key_dim < 2 or self.value_dim <= 0:
            raise ValueError("key_dim must be >= 2 and value_dim positive")
        if not -1.0 <= self.collision_cosine <= 1.0:
            raise ValueError("collision_cosine must be in [-1, 1]")
        if not -1.0 <= self.failure_threshold <= 1.0:
            raise ValueError("failure_threshold must be in [-1, 1]")
        if self.ece_bins <= 0 or self.recall_bins <= 0 or self.entropy_bins <= 0:
            raise ValueError("ece_bins, recall_bins, entropy_bins must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie in (0, 1)")

    def to_dict(self) -> dict:
        """JSON-serializable snapshot embedded in the experiment report."""

        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}


@dataclass
class CalibrationSamples:
    """Per-query calibration samples collected for one variant.

    Attributes:
        variant: Registered variant name.
        labels: ``(N,)`` float array; 1 = recall failure, 0 = success.
        groups: ``(N,)`` int array; unique id of the source sequence
            (``(seed, batch_idx)`` pair) for cluster bootstrap.
        signals: Mapping signal name -> ``(N,)`` float array of *uncertainty
            scores* (sign-adjusted; larger = more uncertain). Signals the
            variant does not expose are all-NaN.
        recall_cosines: ``(N,)`` float array of recall/target cosines.
    """

    variant: str
    labels: np.ndarray
    groups: np.ndarray
    signals: dict[str, np.ndarray]
    recall_cosines: np.ndarray


# ---------------------------------------------------------------------------
# Metric functions (pure numpy; labels: 1 = failure, 0 = success; higher
# scores = more predicted uncertainty).
# ---------------------------------------------------------------------------


def _check_binary(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"labels must be a non-empty 1-D array, got {labels.shape}")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary (0 = success, 1 = failure)")
    return labels


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks with tie handling (ascending order)."""

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U) of a failure-prediction score.

    Args:
        scores: ``(N,)`` uncertainty scores (larger = more uncertain).
        labels: ``(N,)`` binary labels (1 = recall failure).

    Returns:
        The probability that a random failure has a higher score than a
        random success (ties count 0.5). ``0.5`` is chance level; values
        below ``0.5`` mean the signal is *anti*-correlated with failure
        (a flat/negative result, reported as-is). ``nan`` when only one
        class is present.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = _check_binary(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must agree, got {scores.shape} and {labels.shape}")
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels == 1.0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision of a failure-prediction score (no interpolation).

    ``AP = sum_n (R_n - R_{n-1}) * P_n`` over thresholds at each distinct
    score, sorted descending — the standard non-interpolated definition.
    ``nan`` when there are no failures (undefined, since recall has no
    support); equals the failure prevalence for a chance ranking on average
    (not asserted here — deterministic given fixed data).
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = _check_binary(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must agree, got {scores.shape} and {labels.shape}")
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cum_pos = np.cumsum(sorted_labels)
    precision = cum_pos / (np.arange(labels.size) + 1.0)
    recall = cum_pos / n_pos
    is_positive = sorted_labels == 1.0
    delta_recall = np.diff(np.concatenate([[0.0], recall]))[is_positive]
    return float((precision[is_positive] * delta_recall).sum())


def _scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Deterministic min-max mapping of scores to ``[0, 1]`` probabilities.

    Documented approximation (pre-registered): no recalibration model is
    fitted, so ECE measures the raw ordering mapped to ``[0, 1]``. A constant
    score maps to all-zeros (probability 0).
    """

    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < _EPS:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def expected_calibration_error(
    scores: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> float:
    """Expected calibration error over uniform probability bins.

    Scores are mapped to probabilities with the pre-registered min-max rule
    (:func:`_scores_to_probabilities`). ``ECE = sum_m |B_m|/N *
    |acc(B_m) - conf(B_m)|`` where accuracy is the empirical *success* rate
    and confidence is ``1 - mean probability of failure``... note: labels are
    1 = failure, so within bin ``m`` the empirical failure rate is compared
    against the mean predicted failure probability. Empty bins contribute 0.
    """

    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    scores = np.asarray(scores, dtype=np.float64)
    labels = _check_binary(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must agree, got {scores.shape} and {labels.shape}")
    probs = _scores_to_probabilities(scores)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Clip bin index of exact 1.0 into the last bin.
    assignment = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = assignment == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def calibration_curve(
    scores: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> list[dict]:
    """Calibration curve: empirical failure rate vs predicted probability.

    Uses the same min-max probability mapping and uniform bins as
    :func:`expected_calibration_error`; empty bins are omitted.
    """

    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    scores = np.asarray(scores, dtype=np.float64)
    labels = _check_binary(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must agree, got {scores.shape} and {labels.shape}")
    probs = _scores_to_probabilities(scores)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    curve = []
    for b in range(bins):
        mask = assignment == b
        if not mask.any():
            continue
        curve.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "mean_predicted_failure_prob": float(probs[mask].mean()),
                "empirical_failure_rate": float(labels[mask].mean()),
            }
        )
    return curve


def binned_recall(
    scores: np.ndarray, labels: np.ndarray, *, num_bins: int = 10
) -> list[dict]:
    """Recall accuracy per equal-mass uncertainty bin.

    Bins are quantiles of the pooled scores (equal item counts, up to ties);
    each bin reports its score range, item count, failure rate, and recall
    accuracy (``1 - failure_rate``). A constant score yields a single bin.
    """

    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    scores = np.asarray(scores, dtype=np.float64)
    labels = _check_binary(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must agree, got {scores.shape} and {labels.shape}")
    if float(scores.max()) - float(scores.min()) < _EPS:
        return [
            {
                "bin": 0,
                "count": int(scores.size),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "failure_rate": float(labels.mean()),
                "recall_accuracy": float(1.0 - labels.mean()),
            }
        ]
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(scores, quantiles)
    edges[0], edges[-1] = float(scores.min()), float(scores.max())
    assignment = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, num_bins - 1)
    table = []
    for b in range(num_bins):
        mask = assignment == b
        if not mask.any():
            continue
        failure_rate = float(labels[mask].mean())
        table.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "score_min": float(scores[mask].min()),
                "score_max": float(scores[mask].max()),
                "failure_rate": failure_rate,
                "recall_accuracy": 1.0 - failure_rate,
            }
        )
    return table


def bootstrap_ci(
    arrays: tuple[np.ndarray, ...],
    groups: np.ndarray,
    metric_fn: Callable[..., float],
    *,
    resamples: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster-bootstrap percentile confidence interval for a metric.

    Whole sequences (unique ``groups`` values) are resampled with replacement
    so within-sequence correlation between queries is respected; the metric
    is recomputed on the pooled resampled items. ``nan`` metric values (e.g.
    AUROC of a single-class resample) are dropped; if every resample is
    ``nan`` the interval is ``(nan, nan)``.

    Args:
        arrays: Positional arrays forwarded to ``metric_fn`` (e.g.
            ``(scores, labels)``).
        groups: ``(N,)`` integer group id per item.
        metric_fn: Callable ``fn(*arrays) -> float``.
        resamples: Number of bootstrap resamples.
        level: Confidence level in ``(0, 1)``.
        seed: Seed for the resampling generator (deterministic).
    """

    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    groups = np.asarray(groups)
    n = groups.size
    for array in arrays:
        if np.asarray(array).shape[0] != n:
            raise ValueError("all arrays must match groups on the first axis")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inverse == g) for g in range(unique_groups.size)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        draws = rng.integers(0, unique_groups.size, size=unique_groups.size)
        index = np.concatenate([by_group[d] for d in draws])
        value = metric_fn(*(np.asarray(a)[index] for a in arrays))
        if not (isinstance(value, float) and math.isnan(value)):
            estimates.append(float(value))
    if not estimates:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    low, high = np.percentile(estimates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Sample collection: run each variant once per sequence, then derive every
# signal from the captured traces/tensors (no model reruns — issue #27).
# ---------------------------------------------------------------------------


def _sample_gates(
    batch_size: int, seq_len: int, key_dim: int, seed: int, config: CalibrationConfig
) -> dict[str, torch.Tensor]:
    """Pre-registered gate distribution: biased sigmoid-normal alpha and beta.

    ``alpha = sigmoid(N(0, 1) + alpha_bias)`` and likewise for beta; the
    biases are part of the pre-registered config.
    """

    generator = torch.Generator().manual_seed(seed)
    return {
        "alpha": torch.sigmoid(
            torch.randn(batch_size, seq_len, key_dim, generator=generator)
            + config.alpha_bias
        ),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, 1, generator=generator) + config.beta_bias
        ),
    }


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


def _replayed_prediction_error_norms(
    keys: torch.Tensor,
    values: torch.Tensor,
    alpha: torch.Tensor,
    states: torch.Tensor,
) -> torch.Tensor:
    """Per-step prediction-error norms recomputed from a replayed trajectory.

    ``states[:, t]`` is the post-update state ``S_t``; the step-``t`` error is
    ``e_t = v_t - (Diag(alpha_t) S_{t-1})^T k_t`` with ``S_0 = 0`` (or the
    captured initial state). This recomputes the linear recurrence only, from
    captured tensors — identical by construction to the precision-gated
    variant's ``trace["prediction_error"]`` (tested).
    """

    initial = torch.zeros_like(states[:, :1])
    previous = torch.cat([initial, states[:, :-1]], dim=1)  # (B, L, d_h, d_v)
    s_bar = alpha.unsqueeze(-1) * previous
    v_hat = torch.einsum("blhv,blh->blv", s_bar, keys)
    return torch.linalg.vector_norm(values - v_hat, dim=-1)  # (B, L)


def _per_step_gate_entropy(alpha: torch.Tensor, *, bins: int) -> torch.Tensor:
    """Histogram entropy (nats) of the alpha gate per (batch, step)."""

    batch, seq_len, _ = alpha.shape
    entropy = torch.zeros(batch, seq_len)
    for b in range(batch):
        for t in range(seq_len):
            entropy[b, t] = gate_statistics(alpha[b, t], bins=bins).entropy
    return entropy


def _per_key_collision_score(keys: torch.Tensor) -> torch.Tensor:
    """Max cosine similarity of each key to any other key in its sequence."""

    statistics = key_collision_statistics(keys, return_matrix=True)
    similarity = statistics.similarity_matrix  # (B, L, L)
    seq_len = keys.shape[1]
    similarity = similarity.masked_fill(
        torch.eye(seq_len, dtype=torch.bool), float("-inf")
    )
    return similarity.max(dim=-1).values  # (B, L)


def collect_samples_for_batch(
    variant_name: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    gates: dict[str, torch.Tensor],
    config: CalibrationConfig,
    *,
    group_offset: int = 0,
) -> CalibrationSamples:
    """Collect per-query labels and uncertainty signals for one batch.

    Runs one write pass of ``variant_name`` over ``(keys, values, gates)``
    with ``return_trace=True``, evaluates recall of every key against the
    captured final state, and derives all pre-registered signals from the
    trace and replayed trajectory. ``queries`` for the write pass equal
    ``keys`` (the task generator is used with ``shuffle_queries=False``), so
    signal step ``t`` aligns with query/item ``t``.

    Args:
        variant_name: Registered variant name.
        keys: ``(B, L, d_h)`` normalized keys (also used as read queries).
        values: ``(B, L, d_v)`` write values / recall targets.
        gates: Gate mapping with ``alpha`` ``(B, L, d_h)`` and ``beta``
            ``(B, L, 1)`` (beta unused by precision-gated KDA when enabled).
        config: Pre-registered experiment configuration.
        group_offset: Added to batch indices to form unique group ids.
    """

    batch, seq_len, key_dim = keys.shape
    value_dim = values.shape[-1]
    variant = create_variant(variant_name, head_dim=key_dim, value_dim=value_dim)
    with torch.no_grad():
        result = variant(keys, keys, values, gates=gates, return_trace=True)

        # Recall of each key against the captured final state (the variant's
        # own readout o = S^T q applied to stored tensors, not a rerun).
        recall = torch.einsum("bhv,blh->blv", result.final_state, keys)
        cosine = F.cosine_similarity(recall, values, dim=-1, eps=_EPS)  # (B, L)
        labels = (cosine < config.failure_threshold).to(torch.float64).numpy()

        write_vectors = _write_vectors(result.trace, keys)
        states = replay_state_trajectory(
            keys, values, gates["alpha"], write_vectors
        )
        error_norms = _replayed_prediction_error_norms(
            keys, values, gates["alpha"], states
        ).numpy()
        effective_rank = state_metrics(states).effective_rank.numpy()
        collision = _per_key_collision_score(keys).numpy()
        entropy = _per_step_gate_entropy(gates["alpha"], bins=config.entropy_bins).numpy()

        evidence = result.trace.get("evidence_along_key")
        evidence_raw = (
            evidence.squeeze(-1).numpy()
            if isinstance(evidence, torch.Tensor)
            else np.full((batch, seq_len), np.nan)
        )
        precision = result.trace.get("precision")
        precision_score = (
            (precision * keys * keys).sum(dim=-1).numpy()
            if isinstance(precision, torch.Tensor)
            else np.full((batch, seq_len), np.nan)
        )

    raw_signals = {
        "evidence_along_key": evidence_raw,
        "precision_along_key": precision_score,
        "prediction_error_norm": error_norms,
        "key_collision_score": collision,
        "gate_entropy": entropy,
        "state_effective_rank": effective_rank,
    }
    # Uniform sign convention: score = sign * raw, so a larger score always
    # means more predicted uncertainty (NaN signals stay NaN = unavailable).
    signals = {}
    for name in config.signals:
        sign = float(SIGNAL_SPECS[name]["sign"])
        signals[name] = sign * np.asarray(raw_signals[name], dtype=np.float64)

    groups = (np.arange(batch) + group_offset).repeat(seq_len)
    return CalibrationSamples(
        variant=variant_name,
        labels=labels.reshape(-1),
        groups=groups,
        signals={name: value.reshape(-1) for name, value in signals.items()},
        recall_cosines=cosine.numpy().reshape(-1),
    )


def collect_variant_samples(
    variant_name: str, config: CalibrationConfig
) -> CalibrationSamples:
    """Collect pooled samples for one variant over all pre-registered seeds."""

    parts = []
    for seed_index, seed in enumerate(config.seeds):
        task = generate_key_collision_batch(
            KeyCollisionTaskConfig(
                batch_size=config.batch_size,
                num_pairs=config.num_pairs,
                key_dim=config.key_dim,
                value_dim=config.value_dim,
                collision_cosine=config.collision_cosine,
                seed=seed,
                shuffle_queries=False,
            )
        )
        gates = _sample_gates(
            config.batch_size,
            task.keys.shape[1],
            config.key_dim,
            seed + config.gate_seed_offset,
            config,
        )
        parts.append(
            collect_samples_for_batch(
                variant_name,
                task.keys,
                task.values,
                gates,
                config,
                group_offset=seed_index * config.batch_size,
            )
        )
    return CalibrationSamples(
        variant=variant_name,
        labels=np.concatenate([p.labels for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        signals={
            name: np.concatenate([p.signals[name] for p in parts])
            for name in config.signals
        },
        recall_cosines=np.concatenate([p.recall_cosines for p in parts]),
    )


# ---------------------------------------------------------------------------
# Experiment runner: metrics are computed only for the pre-registered
# (signal, metric) pairs in the config; the config is embedded in the report.
# ---------------------------------------------------------------------------


def _metric_with_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    config: CalibrationConfig,
) -> dict:
    """Point estimate plus cluster-bootstrap CI for one metric."""

    return {
        "value": metric_fn(scores, labels),
        "ci": list(
            bootstrap_ci(
                (scores, labels),
                groups,
                metric_fn,
                resamples=config.bootstrap_resamples,
                level=config.ci_level,
                seed=config.bootstrap_seed,
            )
        ),
    }


def run_calibration_experiment(config: CalibrationConfig) -> dict:
    """Run the pre-registered #28 calibration experiment.

    For each variant in ``config.variants`` the experiment collects per-query
    recall-failure labels and uncertainty signals on the key-collision task
    (all seeds in ``config.seeds``), then evaluates every pre-registered
    (signal, metric) pair with bootstrap confidence intervals. Unavailable
    signals (e.g. evidence for baseline KDA) and flat/negative results are
    reported explicitly, never dropped (acceptance criterion).

    Returns:
        A JSON-serializable dict embedding the full config snapshot.
    """

    report = {
        "experiment": "phase4-calibration",
        "issue": 28,
        "description": (
            "Do internal uncertainty signals predict recall failure? "
            "Metrics pre-registered in the embedded config; bootstrap CIs "
            "are cluster resamples over sequences; flat/negative results "
            "are included by construction."
        ),
        "config": config.to_dict(),
        "variants": {},
    }
    for variant_name in config.variants:
        samples = collect_variant_samples(variant_name, config)
        labels, groups = samples.labels, samples.groups
        num_failures = int(labels.sum())
        section = {
            "num_sequences": int(np.unique(groups).size),
            "num_items": int(labels.size),
            "num_failures": num_failures,
            "failure_rate": _metric_with_ci(
                labels,  # scores placeholder; the metric uses labels only
                labels,
                groups,
                lambda _scores, lab: float(lab.mean()),
                config,
            ),
            "mean_recall_cosine": float(samples.recall_cosines.mean()),
            "signals": {},
        }
        for signal_name in config.signals:
            scores = samples.signals[signal_name]
            available = bool(np.isfinite(scores).any())
            entry: dict = {
                "available": available,
                "description": SIGNAL_SPECS[signal_name]["description"],
                "sign": float(SIGNAL_SPECS[signal_name]["sign"]),
            }
            if not available:
                entry["reason"] = f"variant '{variant_name}' exposes no such signal"
                section["signals"][signal_name] = entry
                continue
            mask = np.isfinite(scores)
            sig_scores, sig_labels, sig_groups = scores[mask], labels[mask], groups[mask]
            entry["num_items"] = int(sig_scores.size)
            if "auroc" in config.metrics:
                entry["auroc"] = _metric_with_ci(
                    sig_scores, sig_labels, sig_groups, binary_auroc, config
                )
            if "average_precision" in config.metrics:
                entry["average_precision"] = _metric_with_ci(
                    sig_scores, sig_labels, sig_groups, average_precision, config
                )
            if "ece" in config.metrics:
                entry["ece"] = _metric_with_ci(
                    sig_scores,
                    sig_labels,
                    sig_groups,
                    lambda s, lab: expected_calibration_error(s, lab, bins=config.ece_bins),
                    config,
                )
                entry["calibration_curve"] = calibration_curve(
                    sig_scores, sig_labels, bins=config.ece_bins
                )
            if "calibration_bins" in config.metrics:
                entry["recall_by_uncertainty_bin"] = binned_recall(
                    sig_scores, sig_labels, num_bins=config.recall_bins
                )
            section["signals"][signal_name] = entry
        report["variants"][variant_name] = section
    return report
