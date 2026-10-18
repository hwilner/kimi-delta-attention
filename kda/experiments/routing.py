"""Adaptive hybrid-routing experiment: fallback rate vs recall (issue #38).

Evaluates uncertainty-aware routing between the compressed KDA-family memory
and an exact-attention fallback on the #35 key-collision task. All choices —
variants, seeds, task/gate distributions, signals, threshold grid, and the
fixed-ratio baseline — are pre-registered in :class:`RoutingExperimentConfig`
before any results are collected, and the config is embedded verbatim in the
report (same discipline as the #28 calibration experiment).

Metrics per (variant, signal, threshold) operating point:

- ``fallback_rate`` — fraction of queries routed to the exact fallback. This
  is the **cost proxy**: exact attention over the cache is quadratic in
  sequence length while the compressed readout is not, so the trade-off is
  reported as quality-vs-fallback-rate. No wall-clock latency is measured
  anywhere in this prototype (research scope, per the issue).
- ``recall_accuracy`` — fraction of queries answered correctly after routing
  (routed queries use the exact readout, the rest the compressed one).
- ``improvement_over_compressed`` — recall gain vs the rate-0 endpoint.
- ``fixed_ratio_accuracy`` / ``gain_vs_fixed_ratio`` — the pre-registered
  baseline arm: a fixed-ratio hybrid routing the *same number* of queries
  chosen uniformly at random (deterministic seed). The adaptive arm only
  beats the baseline when the signal identifies *which* queries fail, not
  merely when the exact fallback helps.
- ``overuse_rate`` — fraction of routed queries whose compressed readout was
  already correct (wasted exact calls; router overuse failure mode).
- ``underuse_rate`` — fraction of *unrouted* queries whose compressed readout
  failed (missed failures; router underuse failure mode).

Flat or negative results (e.g. signals that the #28 calibration experiment
found uninformative at default settings) are reported in the same table as
informative ones — the report schema has no per-signal filtering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kda.routing import (
    ROUTING_SIGNALS,
    RoutingPolicy,
    compute_routing_signals,
    exact_attention_readout,
    fixed_ratio_decisions,
)
from kda.tasks import KeyCollisionTaskConfig, generate_key_collision_batch
from kda.variants import available_variants

__all__ = [
    "RoutingExperimentConfig",
    "RoutingVariantData",
    "collect_routing_data",
    "evaluate_operating_point",
    "run_routing_experiment",
]

_EPS = 1e-8


@dataclass(frozen=True)
class RoutingExperimentConfig:
    """Pre-registered configuration for the #38 routing experiment.

    The config is frozen and embedded verbatim in the report so the variant,
    signal, and threshold choices are fixed *before* results are collected.
    All runs are deterministic given the seeds below. Task and gate
    distributions deliberately mirror the #28 calibration defaults so the
    routing results can be read alongside the calibration table.

    Attributes:
        variants: Registered variant names providing the compressed memory.
        seeds: Task-generator seeds; each seed yields one batch.
        batch_size: Sequences per seed.
        num_pairs: Colliding key pairs per sequence (``L = 2 * num_pairs``).
        key_dim: Key/query dimension (d_h).
        value_dim: Value dimension (d_v).
        collision_cosine: Cosine between each base key and its colliding key.
        gate_seed_offset: Gates are drawn from ``seed + gate_seed_offset``.
        alpha_bias: Bias of the alpha gate distribution
            (``alpha = sigmoid(N(0, 1) + alpha_bias)``).
        beta_bias: Bias of the beta gate distribution.
        failure_threshold: Recall cosine below which a query is a failure.
        signals: Routing signals to evaluate (subset of
            :data:`kda.routing.ROUTING_SIGNALS`). Includes signals expected to
            be flat at default settings (honest-negatives criterion).
        threshold_quantiles: Quantiles of each pooled signal at which routing
            thresholds are placed (the pre-registered operating-point grid).
        fixed_ratio_seed: Seed of the fixed-ratio baseline selection.
        entropy_bins: Histogram bins for the gate-entropy signal.
        fallback_temperature: Softmax temperature of the exact-attention
            fallback (pre-registered; see
            :func:`kda.routing.exact_attention_readout`).
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
    signals: tuple[str, ...] = (
        "prediction_error_norm",
        "precision_along_key",
        "evidence_along_key",
        "key_collision_score",
        "gate_entropy",
    )
    threshold_quantiles: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    fixed_ratio_seed: int = 0
    entropy_bins: int = 20
    fallback_temperature: float = 0.1

    def __post_init__(self) -> None:
        unknown_variants = [v for v in self.variants if v not in available_variants()]
        if unknown_variants:
            raise ValueError(
                f"unknown variants in config: {unknown_variants}; "
                f"registered: {list(available_variants())}"
            )
        unknown_signals = [s for s in self.signals if s not in ROUTING_SIGNALS]
        if unknown_signals:
            raise ValueError(
                f"unknown signals in config: {unknown_signals}; "
                f"available: {list(ROUTING_SIGNALS)}"
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
        if not self.threshold_quantiles:
            raise ValueError("threshold_quantiles must be non-empty")
        if not all(0.0 <= q <= 1.0 for q in self.threshold_quantiles):
            raise ValueError("threshold_quantiles must lie in [0, 1]")
        if self.entropy_bins <= 0:
            raise ValueError("entropy_bins must be positive")
        if self.fallback_temperature <= 0.0:
            raise ValueError("fallback_temperature must be positive")

    def to_dict(self) -> dict:
        """JSON-serializable snapshot embedded in the experiment report."""

        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}


@dataclass
class RoutingVariantData:
    """Pooled per-query routing data for one variant.

    Attributes:
        variant: Registered variant name.
        compressed_ok: ``(N,)`` bool array; compressed recall succeeded.
        exact_ok: ``(N,)`` bool array; exact-fallback recall succeeded.
        scores: Mapping signal name -> ``(N,)`` uncertainty scores
            (sign-adjusted; larger = more uncertain; all-NaN = unavailable).
    """

    variant: str
    compressed_ok: np.ndarray
    exact_ok: np.ndarray
    scores: dict[str, np.ndarray]


def _sample_gates(
    batch_size: int, seq_len: int, key_dim: int, seed: int, config: RoutingExperimentConfig
) -> dict[str, torch.Tensor]:
    """Gate distribution pre-registered identically to the #28 experiment."""

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


def collect_routing_data(
    variant_name: str, config: RoutingExperimentConfig
) -> RoutingVariantData:
    """Collect pooled routing data for one variant over all seeds.

    For each pre-registered seed this runs one compressed write pass on the
    key-collision task, then derives recall outcomes (compressed and exact)
    and every routing signal from the captured traces — no model reruns.
    """

    compressed_parts, exact_parts = [], []
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name in config.signals}
    for seed in config.seeds:
        task = generate_key_collision_batch(
            KeyCollisionTaskConfig(
                batch_size=config.batch_size,
                num_pairs=config.num_pairs,
                key_dim=config.key_dim,
                value_dim=config.value_dim,
                collision_cosine=config.collision_cosine,
                seed=seed,
                shuffle_queries=False,  # queries = keys: write-step alignment
            )
        )
        gates = _sample_gates(
            config.batch_size,
            task.keys.shape[1],
            config.key_dim,
            seed + config.gate_seed_offset,
            config,
        )
        raw, result = compute_routing_signals(
            variant_name, task.keys, task.values, gates, entropy_bins=config.entropy_bins
        )
        with torch.no_grad():
            compressed = torch.einsum("bhv,blh->blv", result.final_state, task.keys)
            exact = exact_attention_readout(
                task.keys,
                task.keys,
                task.values,
                temperature=config.fallback_temperature,
            )
            compressed_cos = F.cosine_similarity(compressed, task.values, dim=-1, eps=_EPS)
            exact_cos = F.cosine_similarity(exact, task.values, dim=-1, eps=_EPS)
        compressed_parts.append((compressed_cos >= config.failure_threshold).numpy())
        exact_parts.append((exact_cos >= config.failure_threshold).numpy())
        for name in config.signals:
            sign = float(ROUTING_SIGNALS[name]["sign"])
            score_parts[name].append((sign * raw[name]).numpy())

    return RoutingVariantData(
        variant=variant_name,
        compressed_ok=np.concatenate([p.reshape(-1) for p in compressed_parts]),
        exact_ok=np.concatenate([p.reshape(-1) for p in exact_parts]),
        scores={
            name: np.concatenate([p.reshape(-1) for p in parts])
            for name, parts in score_parts.items()
        },
    )


def evaluate_operating_point(
    scores: np.ndarray,
    compressed_ok: np.ndarray,
    exact_ok: np.ndarray,
    threshold: float,
    *,
    fixed_ratio_seed: int = 0,
) -> dict:
    """Evaluate one routing threshold plus the matched fixed-ratio baseline.

    The baseline routes the *same number* of queries (realized adaptive rate)
    chosen uniformly at random, so the two arms differ only in which queries
    are routed — the pre-registered comparison required by issue #38.
    """

    scores = np.asarray(scores, dtype=np.float64)
    compressed_ok = np.asarray(compressed_ok, dtype=bool)
    exact_ok = np.asarray(exact_ok, dtype=bool)
    if not scores.shape == compressed_ok.shape == exact_ok.shape:
        raise ValueError("scores, compressed_ok, and exact_ok must agree in shape")
    routed = scores >= threshold
    rate = float(routed.mean())
    answered_ok = np.where(routed, exact_ok, compressed_ok)
    fixed_mask = fixed_ratio_decisions(scores.size, rate, seed=fixed_ratio_seed)
    fixed_ok = np.where(fixed_mask, exact_ok, compressed_ok)
    base_accuracy = float(compressed_ok.mean())
    accuracy = float(answered_ok.mean())
    return {
        "threshold": float(threshold),
        "fallback_rate": rate,
        "recall_accuracy": accuracy,
        "improvement_over_compressed": accuracy - base_accuracy,
        "fixed_ratio_accuracy": float(fixed_ok.mean()),
        "gain_vs_fixed_ratio": accuracy - float(fixed_ok.mean()),
        # Overuse: exact calls wasted on queries the compressed memory
        # already answered correctly. Underuse: failures left unrouted.
        "overuse_rate": float(compressed_ok[routed].mean()) if routed.any() else 0.0,
        "underuse_rate": float((~compressed_ok[~routed]).mean()) if (~routed).any() else 0.0,
        "num_routed": int(routed.sum()),
        "num_items": int(scores.size),
    }


def run_routing_experiment(config: RoutingExperimentConfig) -> dict:
    """Run the pre-registered #38 routing experiment.

    For each variant the experiment pools per-query routing data over all
    seeds, then sweeps each signal's threshold grid (quantiles of the pooled
    scores) and reports the quality/fallback-rate trade-off against both
    endpoints (rate 0 = pure compressed, rate 1 = pure exact) and the
    matched-budget fixed-ratio baseline. Unavailable signals and
    flat/negative results are reported explicitly, never dropped.

    Returns:
        A JSON-serializable dict embedding the full config snapshot.
    """

    report = {
        "experiment": "phase4-hybrid-routing",
        "issue": 38,
        "description": (
            "Uncertainty-aware routing between compressed delta memory and an "
            "exact-attention fallback. Thresholds, signals, seeds, and the "
            "fixed-ratio baseline are pre-registered in the embedded config; "
            "fallback rate is the cost proxy (no wall-clock latency measured); "
            "flat/negative results are included by construction."
        ),
        "config": config.to_dict(),
        "variants": {},
    }
    for variant_name in config.variants:
        data = collect_routing_data(variant_name, config)
        base_accuracy = float(data.compressed_ok.mean())
        exact_accuracy = float(data.exact_ok.mean())
        section = {
            "num_items": int(data.compressed_ok.size),
            "compressed_accuracy": base_accuracy,
            "exact_accuracy": exact_accuracy,
            "exact_headroom": exact_accuracy - base_accuracy,
            "signals": {},
        }
        for signal_name in config.signals:
            scores = data.scores[signal_name]
            available = bool(np.isfinite(scores).any())
            entry: dict = {
                "available": available,
                "description": ROUTING_SIGNALS[signal_name]["description"],
            }
            if not available:
                entry["reason"] = f"variant '{variant_name}' exposes no such signal"
                section["signals"][signal_name] = entry
                continue
            finite = np.isfinite(scores)
            sig_scores = scores[finite]
            sig_compressed = data.compressed_ok[finite]
            sig_exact = data.exact_ok[finite]
            thresholds = np.quantile(sig_scores, np.asarray(config.threshold_quantiles))
            entry["operating_points"] = [
                evaluate_operating_point(
                    sig_scores,
                    sig_compressed,
                    sig_exact,
                    float(threshold),
                    fixed_ratio_seed=config.fixed_ratio_seed,
                )
                for threshold in thresholds
            ]
            section["signals"][signal_name] = entry
        report["variants"][variant_name] = section
    return report
