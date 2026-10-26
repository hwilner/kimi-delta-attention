"""Routing robustness sweep across task regimes (extends issue #38).

The #38 hybrid-routing experiment (:mod:`kda.experiments.routing`) evaluates
uncertainty-aware routing between the compressed KDA-family memory and an
exact-attention fallback at one pre-registered operating point
(``num_pairs=8``, ``collision_cosine=0.9``). This module asks the follow-up
question, mirroring the #28 robustness sweep
(:mod:`kda.experiments.calibration_robustness`): *is the routing result
stable across task regimes, or specific to the default operating point?*

Design (all choices pre-registered in :class:`RoutingRobustnessConfig`
before any results are collected, extending the #38 pre-registration):

- **Grid.** The Cartesian product of ``num_pairs_grid`` (sequence load;
  ``L = 2 * num_pairs`` writes per sequence) and ``collision_cosine_grid``
  (interference level). Every other field — variants, seeds, batch size,
  gate distribution, signals, threshold grid, fixed-ratio baseline — is
  fixed at the #38 default, so the cell ``(num_pairs=8,
  collision_cosine=0.9)`` reproduces the #38 default experiment exactly
  (regression-tested against the #38 pipeline).
- **Per cell.** Routing data are collected with the #38 pipeline
  (:func:`kda.experiments.routing.collect_routing_data`): one compressed
  write pass per variant and seed, with recall outcomes (compressed and
  exact) and every routing signal derived from captured traces — no model
  reruns (the #27 discipline). Per (variant, signal) the sweep reports the
  same operating-point metrics as #38 (fallback rate as the cost proxy,
  recall accuracy, gain over the matched-budget fixed-ratio baseline,
  overuse/underuse rates), evaluated per cell on that cell's pooled scores.
- **Honest negatives.** Every ``(cell, variant, signal)`` combination is
  reported — including unavailable signals (with an explicit reason) and
  cells where adaptive routing does not beat the fixed-ratio baseline. The
  report schema has no per-cell or per-signal filtering.

Scope notes: research instrumentation on synthetic tasks only; fallback rate
remains the cost proxy and no wall-clock latency is measured anywhere (the
#38 scope). GDN-2/EDA (#37) are prior baselines and remain outside this
comparison, which #38 fixes to KDA, KDA-RLS-lite, and precision-gated KDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kda.routing import ROUTING_SIGNALS
from kda.variants import available_variants

from .routing import (
    RoutingExperimentConfig,
    collect_routing_data,
    evaluate_operating_point,
)

__all__ = [
    "RoutingRobustnessConfig",
    "routing_regime_cells",
    "run_routing_robustness",
]


@dataclass(frozen=True)
class RoutingRobustnessConfig:
    """Pre-registered configuration for the routing robustness sweep.

    Frozen and embedded verbatim in the report so the grid, signals,
    threshold grid, and baselines are fixed *before* results are collected
    (the #38 acceptance criterion, extended). All runs are deterministic
    given the seeds below.

    Attributes:
        variants: Registered variant names providing the compressed memory
            (fixed by #38 to KDA, KDA-RLS-lite, and precision-gated KDA).
        seeds: Task-generator seeds; each seed yields one batch per cell.
        batch_size: Sequences per seed.
        num_pairs_grid: Sequence-load axis; each value is the number of
            colliding key pairs per sequence (``L = 2 * num_pairs``).
        collision_cosine_grid: Interference axis; each value is the cosine
            between a base key and its colliding key.
        key_dim: Key/query dimension (d_h).
        value_dim: Value dimension (d_v).
        gate_seed_offset: Gates are drawn from ``seed + gate_seed_offset``
            so task content and gates are sampled independently.
        alpha_bias: Bias of the alpha gate distribution (the #38 default:
            ``alpha = sigmoid(N(0, 1) + alpha_bias)``).
        beta_bias: Bias of the beta gate distribution (the #38 default).
        failure_threshold: Recall cosine threshold below which a query
            counts as a recall failure.
        signals: Routing signals to evaluate (subset of
            :data:`kda.routing.ROUTING_SIGNALS`).
        threshold_quantiles: Quantiles of each cell's pooled signal at which
            routing thresholds are placed (the #38 operating-point grid).
        fixed_ratio_seed: Seed of the fixed-ratio baseline selection.
        entropy_bins: Histogram bins for the gate-entropy signal.
        fallback_temperature: Softmax temperature of the exact-attention
            fallback (the #38 default).
    """

    variants: tuple[str, ...] = ("kda", "kda-rls-lite", "precision-gated-kda")
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 8
    num_pairs_grid: tuple[int, ...] = (4, 8, 16)
    collision_cosine_grid: tuple[float, ...] = (0.5, 0.9, 0.99)
    key_dim: int = 32
    value_dim: int = 32
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
        if not self.num_pairs_grid:
            raise ValueError("num_pairs_grid must be a non-empty tuple")
        if not self.collision_cosine_grid:
            raise ValueError("collision_cosine_grid must be a non-empty tuple")
        if any(int(n) != n or n <= 0 for n in self.num_pairs_grid):
            raise ValueError("num_pairs_grid entries must be positive integers")
        if any(not -1.0 <= c <= 1.0 for c in self.collision_cosine_grid):
            raise ValueError("collision_cosine_grid entries must be in [-1, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.key_dim < 2 or self.value_dim <= 0:
            raise ValueError("key_dim must be >= 2 and value_dim positive")
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


def routing_regime_cells(config: RoutingRobustnessConfig) -> list[RoutingExperimentConfig]:
    """Expand the pre-registered grid into one #38 config per regime cell.

    Cells are ordered as the Cartesian product in tuple order: the outer
    loop runs over ``num_pairs_grid``, the inner loop over
    ``collision_cosine_grid``. Every field not swept by the grid is
    inherited from the robustness config unchanged, so a cell holding the
    #38 defaults (``num_pairs=8``, ``collision_cosine=0.9``) is field-for-
    field identical to the #38 default :class:`RoutingExperimentConfig` on
    every field the collection pipeline reads.
    """

    cells = []
    for num_pairs in config.num_pairs_grid:
        for cosine in config.collision_cosine_grid:
            cells.append(
                RoutingExperimentConfig(
                    variants=config.variants,
                    seeds=config.seeds,
                    batch_size=config.batch_size,
                    num_pairs=int(num_pairs),
                    key_dim=config.key_dim,
                    value_dim=config.value_dim,
                    collision_cosine=float(cosine),
                    gate_seed_offset=config.gate_seed_offset,
                    alpha_bias=config.alpha_bias,
                    beta_bias=config.beta_bias,
                    failure_threshold=config.failure_threshold,
                    signals=config.signals,
                    threshold_quantiles=config.threshold_quantiles,
                    fixed_ratio_seed=config.fixed_ratio_seed,
                    entropy_bins=config.entropy_bins,
                    fallback_temperature=config.fallback_temperature,
                )
            )
    return cells


def _signal_entry(data, signal_name: str, config: RoutingRobustnessConfig) -> dict:
    """Operating points for one signal, or an explicit unavailability record.

    Mirrors the #38 report semantics: a signal whose scores are all NaN for
    the variant (e.g. precision for baseline KDA) is reported as unavailable
    with a reason rather than imputed; available signals are evaluated at
    every pre-registered threshold quantile of the cell's pooled scores
    regardless of outcome — flat and negative gains included.
    """

    scores = data.scores[signal_name]
    entry: dict = {
        "available": bool(np.isfinite(scores).any()),
        "description": ROUTING_SIGNALS[signal_name]["description"],
    }
    if not entry["available"]:
        entry["reason"] = f"variant '{data.variant}' exposes no such signal"
        return entry
    finite = np.isfinite(scores)
    sig_scores = scores[finite]
    thresholds = np.quantile(sig_scores, np.asarray(config.threshold_quantiles))
    entry["operating_points"] = [
        evaluate_operating_point(
            sig_scores,
            data.compressed_ok[finite],
            data.exact_ok[finite],
            float(threshold),
            fixed_ratio_seed=config.fixed_ratio_seed,
        )
        for threshold in thresholds
    ]
    return entry


def run_routing_robustness(config: RoutingRobustnessConfig) -> dict:
    """Run the pre-registered routing robustness sweep.

    For each regime cell (load x collision cosine) and each variant, the
    #38 collection pipeline produces per-query compressed/exact recall
    outcomes and routing signals; the sweep reports the cell's endpoint
    accuracies and, per signal, the quality/fallback-rate trade-off at every
    pre-registered threshold, including the matched-budget fixed-ratio
    baseline. Unavailable signals and flat/negative results are included by
    construction (acceptance criterion — no filtering).

    Returns:
        A JSON-serializable dict embedding the full config snapshot.
    """

    report = {
        "experiment": "phase4-routing-robustness",
        "issue": 38,
        "description": (
            "Is the quality/fallback-rate trade-off of uncertainty-aware "
            "routing stable across task regimes (sequence load x collision "
            "cosine), or specific to the #38 default operating point? One "
            "#38 routing cell per grid point; per-cell quantile thresholds; "
            "matched-budget fixed-ratio baseline per operating point; "
            "fallback rate is the cost proxy (no wall-clock latency "
            "measured); unavailable, flat, and negative results are "
            "included by construction."
        ),
        "config": config.to_dict(),
        "cells": [],
    }
    for cell in routing_regime_cells(config):
        cell_section: dict = {
            "num_pairs": cell.num_pairs,
            "collision_cosine": cell.collision_cosine,
            "variants": {},
        }
        for variant_name in config.variants:
            data = collect_routing_data(variant_name, cell)
            base_accuracy = float(data.compressed_ok.mean())
            exact_accuracy = float(data.exact_ok.mean())
            cell_section["variants"][variant_name] = {
                "num_items": int(data.compressed_ok.size),
                "compressed_accuracy": base_accuracy,
                "exact_accuracy": exact_accuracy,
                "exact_headroom": exact_accuracy - base_accuracy,
                "signals": {
                    signal_name: _signal_entry(data, signal_name, config)
                    for signal_name in config.signals
                },
            }
        report["cells"].append(cell_section)
    return report
