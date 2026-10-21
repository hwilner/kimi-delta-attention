"""Calibration robustness sweep across task regimes (extends issue #28).

The #28 calibration experiment (:mod:`kda.experiments.calibration`) evaluates
whether internal uncertainty signals predict recall failure at one
pre-registered operating point (``num_pairs=8``, ``collision_cosine=0.9``).
This module asks the follow-up question: *is that result stable across task
regimes, or specific to the default operating point?*

Design (all choices pre-registered in :class:`CalibrationRobustnessConfig`
before any results are collected, extending the #28 pre-registration):

- **Grid.** The Cartesian product of ``num_pairs_grid`` (sequence load;
  ``L = 2 * num_pairs`` writes per sequence) and ``collision_cosine_grid``
  (interference level). Every other field — variants, seeds, batch size,
  gate distribution, failure threshold, signal set — is fixed at the #28
  default, so the cell ``(num_pairs=8, collision_cosine=0.9)`` reproduces
  the #28 default experiment exactly (regression-tested against the #28
  pipeline).
- **Per cell.** Samples are collected with the #28 pipeline
  (:func:`kda.experiments.calibration.collect_variant_samples`): one write
  pass per variant and sequence, with every signal derived from captured
  traces and tensors — no model reruns (the #27 discipline). The
  pre-registered metrics for this sweep are the per-cell *failure rate* and
  the *AUROC* of each uncertainty signal, each with a cluster-bootstrap
  confidence interval over sequences (the #28 bootstrap discipline).
  ECE / calibration curves are intentionally out of scope here: the
  question is the stability of the *ranking* quality, not of the
  (documented, deterministic) score-to-probability mapping.
- **Honest negatives.** Every ``(cell, variant, signal)`` combination is
  reported — including unavailable signals (with an explicit reason) and
  flat (AUROC ~ 0.5) or negative (AUROC < 0.5) cells. The report schema has
  no per-cell or per-signal filtering.

Scope notes: research instrumentation on synthetic tasks only; no production
or performance claims. GDN-2/EDA (#37) are prior baselines and remain
outside this comparison, which #28 fixes to KDA, KDA-RLS-lite, and
precision-gated KDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kda.variants import available_variants

from .calibration import (
    SIGNAL_SPECS,
    CalibrationConfig,
    CalibrationSamples,
    binary_auroc,
    bootstrap_ci,
    collect_variant_samples,
)

__all__ = [
    "CalibrationRobustnessConfig",
    "regime_cells",
    "run_calibration_robustness",
]


@dataclass(frozen=True)
class CalibrationRobustnessConfig:
    """Pre-registered configuration for the calibration robustness sweep.

    Frozen and embedded verbatim in the report so the grid, signals, and
    metrics are fixed *before* results are collected (the #28 acceptance
    criterion, extended). All runs are deterministic given the seeds below.

    Attributes:
        variants: Registered variant names to compare (fixed by #28 to KDA,
            KDA-RLS-lite, and precision-gated KDA).
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
        alpha_bias: Bias of the alpha gate distribution (the #28 default:
            ``alpha = sigmoid(N(0, 1) + alpha_bias)``).
        beta_bias: Bias of the beta gate distribution (the #28 default:
            ``beta = sigmoid(N(0, 1) + beta_bias)``).
        failure_threshold: Cosine recall threshold below which a query
            counts as a recall failure.
        signals: Uncertainty signals to evaluate (subset of
            :data:`kda.experiments.calibration.SIGNAL_SPECS`).
        entropy_bins: Histogram bins for the per-step gate-entropy signal
            (passed through to the per-cell #28 config).
        bootstrap_resamples: Cluster-bootstrap resamples (over sequences).
        ci_level: Confidence level for the percentile intervals.
        bootstrap_seed: Seed for the bootstrap resampling generator.
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
    signals: tuple[str, ...] = tuple(SIGNAL_SPECS)
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
        if self.entropy_bins <= 0:
            raise ValueError("entropy_bins must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie in (0, 1)")

    def to_dict(self) -> dict:
        """JSON-serializable snapshot embedded in the experiment report."""

        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}


def regime_cells(config: CalibrationRobustnessConfig) -> list[CalibrationConfig]:
    """Expand the pre-registered grid into one #28 config per regime cell.

    Cells are ordered as the Cartesian product in tuple order: the outer
    loop runs over ``num_pairs_grid``, the inner loop over
    ``collision_cosine_grid``. Every field not swept by the grid is
    inherited from the robustness config unchanged, so a cell holding the
    #28 defaults (``num_pairs=8``, ``collision_cosine=0.9``) is field-for-
    field identical to the #28 default :class:`CalibrationConfig` on every
    field the collection pipeline reads.
    """

    cells = []
    for num_pairs in config.num_pairs_grid:
        for cosine in config.collision_cosine_grid:
            cells.append(
                CalibrationConfig(
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
                    metrics=("auroc",),  # only AUROC is consumed by this sweep
                    entropy_bins=config.entropy_bins,
                    bootstrap_resamples=config.bootstrap_resamples,
                    ci_level=config.ci_level,
                    bootstrap_seed=config.bootstrap_seed,
                )
            )
    return cells


def _metric_with_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    config: CalibrationRobustnessConfig,
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


def _signal_entry(
    samples: CalibrationSamples,
    signal_name: str,
    config: CalibrationRobustnessConfig,
) -> dict:
    """AUROC (+CI) for one signal, or an explicit unavailability record.

    Mirrors the #28 report semantics: a signal whose scores are all NaN for
    the variant (e.g. evidence for baseline KDA) is reported as unavailable
    with a reason rather than imputed; available signals are reported
    regardless of value — flat and negative AUROCs included.
    """

    scores = samples.signals[signal_name]
    entry: dict = {
        "available": bool(np.isfinite(scores).any()),
        "description": SIGNAL_SPECS[signal_name]["description"],
        "sign": float(SIGNAL_SPECS[signal_name]["sign"]),
    }
    if not entry["available"]:
        entry["reason"] = f"variant '{samples.variant}' exposes no such signal"
        return entry
    mask = np.isfinite(scores)
    entry["num_items"] = int(mask.sum())
    entry["auroc"] = _metric_with_ci(
        scores[mask], samples.labels[mask], samples.groups[mask],
        binary_auroc, config,
    )
    return entry


def run_calibration_robustness(config: CalibrationRobustnessConfig) -> dict:
    """Run the pre-registered calibration robustness sweep.

    For each regime cell (load × collision cosine) and each variant, the
    #28 collection pipeline produces per-query failure labels and
    uncertainty signals; the sweep reports the cell's failure rate and the
    AUROC of every pre-registered signal, each with a cluster-bootstrap CI.
    Unavailable signals and flat/negative results are included by
    construction (acceptance criterion — no filtering).

    Returns:
        A JSON-serializable dict embedding the full config snapshot.
    """

    report = {
        "experiment": "phase4-calibration-robustness",
        "issue": 28,
        "description": (
            "Is the failure-prediction quality of each uncertainty signal "
            "stable across task regimes (sequence load x collision cosine), "
            "or specific to the #28 default operating point? One #28 "
            "calibration cell per grid point; AUROC and failure rate with "
            "cluster-bootstrap CIs over sequences; unavailable, flat, and "
            "negative results are included by construction."
        ),
        "config": config.to_dict(),
        "cells": [],
    }
    for cell in regime_cells(config):
        cell_section: dict = {
            "num_pairs": cell.num_pairs,
            "collision_cosine": cell.collision_cosine,
            "variants": {},
        }
        for variant_name in config.variants:
            samples = collect_variant_samples(variant_name, cell)
            labels, groups = samples.labels, samples.groups
            section = {
                "num_sequences": int(np.unique(groups).size),
                "num_items": int(labels.size),
                "num_failures": int(labels.sum()),
                "failure_rate": _metric_with_ci(
                    labels,  # scores placeholder; the metric uses labels only
                    labels,
                    groups,
                    lambda _scores, lab: float(lab.mean()),
                    config,
                ),
                "mean_recall_cosine": float(samples.recall_cosines.mean()),
                "signals": {
                    signal_name: _signal_entry(samples, signal_name, config)
                    for signal_name in config.signals
                },
            }
            cell_section["variants"][variant_name] = section
        report["cells"].append(cell_section)
    return report
