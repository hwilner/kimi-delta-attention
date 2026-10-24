"""Hyperparameter ablation of the diagonal precision-gated KDA variant (#33).

The #33 prototype (:mod:`kda.variants.precision_gated_kda`, PR #44) landed
with a single default hyperparameter setting (``lambda_=1.0``,
``forgetting=1.0``, ``process_noise=0.0``), and the #34 results report
measured a *modest* precision-gated advantage over fixed-beta KDA at high
load. That raises the obvious follow-up question this module answers: *does
the #33 result depend on the default hyperparameters, or is it stable across
the natural perturbations of the gain denominator (lambda), the precision
time update (forgetting), and the state-drift term (process noise)?*

Design (all choices pre-registered in :class:`PrecisionGatedAblationConfig`
before any results are collected, extending the #28/#34 discipline):

- **Grid.** The Cartesian product of ``lambda_grid`` (gain-denominator
  offset), ``forgetting_grid`` (precision time-update forgetting factor),
  and ``process_noise_grid`` (additive drift term). The grid is centered on
  the #33 default cell ``(lambda_=1.0, forgetting=1.0, process_noise=0.0)``,
  which reproduces the #34 precision-gated rows exactly (regression-tested
  against the #34 capacity pipeline). The remaining variant hyperparameters
  (``precision_init``, ``p_min``, ``p_max``) are *not* swept: they are clamp
  bookkeeping fixed at the #33 defaults, documented here as a scope
  decision, not tuned after seeing results.
- **Operating points.** ``operating_points`` fixes the task regimes up
  front: the #28 default cell (``num_pairs=8``, ``collision_cosine=0.9``)
  and the high-load cell (``num_pairs=16``, ``collision_cosine=0.9``) where
  #34 observed the precision-gated advantage. Every other field — seeds,
  batch size, gate distribution, failure threshold — is fixed at the shared
  Phase 4 defaults (identical gate draws to #28/#34, so tables are
  collected under the same regime).
- **Reference rows.** ``reference_variants`` (fixed-beta KDA) run on the
  identical task/gate draws at each operating point, so every ablation cell
  is read against the same baseline it would be compared to in #34. No
  per-cell filtering: weak, flat, or collapsed cells are reported alongside
  strong ones (honest negatives by construction).
- **Metrics.** Per (operating point, cell): recall accuracy (cosine >=
  ``failure_threshold``) with a cluster-bootstrap CI over sequences, and
  the mean recall cosine. Uncertainty-signal calibration is out of scope
  here (covered by #28/#50); this sweep is about the *write rule's*
  sensitivity to its own hyperparameters.

Scope notes: research instrumentation on synthetic tasks only; sequential
reference recurrences; no wall-clock, kernel, or production-performance
measurements. GDN-2/EDA (#37) are prior baselines and are not part of this
ablation, which concerns only the #33 variant's own hyperparameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kda.tasks import KeyCollisionTaskConfig, generate_key_collision_batch
from kda.variants import available_variants, create_variant

from .calibration import bootstrap_ci
from .report_experiments import _sample_gates

__all__ = [
    "PrecisionGatedAblationConfig",
    "ablation_cells",
    "is_default_cell",
    "run_precision_gated_ablation",
]

_EPS = 1e-8

# The #33 default hyperparameters (PR #44 constructor defaults). The ablation
# grid is centered on this cell so the sweep always contains the exact
# configuration measured in the #34 report.
DEFAULT_LAMBDA = 1.0
DEFAULT_FORGETTING = 1.0
DEFAULT_PROCESS_NOISE = 0.0


@dataclass(frozen=True)
class PrecisionGatedAblationConfig:
    """Pre-registered configuration for the #33 hyperparameter ablation.

    Frozen and embedded verbatim in the report so the grid, operating
    points, and metrics are fixed *before* results are collected. All runs
    are deterministic given the seeds below.

    Attributes:
        lambda_grid: Gain-denominator offsets ``λ`` to sweep. Larger values
            shrink the Kalman/RLS-style gain toward zero.
        forgetting_grid: Forgetting factors ``γ ∈ (0, 1]`` for the precision
            time update ``P^- = P / γ``. Values below 1 inflate precision
            over time (older evidence discounted).
        process_noise_grid: Non-negative additive drift terms ``q`` for the
            precision time update.
        operating_points: ``(num_pairs, collision_cosine)`` task regimes at
            which every grid cell is evaluated (``L = 2 * num_pairs``).
        reference_variants: Fixed-hyperparameter reference variants run on
            the identical draws (fixed-beta KDA by default).
        seeds: Task-generator seeds; each seed yields one batch per point.
        batch_size: Sequences per seed.
        key_dim: Key/query dimension (d_h).
        value_dim: Value dimension (d_v).
        gate_seed_offset: Gates are drawn from ``seed + gate_seed_offset``
            so task content and gates are sampled independently (shared
            with #28/#34).
        alpha_bias: Bias of the alpha gate distribution
            (``alpha = sigmoid(N(0, 1) + alpha_bias)``), mirroring #28.
        beta_bias: Bias of the beta gate distribution, mirroring #28.
        failure_threshold: Recall cosine at/above which a query counts as
            correctly recalled.
        precision_init: Initial per-channel precision (fixed #33 default;
            not swept — clamp bookkeeping, see module docstring).
        p_min: Lower precision clamp (fixed #33 default; not swept).
        p_max: Upper precision clamp (fixed #33 default; not swept).
        bootstrap_resamples: Cluster-bootstrap resamples over sequences.
        ci_level: Confidence level of the percentile intervals.
        bootstrap_seed: Seed of the bootstrap resampling generator.
    """

    lambda_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
    forgetting_grid: tuple[float, ...] = (0.9, 1.0)
    process_noise_grid: tuple[float, ...] = (0.0, 0.05)
    operating_points: tuple[tuple[int, float], ...] = ((8, 0.9), (16, 0.9))
    reference_variants: tuple[str, ...] = ("kda",)
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 8
    key_dim: int = 32
    value_dim: int = 32
    gate_seed_offset: int = 10000
    alpha_bias: float = 2.0
    beta_bias: float = 1.0
    failure_threshold: float = 0.5
    precision_init: float = 1.0
    p_min: float = 1e-3
    p_max: float = 1e3
    bootstrap_resamples: int = 200
    ci_level: float = 0.95
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        unknown = [
            v for v in self.reference_variants if v not in available_variants()
        ]
        if unknown:
            raise ValueError(
                f"unknown reference variants in config: {unknown}; "
                f"registered: {list(available_variants())}"
            )
        if not self.lambda_grid or any(lam <= 0.0 for lam in self.lambda_grid):
            raise ValueError("lambda_grid must be non-empty positive floats")
        if not self.forgetting_grid or any(
            not 0.0 < gamma <= 1.0 for gamma in self.forgetting_grid
        ):
            raise ValueError("forgetting_grid values must lie in (0, 1]")
        if not self.process_noise_grid or any(
            q < 0.0 for q in self.process_noise_grid
        ):
            raise ValueError(
                "process_noise_grid must be non-empty non-negative floats"
            )
        if not self.operating_points:
            raise ValueError("operating_points must be non-empty")
        for point in self.operating_points:
            num_pairs, cosine = point
            if num_pairs <= 0:
                raise ValueError("operating point num_pairs must be positive")
            if not -1.0 <= cosine <= 1.0:
                raise ValueError(
                    "operating point collision_cosine must lie in [-1, 1]"
                )
        if not self.seeds:
            raise ValueError("seeds must be a non-empty tuple")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.key_dim < 2 or self.value_dim <= 0:
            raise ValueError("key_dim must be >= 2 and value_dim positive")
        if not -1.0 <= self.failure_threshold <= 1.0:
            raise ValueError("failure_threshold must be in [-1, 1]")
        if not 0.0 < self.p_min <= self.p_max:
            raise ValueError("require 0 < p_min <= p_max")
        if not self.p_min <= self.precision_init <= self.p_max:
            raise ValueError("precision_init must lie in [p_min, p_max]")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError("ci_level must lie in (0, 1)")

    def to_dict(self) -> dict:
        """JSON-serializable snapshot embedded in the experiment report."""

        snapshot = {}
        for key, value in asdict(self).items():
            if isinstance(value, tuple):
                snapshot[key] = [
                    list(item) if isinstance(item, tuple) else item
                    for item in value
                ]
            else:
                snapshot[key] = value
        return snapshot


def ablation_cells(
    config: PrecisionGatedAblationConfig,
) -> list[dict[str, float]]:
    """Expand the pre-registered grids into hyperparameter cells.

    Cartesian product in deterministic order — ``lambda_grid`` outermost,
    then ``forgetting_grid``, then ``process_noise_grid`` — so the report
    tables and their tests have a fixed layout.
    """

    return [
        {
            "lambda_": lambda_,
            "forgetting": forgetting,
            "process_noise": process_noise,
        }
        for lambda_ in config.lambda_grid
        for forgetting in config.forgetting_grid
        for process_noise in config.process_noise_grid
    ]


def is_default_cell(cell: dict[str, float]) -> bool:
    """Whether a cell coincides with the #33 default hyperparameters (#34 rows)."""

    return (
        cell["lambda_"] == DEFAULT_LAMBDA
        and cell["forgetting"] == DEFAULT_FORGETTING
        and cell["process_noise"] == DEFAULT_PROCESS_NOISE
    )


def _collect_point(
    variant_name: str,
    hyperparameters: dict[str, float] | None,
    num_pairs: int,
    collision_cosine: float,
    config: PrecisionGatedAblationConfig,
) -> dict:
    """Recall outcomes for one (cell or reference) at one operating point.

    One write pass per seed on the #35 key-collision task; every key is then
    recalled against the captured final state (the variant's own readout
    ``S_L^T k_i`` on stored tensors — not a rerun), exactly as in #28/#34.
    Task and gate draws use the identical seeds and gate distribution as the
    #34 capacity pipeline, so the default cell at ``(8, 0.9)`` reproduces
    the #34 precision-gated row bit-for-bit (regression-tested).
    """

    success_parts, group_parts, cosine_parts = [], [], []
    for seed_index, seed in enumerate(config.seeds):
        task = generate_key_collision_batch(
            KeyCollisionTaskConfig(
                batch_size=config.batch_size,
                num_pairs=num_pairs,
                key_dim=config.key_dim,
                value_dim=config.value_dim,
                collision_cosine=collision_cosine,
                seed=seed,
                shuffle_queries=False,  # queries = keys: write-step alignment
            )
        )
        gates = _sample_gates(
            config.batch_size,
            task.keys.shape[1],
            config.key_dim,
            seed + config.gate_seed_offset,
            config.alpha_bias,
            config.beta_bias,
        )
        variant = create_variant(
            variant_name,
            head_dim=config.key_dim,
            value_dim=config.value_dim,
            **(hyperparameters or {}),
        )
        with torch.no_grad():
            result = variant(task.keys, task.keys, task.values, gates=gates)
            recall = torch.einsum("bhv,blh->blv", result.final_state, task.keys)
            cosine = F.cosine_similarity(recall, task.values, dim=-1, eps=_EPS)
        success_parts.append(
            (cosine >= config.failure_threshold).to(torch.float64).numpy()
        )
        batch, seq_len = task.keys.shape[:2]
        group_parts.append(
            (np.arange(batch) + seed_index * batch).repeat(seq_len)
        )
        cosine_parts.append(cosine.numpy())

    success = np.concatenate([p.reshape(-1) for p in success_parts])
    groups = np.concatenate(group_parts)
    cosines = np.concatenate([p.reshape(-1) for p in cosine_parts])
    return {
        "num_items": int(success.size),
        "recall_accuracy": {
            "value": float(success.mean()),
            "ci": list(
                bootstrap_ci(
                    (success,),
                    groups,
                    lambda lab: float(lab.mean()),
                    resamples=config.bootstrap_resamples,
                    level=config.ci_level,
                    seed=config.bootstrap_seed,
                )
            ),
        },
        "mean_recall_cosine": float(cosines.mean()),
    }


def run_precision_gated_ablation(
    config: PrecisionGatedAblationConfig,
) -> dict:
    """Run the pre-registered #33 hyperparameter ablation.

    For each operating point, every reference variant and every grid cell is
    evaluated on identical task/gate draws. The default cell
    ``(lambda_=1.0, forgetting=1.0, process_noise=0.0)`` coincides with the
    #34 precision-gated rows; all other cells are one-to-three-axis
    perturbations of it. No cell is filtered from the report.

    Returns:
        A JSON-serializable dict embedding the full config snapshot.
    """

    cells = ablation_cells(config)
    report = {
        "experiment": "phase4-precision-gated-ablation",
        "issue": 33,
        "description": (
            "Hyperparameter ablation of the diagonal precision-gated KDA "
            "variant (#33): recall accuracy and mean recall cosine over the "
            "pre-registered (lambda, forgetting, process_noise) grid at "
            "pre-registered task operating points, against fixed-beta KDA "
            "reference rows on identical draws. The default cell reproduces "
            "the #34 precision-gated rows. Pre-registered in the embedded "
            "config; CIs are cluster bootstrap over sequences; no cell is "
            "filtered, so collapsed or flat settings are reported as-is."
        ),
        "config": config.to_dict(),
        "default_cell": {
            "lambda_": DEFAULT_LAMBDA,
            "forgetting": DEFAULT_FORGETTING,
            "process_noise": DEFAULT_PROCESS_NOISE,
        },
        "operating_points": [],
    }
    for num_pairs, collision_cosine in config.operating_points:
        point_report = {
            "num_pairs": num_pairs,
            "collision_cosine": collision_cosine,
            "reference": {
                variant_name: _collect_point(
                    variant_name, None, num_pairs, collision_cosine, config
                )
                for variant_name in config.reference_variants
            },
            "cells": [
                {
                    **cell,
                    "is_default": is_default_cell(cell),
                    **_collect_point(
                        "precision-gated-kda",
                        cell,
                        num_pairs,
                        collision_cosine,
                        config,
                    ),
                }
                for cell in cells
            ],
        }
        report["operating_points"].append(point_report)
    return report
