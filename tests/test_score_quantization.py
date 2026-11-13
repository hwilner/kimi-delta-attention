"""Tests for score quantization (Phase 4 report reproducibility).

Several collected scores are near-tied *by construction* (key-collision
scores within a collision group, histogram gate entropies, effective ranks
of low-write-count states).  Rank-based statistics over such scores — AUROC
midranks, quantile thresholds — are otherwise decided by float32 ulp noise
(~1e-7) whose sign pattern depends on the BLAS/LAPACK build, so the affected
report cells did not reproduce across environments.  ``quantize_scores``
rounds scores to a fixed decimal grid at collection time, turning near-ties
into exact ties.  These tests pin the helper, the invariance it buys, and
the fact that both collection paths (calibration #28, routing #38) emit
quantized scores.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from kda.experiments import (
    CalibrationConfig,
    RoutingExperimentConfig,
    binary_auroc,
    collect_routing_data,
    collect_samples_for_batch,
    quantize_scores,
)
from kda.experiments.calibration import SCORE_QUANTIZATION_DECIMALS
from kda.tasks import KeyCollisionTaskConfig, generate_key_collision_batch


def _tiny_task(batch_size=2, num_pairs=4, key_dim=16, value_dim=8, seed=0, cosine=0.9):
    return generate_key_collision_batch(
        KeyCollisionTaskConfig(
            batch_size=batch_size,
            num_pairs=num_pairs,
            key_dim=key_dim,
            value_dim=value_dim,
            collision_cosine=cosine,
            seed=seed,
            shuffle_queries=False,
        )
    )


def _constant_gates(batch, seq_len, key_dim, alpha=0.9, beta=0.8):
    return {
        "alpha": torch.full((batch, seq_len, key_dim), alpha),
        "beta": torch.full((batch, seq_len, 1), beta),
    }


def _on_grid(values: np.ndarray) -> bool:
    finite = np.isfinite(values)
    return bool(
        np.allclose(
            values[finite],
            np.round(values[finite], SCORE_QUANTIZATION_DECIMALS),
            atol=0.0,
        )
    )


# ---------------------------------------------------------------------------
# quantize_scores helper
# ---------------------------------------------------------------------------


def test_quantize_scores_merges_ulp_near_ties():
    # A nominal value plus a few float32 ulps of build-dependent noise must
    # collapse to one exact grid point.
    nominal = np.full(64, 0.5)
    rng = np.random.default_rng(0)
    noisy = nominal + rng.uniform(-3e-7, 3e-7, size=nominal.shape)
    assert np.unique(noisy).size > 1  # fixture really is near-tied, not tied
    quantized = quantize_scores(noisy)
    assert np.unique(quantized).size == 1
    assert quantized[0] == 0.5


def test_quantize_scores_preserves_well_separated_values():
    # Differences far above the quantum are preserved exactly on the grid.
    scores = np.array([0.123456789, 0.123457789, 0.9, -0.25, 2.5])
    quantized = quantize_scores(scores)
    assert np.all(np.diff(quantized) != 0)
    assert np.max(np.abs(quantized - scores)) <= 0.5 * 10 ** -SCORE_QUANTIZATION_DECIMALS
    assert _on_grid(quantized)


def test_quantize_scores_nan_passthrough_and_float64():
    scores = np.array([np.nan, 0.25, np.nan], dtype=np.float32)
    quantized = quantize_scores(scores)
    assert quantized.dtype == np.float64
    assert np.isnan(quantized[0]) and np.isnan(quantized[2])
    assert quantized[1] == 0.25
    # Idempotent: quantizing twice changes nothing.
    assert np.array_equal(
        quantize_scores(quantized)[~np.isnan(quantized)],
        quantized[~np.isnan(quantized)],
    )


def test_auroc_invariant_to_ulp_noise_after_quantization():
    # The regression this module fixes: with near-tied score blocks and mixed
    # labels inside a block, unquantized AUROC depends on the ulp-level
    # ordering (build-dependent); quantized AUROC is invariant.
    rng = np.random.default_rng(1)
    block_sizes = [100, 48, 22, 9]
    nominals = [0.5, 0.25, 0.9, 0.1]
    base = np.concatenate([np.full(n, v) for n, v in zip(block_sizes, nominals)])
    labels = np.concatenate(
        [rng.integers(0, 2, size=n).astype(np.float64) for n in block_sizes]
    )
    # Fixture sanity: every block mixes failures and successes, so the
    # within-block ordering really matters for AUROC.
    offset = 0
    for n in block_sizes:
        block_labels = labels[offset : offset + n]
        assert 0 < block_labels.sum() < n
        offset += n

    unquantized, quantized = set(), set()
    for trial in range(25):
        noise = rng.uniform(-3e-7, 3e-7, size=base.shape)
        perturbed = base + noise
        unquantized.add(binary_auroc(perturbed, labels))
        quantized.add(binary_auroc(quantize_scores(perturbed), labels))
    assert len(unquantized) > 1  # the fragility is real on this fixture
    assert len(quantized) == 1  # and quantization removes it
    # The invariant value equals the exact-tie AUROC of the nominal scores.
    assert quantized.pop() == binary_auroc(base, labels)


# ---------------------------------------------------------------------------
# Collection paths emit quantized scores
# ---------------------------------------------------------------------------


def test_collected_calibration_signals_and_cosines_are_on_grid():
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16)
    config = CalibrationConfig(
        seeds=(0,), batch_size=2, num_pairs=4, key_dim=16, value_dim=8,
        bootstrap_resamples=10,
    )
    for variant_name in ("kda", "kda-rls-lite", "precision-gated-kda"):
        samples = collect_samples_for_batch(
            variant_name, task.keys, task.values, gates, config
        )
        assert _on_grid(samples.recall_cosines), variant_name
        for name, scores in samples.signals.items():
            assert _on_grid(scores), (variant_name, name)
        # Labels agree with thresholding the quantized cosines.
        expected = (
            samples.recall_cosines < config.failure_threshold
        ).astype(np.float64)
        assert np.array_equal(samples.labels, expected)


def test_quantization_flips_no_labels_on_tiny_task():
    # Guard the "no item near the threshold" invariant at the tiny scale:
    # thresholding raw vs quantized cosines must agree here.
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16)
    config = CalibrationConfig(
        seeds=(0,), batch_size=2, num_pairs=4, key_dim=16, value_dim=8,
        bootstrap_resamples=10,
    )
    from kda.variants import create_variant

    variant = create_variant("kda", head_dim=16, value_dim=8)
    with torch.no_grad():
        result = variant(task.keys, task.keys, task.values, gates=gates)
        recall = torch.einsum("bhv,blh->blv", result.final_state, task.keys)
        cosine = F.cosine_similarity(recall, task.values, dim=-1, eps=1e-8).numpy()
    raw_labels = (cosine < config.failure_threshold)
    quantized_labels = (quantize_scores(cosine) < config.failure_threshold)
    assert np.array_equal(raw_labels, quantized_labels)
    # And nothing sits within 1e-4 of the threshold on this fixture.
    assert np.min(np.abs(cosine - config.failure_threshold)) > 1e-4


def test_collected_routing_scores_are_on_grid():
    config = RoutingExperimentConfig(
        seeds=(0,), batch_size=2, num_pairs=4, key_dim=16, value_dim=8
    )
    data = collect_routing_data("kda", config)
    for name, scores in data.scores.items():
        assert _on_grid(scores), name
    # The default-config collision score is exactly constant (every key
    # collides at the same constructed cosine), so the routing threshold
    # grid is degenerate for it — the deterministic behavior the fix pins.
    collision = data.scores["key_collision_score"]
    assert np.unique(collision).size == 1


def test_routing_thresholds_invariant_to_ulp_noise_after_quantization():
    # Quantile thresholds and realized fallback rates on quantized near-tied
    # scores no longer depend on the ulp-level ordering.
    rng = np.random.default_rng(2)
    base = np.concatenate([np.full(100, 0.5), np.full(60, 0.25), np.full(32, 0.9)])
    quantiles = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    reference_thresholds = np.quantile(quantize_scores(base), quantiles)
    reference_rates = np.array(
        [(quantize_scores(base) >= t).mean() for t in reference_thresholds]
    )
    for _ in range(10):
        perturbed = base + rng.uniform(-3e-7, 3e-7, size=base.shape)
        scores = quantize_scores(perturbed)
        thresholds = np.quantile(scores, quantiles)
        rates = np.array([(scores >= t).mean() for t in thresholds])
        assert np.array_equal(thresholds, reference_thresholds)
        assert np.array_equal(rates, reference_rates)
