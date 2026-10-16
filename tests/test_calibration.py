"""Tests for the calibration experiment module (Phase 4, issue #28)."""

import json
import math

import numpy as np
import pytest
import torch

from kda.experiments import (
    CalibrationConfig,
    SIGNAL_SPECS,
    average_precision,
    binary_auroc,
    binned_recall,
    bootstrap_ci,
    calibration_curve,
    collect_samples_for_batch,
    expected_calibration_error,
    run_calibration_experiment,
)
from kda.experiments.calibration import (
    _per_step_gate_entropy,
    _replayed_prediction_error_norms,
    _write_vectors,
)
from kda.instrumentation import replay_state_trajectory
from kda.tasks import KeyCollisionTaskConfig, generate_key_collision_batch
from kda.variants import create_variant


def _tiny_config(**overrides):
    """Fast deterministic config for end-to-end tests."""
    base = dict(
        seeds=(0, 1),
        batch_size=4,
        num_pairs=4,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=50,
    )
    base.update(overrides)
    return CalibrationConfig(**base)


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


# ---------------------------------------------------------------------------
# CalibrationConfig (pre-registration)
# ---------------------------------------------------------------------------


def test_config_snapshot_is_json_serializable():
    config = _tiny_config()
    snapshot = config.to_dict()
    assert json.loads(json.dumps(snapshot)) == snapshot
    # Tuples become lists in the snapshot (JSON-safe).
    assert snapshot["variants"] == ["kda", "kda-rls-lite", "precision-gated-kda"]
    assert "auroc" in snapshot["metrics"]


def test_config_is_frozen():
    config = _tiny_config()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        config.failure_threshold = 0.9  # type: ignore[misc]


def test_config_validation():
    with pytest.raises(ValueError, match="failure_threshold"):
        _tiny_config(failure_threshold=2.0)
    with pytest.raises(ValueError, match="unknown variants"):
        _tiny_config(variants=("not-a-variant",))
    with pytest.raises(ValueError, match="unknown signals"):
        _tiny_config(signals=("vibes",))
    with pytest.raises(ValueError, match="unknown metrics"):
        _tiny_config(metrics=("p-value",))
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        _tiny_config(bootstrap_resamples=0)
    with pytest.raises(ValueError, match="ci_level"):
        _tiny_config(ci_level=1.5)
    with pytest.raises(ValueError, match="collision_cosine"):
        _tiny_config(collision_cosine=1.5)
    with pytest.raises(ValueError, match="seeds"):
        _tiny_config(seeds=())


# ---------------------------------------------------------------------------
# binary_auroc
# ---------------------------------------------------------------------------


def test_auroc_hand_computed():
    # Classic example: failures score 0.35 and 0.8, successes 0.1 and 0.4 —
    # exactly one of the four failure/success pairs is mis-ordered.
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert binary_auroc(scores, labels) == pytest.approx(0.75)


def test_auroc_extremes_and_ties():
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert binary_auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)
    # Constant score: every pair ties at 0.5.
    assert binary_auroc(np.full(4, 0.3), labels) == pytest.approx(0.5)


def test_auroc_single_class_is_nan():
    assert math.isnan(binary_auroc(np.array([0.1, 0.2]), np.zeros(2)))
    assert math.isnan(binary_auroc(np.array([0.1, 0.2]), np.ones(2)))


def test_auroc_rejects_bad_input():
    with pytest.raises(ValueError, match="binary"):
        binary_auroc(np.array([0.1]), np.array([0.5]))
    with pytest.raises(ValueError, match="agree"):
        binary_auroc(np.array([0.1, 0.2]), np.array([0.0]))


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


def test_average_precision_hand_computed():
    # Sorted by score: labels [1, 0, 1, 0]; P@1 = 1, P@3 = 2/3;
    # AP = (1/2)*1 + (1/2)*(2/3) = 5/6.
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert average_precision(scores, labels) == pytest.approx(5.0 / 6.0)


def test_average_precision_perfect_and_nan():
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert average_precision(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert math.isnan(average_precision(np.array([0.1, 0.2]), np.zeros(2)))


# ---------------------------------------------------------------------------
# expected_calibration_error / calibration_curve
# ---------------------------------------------------------------------------


def test_ece_constant_score_conventions():
    # A constant score maps to probability 0 for every item (documented
    # min-max convention), so ECE equals the empirical failure rate.
    assert expected_calibration_error(np.full(4, 0.7), np.zeros(4)) == pytest.approx(0.0)
    assert expected_calibration_error(np.full(4, 0.7), np.ones(4)) == pytest.approx(1.0)
    assert expected_calibration_error(
        np.full(4, 0.7), np.array([0.0, 1.0, 0.0, 1.0])
    ) == pytest.approx(0.5)


def test_ece_hand_computed():
    # scores -> min-max probabilities [0, 1/8, 7/8, 1]; bins=2:
    # bin 0 (prob < 0.5): items 0,1 -> failure rate 0, mean prob (0+1/8)/2 = 1/16
    # bin 1: items 2,3 -> failure rate 1, mean prob (7/8+1)/2 = 15/16
    # ECE = 0.5*|0 - 1/16| + 0.5*|1 - 15/16| = 1/32 + 1/32 = 1/16.
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(scores, labels, bins=2) == pytest.approx(1.0 / 16.0)


def test_calibration_curve_known():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    curve = calibration_curve(scores, labels, bins=2)
    assert [entry["count"] for entry in curve] == [2, 2]
    assert curve[0]["empirical_failure_rate"] == pytest.approx(0.0)
    assert curve[1]["empirical_failure_rate"] == pytest.approx(1.0)
    assert curve[1]["mean_predicted_failure_prob"] == pytest.approx(15.0 / 16.0)


# ---------------------------------------------------------------------------
# binned_recall
# ---------------------------------------------------------------------------


def test_binned_recall_known_bins():
    scores = np.arange(10, dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
    table = binned_recall(scores, labels, num_bins=2)
    assert len(table) == 2
    assert [entry["count"] for entry in table] == [5, 5]
    assert table[0]["recall_accuracy"] == pytest.approx(1.0)
    assert table[1]["recall_accuracy"] == pytest.approx(0.0)
    assert table[1]["failure_rate"] == pytest.approx(1.0)


def test_binned_recall_constant_score_single_bin():
    table = binned_recall(np.full(6, 0.4), np.array([0, 0, 0, 1, 1, 1], dtype=np.float64))
    assert len(table) == 1
    assert table[0]["count"] == 6
    assert table[0]["recall_accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic_and_covers_estimate():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(10), 5)
    scores = rng.normal(size=50)
    labels = (scores + rng.normal(scale=0.5, size=50) > 0).astype(float)
    ci_a = bootstrap_ci((scores, labels), groups, binary_auroc, resamples=100, seed=7)
    ci_b = bootstrap_ci((scores, labels), groups, binary_auroc, resamples=100, seed=7)
    assert ci_a == ci_b  # deterministic given the seed
    estimate = binary_auroc(scores, labels)
    assert ci_a[0] <= estimate <= ci_a[1]
    assert ci_a[0] < ci_a[1]  # non-degenerate interval


def test_bootstrap_ci_single_class_resamples_dropped():
    # Only one positive in four groups: some resamples are single-class and
    # must be dropped rather than poisoning the interval.
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    labels = np.array([0.0, 0.0, 0.0, 1.0])
    groups = np.arange(4)
    low, high = bootstrap_ci((scores, labels), groups, binary_auroc, resamples=50, seed=0)
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_ci_validation():
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ci((np.zeros(3),), np.arange(3), lambda s: 0.0, resamples=0)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci((np.zeros(3),), np.arange(3), lambda s: 0.0, level=1.0)
    with pytest.raises(ValueError, match="first axis"):
        bootstrap_ci((np.zeros(4),), np.arange(3), lambda s: 0.0)


# ---------------------------------------------------------------------------
# collect_samples_for_batch: labels, signals, alignment
# ---------------------------------------------------------------------------


def test_beta_zero_writes_nothing_so_every_query_fails():
    # beta = 0 freezes the KDA / KDA-RLS-lite state at zero, so recall is the
    # zero vector, cosine 0 < threshold, and every query is a failure.
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16, alpha=0.9, beta=0.0)
    config = _tiny_config()
    for name in ("kda", "kda-rls-lite"):
        samples = collect_samples_for_batch(name, task.keys, task.values, gates, config)
        assert samples.labels.mean() == 1.0
        assert np.all(samples.recall_cosines == 0.0)
        assert samples.labels.shape == (2 * 8,)
        assert samples.groups.tolist() == sorted(samples.groups.tolist())


def test_signal_availability_by_variant():
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16)
    config = _tiny_config()
    samples = collect_samples_for_batch("kda", task.keys, task.values, gates, config)
    assert np.all(np.isnan(samples.signals["evidence_along_key"]))
    assert np.all(np.isnan(samples.signals["precision_along_key"]))
    for shared in ("prediction_error_norm", "key_collision_score", "gate_entropy",
                   "state_effective_rank"):
        assert np.all(np.isfinite(samples.signals[shared])), shared

    rls = collect_samples_for_batch("kda-rls-lite", task.keys, task.values, gates, config)
    assert np.all(np.isfinite(rls.signals["evidence_along_key"]))
    assert np.all(np.isnan(rls.signals["precision_along_key"]))

    gated = collect_samples_for_batch(
        "precision-gated-kda", task.keys, task.values, gates, config
    )
    assert np.all(np.isfinite(gated.signals["precision_along_key"]))
    assert np.all(np.isnan(gated.signals["evidence_along_key"]))


def test_evidence_signal_is_negated_occupancy():
    # Sign convention: score = sign * raw with sign=-1 for evidence, so the
    # uncertainty score is the negated trace occupancy.
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16)
    config = _tiny_config()
    samples = collect_samples_for_batch("kda-rls-lite", task.keys, task.values, gates, config)
    variant = create_variant("kda-rls-lite", head_dim=16, value_dim=8)
    with torch.no_grad():
        result = variant(task.keys, task.keys, task.values, gates=gates, return_trace=True)
    raw = result.trace["evidence_along_key"].squeeze(-1).numpy().reshape(-1)
    assert np.allclose(samples.signals["evidence_along_key"], -raw)


def test_replayed_errors_match_precision_gated_trace():
    # The replayed prediction-error signal is identical by construction to the
    # variant's captured prediction_error trace.
    task = _tiny_task(seed=3)
    gates = _constant_gates(2, 8, 16)
    variant = create_variant("precision-gated-kda", head_dim=16, value_dim=8)
    with torch.no_grad():
        result = variant(task.keys, task.keys, task.values, gates=gates, return_trace=True)
        write_vectors = _write_vectors(result.trace, task.keys)
        states = replay_state_trajectory(task.keys, task.values, gates["alpha"], write_vectors)
    replayed = _replayed_prediction_error_norms(task.keys, task.values, gates["alpha"], states)
    captured = torch.linalg.vector_norm(result.trace["prediction_error"], dim=-1)
    assert torch.allclose(replayed, captured, atol=1e-5)
    # The replayed final state also matches the variant's final state.
    assert torch.allclose(states[:, -1], result.final_state, atol=1e-5)


def test_collision_score_includes_constructed_pair():
    # Each base key's max-similarity is at least the constructed pair cosine.
    task = _tiny_task(cosine=0.9)
    gates = _constant_gates(2, 8, 16)
    samples = collect_samples_for_batch("kda", task.keys, task.values, gates, _tiny_config())
    scores = samples.signals["key_collision_score"].reshape(2, 8)
    assert np.all(scores >= 0.9 - 1e-5)
    # Base key i sits at position 2*i and its colliding partner at 2*i+1:
    # their mutual similarity is exactly the configured cosine.
    pair_cos = task.collision_cosines[:, 0, 1].numpy()
    assert np.allclose(scores[:, 0], np.maximum(pair_cos, scores[:, 0]))


def test_gate_entropy_zero_for_constant_alpha():
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16, alpha=0.9, beta=0.8)  # constant alpha
    samples = collect_samples_for_batch("kda", task.keys, task.values, gates, _tiny_config())
    assert np.allclose(samples.signals["gate_entropy"], 0.0, atol=1e-6)
    # A varied alpha has strictly positive per-step entropy.
    varied = _per_step_gate_entropy(
        torch.sigmoid(torch.randn(2, 8, 16, generator=torch.Generator().manual_seed(0))),
        bins=20,
    )
    assert (varied > 0).all()


def test_effective_rank_after_first_write_is_one():
    # After a single rank-1 write the state has one nonzero singular value.
    task = _tiny_task()
    gates = _constant_gates(2, 8, 16, alpha=1.0, beta=1.0)
    samples = collect_samples_for_batch("kda", task.keys, task.values, gates, _tiny_config())
    first_step = samples.signals["state_effective_rank"].reshape(2, 8)[:, 0]
    assert np.allclose(first_step, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# End-to-end experiment
# ---------------------------------------------------------------------------


def test_end_to_end_report_structure_and_preregistration():
    config = _tiny_config()
    report = run_calibration_experiment(config)
    # JSON-serializable, config embedded verbatim (pre-registration evidence).
    assert json.loads(json.dumps(report))["config"] == config.to_dict()
    assert report["issue"] == 28
    assert set(report["variants"]) == {"kda", "kda-rls-lite", "precision-gated-kda"}
    for name, section in report["variants"].items():
        assert section["num_items"] == 2 * 4 * 8  # seeds * batch * L
        assert 0.0 <= section["failure_rate"]["value"] <= 1.0
        assert set(section["signals"]) == set(config.signals)
        for signal, entry in section["signals"].items():
            if not entry["available"]:
                assert "reason" in entry
                continue
            for metric in ("auroc", "average_precision", "ece"):
                assert -0.05 <= entry[metric]["value"] <= 1.05
                low, high = entry[metric]["ci"]
                assert math.isnan(low) or low <= high
            assert len(entry["calibration_curve"]) <= config.ece_bins
            assert len(entry["recall_by_uncertainty_bin"]) <= config.recall_bins


def test_metric_subset_is_respected():
    config = _tiny_config(metrics=("auroc",))
    report = run_calibration_experiment(config)
    for section in report["variants"].values():
        for entry in section["signals"].values():
            if entry["available"]:
                assert "auroc" in entry
                assert "average_precision" not in entry
                assert "ece" not in entry


def test_deterministic_across_runs():
    config = _tiny_config(bootstrap_resamples=20)
    first = run_calibration_experiment(config)
    second = run_calibration_experiment(config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_positive_control_native_signals_predict_failure():
    # Pre-registered positive control on the default-regime task: the native
    # uncertainty signals (precision / evidence along the key) must separate
    # failures from successes well above chance. Measured at collection time
    # (2026-10-16): 0.88 (precision) and 0.79 (evidence); thresholds carry a
    # wide margin and the config is fully seeded, so this is deterministic.
    config = CalibrationConfig(seeds=(0, 1), bootstrap_resamples=50)
    report = run_calibration_experiment(config)
    gated = report["variants"]["precision-gated-kda"]["signals"]["precision_along_key"]
    rls = report["variants"]["kda-rls-lite"]["signals"]["evidence_along_key"]
    assert gated["auroc"]["value"] > 0.75
    assert rls["auroc"]["value"] > 0.65
    # Sanities: failure rates are in a discriminating regime (not saturated).
    for name in config.variants:
        rate = report["variants"][name]["failure_rate"]["value"]
        assert 0.1 < rate < 0.9


def test_signal_specs_cover_issue_list():
    # The five signal families named in issue #28 are all pre-registered.
    expected = {
        "evidence_along_key",
        "precision_along_key",
        "prediction_error_norm",
        "key_collision_score",
        "gate_entropy",
        "state_effective_rank",
    }
    assert expected <= set(SIGNAL_SPECS)
