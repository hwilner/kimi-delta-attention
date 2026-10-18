"""Synthetic tests for the adaptive hybrid-routing prototype (issue #38).

Covers the exact-attention fallback oracle, routing-signal computation from
captured traces (no reruns), policy semantics (threshold, budget cap,
validation), the deterministic fixed-ratio baseline, the audited hybrid
readout, operating-point bookkeeping, and a smoke run of the pre-registered
experiment — including honest reporting of unavailable/flat signals.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kda.experiments import (
    RoutingExperimentConfig,
    evaluate_operating_point,
    run_routing_experiment,
)
from kda.routing import (
    ROUTING_SIGNALS,
    RoutingPolicy,
    compute_routing_signals,
    exact_attention_readout,
    fixed_ratio_decisions,
    route_hybrid_readout,
)

SEED = 0


def _orthogonal_batch(batch: int, seq_len: int, dim: int):
    """Deterministic orthonormal-ish keys and random values.

    Keys are random and (near-)orthogonal for ``seq_len <= dim``; with such
    keys the exact softmax readout is essentially perfect while collision
    scores are near zero — handy for isolating router mechanics.
    """

    generator = torch.Generator().manual_seed(SEED)
    keys = F.normalize(torch.randn(batch, seq_len, dim, generator=generator), dim=-1)
    values = torch.randn(batch, seq_len, dim, generator=generator)
    gates = {
        "alpha": torch.full((batch, seq_len, dim), 0.9),
        "beta": torch.full((batch, seq_len, 1), 0.5),
    }
    return keys, values, gates


# ---------------------------------------------------------------------------
# Exact fallback oracle
# ---------------------------------------------------------------------------


def test_exact_attention_readout_matches_manual_softmax():
    keys, values, _ = _orthogonal_batch(2, 5, 8)
    queries = keys.clone()
    out = exact_attention_readout(queries, keys, values, temperature=0.1)
    logits = torch.matmul(queries, keys.transpose(1, 2)) / 0.1
    expected = torch.matmul(torch.softmax(logits, dim=-1), values)
    assert torch.allclose(out, expected, atol=1e-6)
    with pytest.raises(ValueError, match="temperature"):
        exact_attention_readout(queries, keys, values, temperature=0.0)


def test_exact_readout_recovers_values_for_orthogonal_keys():
    # For orthogonal keys the softmax weight on the matching key dominates.
    keys, values, _ = _orthogonal_batch(2, 6, 32)
    out = exact_attention_readout(keys, keys, values)
    cos = F.cosine_similarity(out, values, dim=-1)
    assert (cos > 0.95).all()


def test_exact_attention_readout_validates_shapes():
    keys, values, _ = _orthogonal_batch(1, 4, 8)
    with pytest.raises(ValueError, match="rank-3"):
        exact_attention_readout(keys[0], keys, values)
    with pytest.raises(ValueError, match="batch and sequence"):
        exact_attention_readout(keys, keys, values[:, :2])


# ---------------------------------------------------------------------------
# Policy semantics
# ---------------------------------------------------------------------------


def test_policy_validation():
    with pytest.raises(ValueError, match="unknown routing signal"):
        RoutingPolicy(signal="no-such-signal", threshold=0.0)
    with pytest.raises(ValueError, match="finite"):
        RoutingPolicy(signal="key_collision_score", threshold=float("inf"))
    with pytest.raises(ValueError, match="max_fallback_rate"):
        RoutingPolicy(signal="key_collision_score", threshold=0.0, max_fallback_rate=1.5)


def test_policy_threshold_extremes_route_all_or_none():
    scores = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.4, 0.6]])
    low = RoutingPolicy(signal="key_collision_score", threshold=0.05)
    high = RoutingPolicy(signal="key_collision_score", threshold=0.95)
    assert low.apply(scores).all()
    assert not high.apply(scores).any()


def test_policy_budget_cap_keeps_top_scores():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    policy = RoutingPolicy(
        signal="key_collision_score", threshold=0.0, max_fallback_rate=0.5
    )
    mask = policy.apply(scores)
    assert mask.tolist() == [[True, True, False, False]]
    zero_budget = RoutingPolicy(
        signal="key_collision_score", threshold=0.0, max_fallback_rate=0.0
    )
    assert not zero_budget.apply(scores).any()


def test_fixed_ratio_decisions_deterministic_and_exact_count():
    mask_a = fixed_ratio_decisions(100, 0.3, seed=7)
    mask_b = fixed_ratio_decisions(100, 0.3, seed=7)
    assert np.array_equal(mask_a, mask_b)
    assert mask_a.sum() == 30
    assert fixed_ratio_decisions(10, 0.0).sum() == 0
    assert fixed_ratio_decisions(10, 1.0).all()
    with pytest.raises(ValueError, match="ratio"):
        fixed_ratio_decisions(10, 1.1)


# ---------------------------------------------------------------------------
# Signal computation and hybrid readout
# ---------------------------------------------------------------------------


def test_routing_signals_collision_score_known_value():
    # Two keys at an exact cosine of 0.9 (built via an orthogonalized noise
    # component): each key's collision score must equal that cosine.
    keys, values, gates = _orthogonal_batch(1, 2, 16)
    keys = keys.clone()
    generator = torch.Generator().manual_seed(SEED + 1)
    k0 = keys[0, 0]
    noise = torch.randn(16, generator=generator)
    orthogonal = F.normalize(noise - (noise * k0).sum() * k0, dim=-1)
    keys[0, 1] = 0.9 * k0 + (1 - 0.9**2) ** 0.5 * orthogonal
    raw, _ = compute_routing_signals("kda", keys, values, gates)
    collision = raw["key_collision_score"]
    assert collision.shape == (1, 2)
    assert collision[0, 0].item() == pytest.approx(0.9, abs=1e-4)
    assert collision[0, 1].item() == pytest.approx(0.9, abs=1e-4)


def test_routing_signals_availability_marks_nan():
    keys, values, gates = _orthogonal_batch(1, 4, 8)
    raw, _ = compute_routing_signals("kda", keys, values, gates)
    # Baseline KDA exposes neither evidence nor precision.
    assert torch.isnan(raw["evidence_along_key"]).all()
    assert torch.isnan(raw["precision_along_key"]).all()
    assert torch.isfinite(raw["prediction_error_norm"]).all()
    raw, _ = compute_routing_signals("precision-gated-kda", keys, values, gates)
    assert torch.isfinite(raw["precision_along_key"]).all()


def test_hybrid_readout_routed_queries_match_exact_unrouted_match_compressed():
    keys, values, gates = _orthogonal_batch(2, 6, 16)
    policy = RoutingPolicy(signal="prediction_error_norm", threshold=0.0)
    result = route_hybrid_readout("kda", keys, values, gates, policy)
    assert result.routed.all()  # every write has nonzero prediction error
    assert torch.allclose(result.outputs, result.exact_recall, atol=1e-6)
    none = RoutingPolicy(signal="prediction_error_norm", threshold=1e9)
    result_none = route_hybrid_readout("kda", keys, values, gates, none)
    assert not result_none.routed.any()
    assert result_none.fallback_rate == 0.0
    assert torch.allclose(result_none.outputs, result_none.compressed_recall, atol=1e-6)


def test_hybrid_readout_audit_log_covers_every_query():
    keys, values, gates = _orthogonal_batch(2, 3, 8)
    policy = RoutingPolicy(signal="key_collision_score", threshold=0.5, name="test")
    result = route_hybrid_readout("kda", keys, values, gates, policy)
    assert len(result.audit_log) == 2 * 3
    for entry in result.audit_log:
        assert set(entry) >= {
            "batch", "query", "policy", "signal", "score", "threshold",
            "routed_to_exact",
        }
        assert entry["policy"] == "test"
        assert entry["routed_to_exact"] == (entry["score"] >= entry["threshold"])
    json.dumps(result.audit_log)  # audit log must stay JSON-serializable


def test_hybrid_readout_external_decisions_override_policy():
    keys, values, gates = _orthogonal_batch(1, 4, 8)
    policy = RoutingPolicy(signal="key_collision_score", threshold=1e9)
    decisions = torch.tensor([[True, False, True, False]])
    result = route_hybrid_readout(
        "kda", keys, values, gates, policy, decisions=decisions
    )
    assert result.routed.tolist() == decisions.tolist()
    assert result.fallback_rate == pytest.approx(0.5)
    with pytest.raises(ValueError, match="decisions must match"):
        route_hybrid_readout(
            "kda", keys, values, gates, policy, decisions=torch.zeros(1, 2)
        )


def test_hybrid_readout_rejects_unavailable_or_unaligned_signals():
    keys, values, gates = _orthogonal_batch(1, 4, 8)
    with pytest.raises(ValueError, match="exposes no 'evidence_along_key'"):
        route_hybrid_readout(
            "kda", keys, values, gates,
            RoutingPolicy(signal="evidence_along_key", threshold=0.0),
        )
    shuffled = keys[:, [1, 0, 2, 3]]
    with pytest.raises(ValueError, match="aligned"):
        route_hybrid_readout(
            "kda", keys, values, gates,
            RoutingPolicy(signal="prediction_error_norm", threshold=0.0),
            queries=shuffled,
        )
    # Non-aligned queries are allowed for the collision signal.
    result = route_hybrid_readout(
        "kda", keys, values, gates,
        RoutingPolicy(signal="key_collision_score", threshold=0.5),
        queries=shuffled,
    )
    assert result.outputs.shape == (1, 4, 8)


# ---------------------------------------------------------------------------
# Operating-point evaluation
# ---------------------------------------------------------------------------


def test_operating_point_bookkeeping_sums():
    rng = np.random.default_rng(0)
    n = 200
    compressed_ok = rng.random(n) < 0.6
    exact_ok = rng.random(n) < 0.95
    scores = rng.normal(size=n)
    point = evaluate_operating_point(scores, compressed_ok, exact_ok, 0.0)
    routed = scores >= 0.0
    assert point["num_routed"] == int(routed.sum())
    assert point["num_items"] == n
    assert point["fallback_rate"] == pytest.approx(routed.mean())
    # Overuse and underuse partition the two error modes of the router.
    assert point["overuse_rate"] == pytest.approx(compressed_ok[routed].mean())
    assert point["underuse_rate"] == pytest.approx((~compressed_ok[~routed]).mean())
    # Expected accuracy under the routed mixture.
    expected = np.where(routed, exact_ok, compressed_ok).mean()
    assert point["recall_accuracy"] == pytest.approx(expected)
    assert point["improvement_over_compressed"] == pytest.approx(
        expected - compressed_ok.mean()
    )


def test_operating_point_perfect_signal_beats_fixed_ratio():
    # A perfect oracle signal (score == failure indicator) routed at the
    # failure rate must capture every failure; random routing cannot (w.h.p.
    # at this size) — the adaptive-vs-fixed gap is the signal's value.
    n = 400
    compressed_ok = np.zeros(n, dtype=bool)
    compressed_ok[: n // 2] = True  # first half fine, second half fails
    exact_ok = np.ones(n, dtype=bool)
    scores = (~compressed_ok).astype(float) + 1e-6 * np.arange(n) / n
    point = evaluate_operating_point(
        scores, compressed_ok, exact_ok, 0.5, fixed_ratio_seed=3
    )
    assert point["recall_accuracy"] == pytest.approx(1.0)
    assert point["gain_vs_fixed_ratio"] > 0.2
    assert point["overuse_rate"] == pytest.approx(0.0)
    assert point["underuse_rate"] == pytest.approx(0.0)


def test_operating_point_validation():
    with pytest.raises(ValueError, match="agree in shape"):
        evaluate_operating_point(
            np.zeros(3), np.zeros(4, dtype=bool), np.zeros(3, dtype=bool), 0.0
        )


# ---------------------------------------------------------------------------
# Pre-registered experiment (smoke)
# ---------------------------------------------------------------------------


def _tiny_config(**overrides):
    kwargs = dict(
        seeds=(0,),
        batch_size=2,
        num_pairs=4,
        key_dim=16,
        value_dim=16,
        bootstrap_resamples=None,  # not a field; filtered below
        threshold_quantiles=(0.5, 0.9),
    )
    kwargs.pop("bootstrap_resamples")
    kwargs.update(overrides)
    return RoutingExperimentConfig(**kwargs)


def test_experiment_smoke_report_structure_and_honest_negatives():
    report = run_routing_experiment(_tiny_config())
    json.dumps(report)  # whole report must be JSON-serializable
    assert report["experiment"] == "phase4-hybrid-routing"
    assert report["issue"] == 38
    assert report["config"]["signals"] == list(_tiny_config().signals)
    for variant_name in report["config"]["variants"]:
        section = report["variants"][variant_name]
        assert 0.0 <= section["compressed_accuracy"] <= 1.0
        assert section["exact_accuracy"] >= section["compressed_accuracy"] - 1e-9
        # gate_entropy is pre-registered even though #28 found it flat here;
        # it must appear in the report, not be filtered out.
        assert "gate_entropy" in section["signals"]
        for signal_name, entry in section["signals"].items():
            if not entry["available"]:
                assert "reason" in entry  # unavailable signals reported, not dropped
                continue
            points = entry["operating_points"]
            assert len(points) == len(_tiny_config().threshold_quantiles)
            for point in points:
                assert 0.0 <= point["fallback_rate"] <= 1.0
                assert 0.0 <= point["recall_accuracy"] <= 1.0
                assert 0.0 <= point["fixed_ratio_accuracy"] <= 1.0


def test_experiment_exact_fallback_recovers_failures_at_full_budget():
    # With everything routed (lowest quantile threshold on a constant signal
    # range), accuracy must reach the exact-attention endpoint.
    report = run_routing_experiment(_tiny_config(threshold_quantiles=(0.0,)))
    for section in report["variants"].values():
        point = section["signals"]["key_collision_score"]["operating_points"][0]
        assert point["fallback_rate"] == pytest.approx(1.0)
        assert point["recall_accuracy"] == pytest.approx(section["exact_accuracy"])


def test_config_validation():
    with pytest.raises(ValueError, match="unknown signals"):
        _tiny_config(signals=("bogus",))
    with pytest.raises(ValueError, match="threshold_quantiles"):
        _tiny_config(threshold_quantiles=(1.5,))
    with pytest.raises(ValueError, match="unknown variants"):
        _tiny_config(variants=("not-a-variant",))
