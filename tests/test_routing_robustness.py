"""Tests for the routing robustness sweep (extends Phase 4 issue #38)."""

import dataclasses
import json

import numpy as np
import pytest

from kda.experiments import (
    RoutingExperimentConfig,
    RoutingRobustnessConfig,
    collect_routing_data,
    routing_regime_cells,
    run_routing_experiment,
    run_routing_robustness,
)


def _tiny_config(**overrides):
    """Fast deterministic robustness config for end-to-end tests."""
    base = dict(
        seeds=(0, 1),
        batch_size=2,
        num_pairs_grid=(2, 3),
        collision_cosine_grid=(0.5, 0.9),
        key_dim=16,
        value_dim=8,
    )
    base.update(overrides)
    return RoutingRobustnessConfig(**base)


def test_config_snapshot_is_json_serializable():
    config = RoutingRobustnessConfig()
    snapshot = json.loads(json.dumps(config.to_dict(), sort_keys=True))
    assert snapshot["variants"] == ["kda", "kda-rls-lite", "precision-gated-kda"]
    assert snapshot["seeds"] == [0, 1, 2]
    assert snapshot["num_pairs_grid"] == [4, 8, 16]
    assert snapshot["collision_cosine_grid"] == [0.5, 0.9, 0.99]
    assert snapshot["failure_threshold"] == 0.5
    assert snapshot["threshold_quantiles"] == [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    assert snapshot["fallback_temperature"] == 0.1


def test_config_is_frozen():
    config = _tiny_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.fixed_ratio_seed = 1


def test_config_validation():
    with pytest.raises(ValueError, match="unknown variants"):
        _tiny_config(variants=("kda", "not-a-variant"))
    with pytest.raises(ValueError, match="unknown signals"):
        _tiny_config(signals=("gate_entropy", "not-a-signal"))
    with pytest.raises(ValueError, match="seeds"):
        _tiny_config(seeds=())
    with pytest.raises(ValueError, match="num_pairs_grid"):
        _tiny_config(num_pairs_grid=())
    with pytest.raises(ValueError, match="num_pairs_grid"):
        _tiny_config(num_pairs_grid=(4, 0))
    with pytest.raises(ValueError, match="collision_cosine_grid"):
        _tiny_config(collision_cosine_grid=())
    with pytest.raises(ValueError, match="collision_cosine_grid"):
        _tiny_config(collision_cosine_grid=(0.5, 1.5))
    with pytest.raises(ValueError, match="batch_size"):
        _tiny_config(batch_size=0)
    with pytest.raises(ValueError, match="key_dim"):
        _tiny_config(key_dim=1)
    with pytest.raises(ValueError, match="failure_threshold"):
        _tiny_config(failure_threshold=2.0)
    with pytest.raises(ValueError, match="threshold_quantiles"):
        _tiny_config(threshold_quantiles=())
    with pytest.raises(ValueError, match="threshold_quantiles"):
        _tiny_config(threshold_quantiles=(0.5, 1.5))
    with pytest.raises(ValueError, match="entropy_bins"):
        _tiny_config(entropy_bins=0)
    with pytest.raises(ValueError, match="fallback_temperature"):
        _tiny_config(fallback_temperature=0.0)


def test_routing_regime_cells_order_and_inheritance():
    config = _tiny_config(seeds=(5, 6), batch_size=3, alpha_bias=1.5)
    cells = routing_regime_cells(config)
    # Cartesian product, outer loop over num_pairs, inner over cosines.
    assert [(c.num_pairs, c.collision_cosine) for c in cells] == [
        (2, 0.5),
        (2, 0.9),
        (3, 0.5),
        (3, 0.9),
    ]
    for cell in cells:
        assert isinstance(cell, RoutingExperimentConfig)
        # Non-swept fields are inherited unchanged.
        assert cell.seeds == (5, 6)
        assert cell.batch_size == 3
        assert cell.alpha_bias == 1.5
        assert cell.beta_bias == config.beta_bias
        assert cell.key_dim == config.key_dim
        assert cell.value_dim == config.value_dim
        assert cell.gate_seed_offset == config.gate_seed_offset
        assert cell.failure_threshold == config.failure_threshold
        assert cell.signals == config.signals
        assert cell.threshold_quantiles == config.threshold_quantiles
        assert cell.fixed_ratio_seed == config.fixed_ratio_seed
        assert cell.entropy_bins == config.entropy_bins
        assert cell.fallback_temperature == config.fallback_temperature
        assert cell.variants == config.variants


def test_default_grid_contains_issue38_default_cell():
    """The (8, 0.9) cell must coincide with the #38 default config."""

    cells = routing_regime_cells(RoutingRobustnessConfig())
    default_cells = [
        c for c in cells if c.num_pairs == 8 and c.collision_cosine == 0.9
    ]
    assert len(default_cells) == 1
    cell = default_cells[0]
    reference = RoutingExperimentConfig()
    for field in (
        "variants",
        "seeds",
        "batch_size",
        "num_pairs",
        "key_dim",
        "value_dim",
        "collision_cosine",
        "gate_seed_offset",
        "alpha_bias",
        "beta_bias",
        "failure_threshold",
        "signals",
        "threshold_quantiles",
        "fixed_ratio_seed",
        "entropy_bins",
        "fallback_temperature",
    ):
        assert getattr(cell, field) == getattr(reference, field), field


def test_default_cell_reproduces_issue38_pipeline():
    """Regression: the default cell equals the #38 pipeline bit-for-bit.

    Sample collection and operating-point evaluation are deterministic, so
    the sweep's per-variant section at the #38 default operating point must
    equal the #38 experiment's section exactly.
    """

    config = RoutingRobustnessConfig(
        num_pairs_grid=(8,), collision_cosine_grid=(0.9,)
    )
    report = run_routing_robustness(config)
    reference = run_routing_experiment(RoutingExperimentConfig())
    for variant in config.variants:
        cell_section = report["cells"][0]["variants"][variant]
        assert json.dumps(cell_section, sort_keys=True) == json.dumps(
            reference["variants"][variant], sort_keys=True
        )


def test_default_cell_data_matches_direct_collection():
    """The sweep's collection call is the #38 collection call on the cell."""

    config = _tiny_config()
    cell = routing_regime_cells(config)[0]
    swept = collect_routing_data("kda-rls-lite", cell)
    direct = collect_routing_data("kda-rls-lite", cell)
    assert swept.variant == direct.variant == "kda-rls-lite"
    np.testing.assert_array_equal(swept.compressed_ok, direct.compressed_ok)
    np.testing.assert_array_equal(swept.exact_ok, direct.exact_ok)
    for name in config.signals:
        np.testing.assert_array_equal(swept.scores[name], direct.scores[name])


def test_report_schema_tiny_grid():
    config = _tiny_config()
    report = run_routing_robustness(config)
    assert report["experiment"] == "phase4-routing-robustness"
    assert report["issue"] == 38
    assert report["config"] == config.to_dict()
    assert len(report["cells"]) == 4  # 2 loads x 2 cosines

    for cell in report["cells"]:
        assert set(cell) == {"num_pairs", "collision_cosine", "variants"}
        assert set(cell["variants"]) == set(config.variants)
        for name in config.variants:
            section = cell["variants"][name]
            assert set(section) == {
                "num_items",
                "compressed_accuracy",
                "exact_accuracy",
                "exact_headroom",
                "signals",
            }
            assert set(section["signals"]) == set(config.signals)
            for signal, entry in section["signals"].items():
                assert entry["description"] is not None
                if entry["available"]:
                    assert set(entry) == {"available", "description", "operating_points"}
                    assert len(entry["operating_points"]) == len(
                        config.threshold_quantiles
                    )
                    for point in entry["operating_points"]:
                        assert set(point) == {
                            "threshold",
                            "fallback_rate",
                            "recall_accuracy",
                            "improvement_over_compressed",
                            "fixed_ratio_accuracy",
                            "gain_vs_fixed_ratio",
                            "overuse_rate",
                            "underuse_rate",
                            "num_routed",
                            "num_items",
                        }
                else:
                    assert set(entry) == {"available", "description", "reason"}


def test_signal_availability_by_variant():
    report = run_routing_robustness(_tiny_config())
    generic = ("prediction_error_norm", "key_collision_score", "gate_entropy")
    for cell in report["cells"]:
        signals = cell["variants"]["kda"]["signals"]
        assert not signals["precision_along_key"]["available"]
        assert not signals["evidence_along_key"]["available"]

        signals = cell["variants"]["kda-rls-lite"]["signals"]
        assert signals["evidence_along_key"]["available"]
        assert not signals["precision_along_key"]["available"]

        signals = cell["variants"]["precision-gated-kda"]["signals"]
        assert signals["precision_along_key"]["available"]
        assert not signals["evidence_along_key"]["available"]

        for variant in config_variants():
            for name in generic:
                assert cell["variants"][variant]["signals"][name]["available"]


def config_variants():
    return ("kda", "kda-rls-lite", "precision-gated-kda")


def test_item_counts_and_metric_bounds():
    config = _tiny_config()
    report = run_routing_robustness(config)
    for cell in report["cells"]:
        expected_items = len(config.seeds) * config.batch_size * 2 * cell["num_pairs"]
        for name in config.variants:
            section = cell["variants"][name]
            assert section["num_items"] == expected_items
            for key in ("compressed_accuracy", "exact_accuracy"):
                assert 0.0 <= section[key] <= 1.0
            for entry in section["signals"].values():
                if not entry["available"]:
                    continue
                for point in entry["operating_points"]:
                    assert point["num_items"] == expected_items
                    assert 0.0 <= point["fallback_rate"] <= 1.0
                    assert 0.0 <= point["recall_accuracy"] <= 1.0
                    assert 0.0 <= point["overuse_rate"] <= 1.0
                    assert 0.0 <= point["underuse_rate"] <= 1.0
                    # Accuracy is a mixture of the two endpoint arms.
                    assert point["recall_accuracy"] >= min(
                        section["compressed_accuracy"], section["exact_accuracy"]
                    ) - 1e-9


def test_deterministic_across_runs():
    config = _tiny_config()
    first = json.dumps(run_routing_robustness(config), sort_keys=True)
    second = json.dumps(run_routing_robustness(config), sort_keys=True)
    assert first == second


def test_positive_control_native_signal_routing_beats_fixed_ratio():
    """Seeded positive control with wide margins (cf. the #38 defaults).

    At the #38 default operating point the native precision signal ranks
    failing queries above succeeding ones (the #28 calibration found AUROC
    0.872 there), so adaptive routing on it must beat the matched-budget
    fixed-ratio baseline at the 0.9-quantile operating point by a clear
    margin. The bound stays loose to avoid over-fitting the point estimate.
    """

    config = RoutingRobustnessConfig(
        num_pairs_grid=(8,), collision_cosine_grid=(0.9,)
    )
    report = run_routing_robustness(config)
    signals = report["cells"][0]["variants"]["precision-gated-kda"]["signals"]
    entry = signals["precision_along_key"]
    assert entry["available"]
    # threshold_quantiles index 3 corresponds to the 0.9 quantile.
    point = entry["operating_points"][3]
    assert point["gain_vs_fixed_ratio"] > 0.02
    assert point["recall_accuracy"] > point["fixed_ratio_accuracy"]
    assert point["fallback_rate"] == pytest.approx(0.1, abs=0.05)
