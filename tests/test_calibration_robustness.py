"""Tests for the calibration robustness sweep (extends Phase 4 issue #28)."""

import dataclasses
import json
import math

import numpy as np
import pytest

from kda.experiments import (
    CalibrationConfig,
    CalibrationRobustnessConfig,
    binary_auroc,
    collect_variant_samples,
    regime_cells,
    run_calibration_robustness,
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
        bootstrap_resamples=8,
    )
    base.update(overrides)
    return CalibrationRobustnessConfig(**base)


def test_config_snapshot_is_json_serializable():
    config = CalibrationRobustnessConfig()
    snapshot = json.loads(json.dumps(config.to_dict(), sort_keys=True))
    assert snapshot["variants"] == ["kda", "kda-rls-lite", "precision-gated-kda"]
    assert snapshot["seeds"] == [0, 1, 2]
    assert snapshot["num_pairs_grid"] == [4, 8, 16]
    assert snapshot["collision_cosine_grid"] == [0.5, 0.9, 0.99]
    assert snapshot["failure_threshold"] == 0.5
    assert snapshot["bootstrap_resamples"] == 200
    assert snapshot["ci_level"] == 0.95


def test_config_is_frozen():
    config = _tiny_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.bootstrap_resamples = 4


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
    with pytest.raises(ValueError, match="entropy_bins"):
        _tiny_config(entropy_bins=0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        _tiny_config(bootstrap_resamples=0)
    with pytest.raises(ValueError, match="ci_level"):
        _tiny_config(ci_level=1.0)


def test_regime_cells_order_and_inheritance():
    config = _tiny_config(seeds=(5, 6), batch_size=3, alpha_bias=1.5)
    cells = regime_cells(config)
    # Cartesian product, outer loop over num_pairs, inner loop over cosines.
    assert [(c.num_pairs, c.collision_cosine) for c in cells] == [
        (2, 0.5),
        (2, 0.9),
        (3, 0.5),
        (3, 0.9),
    ]
    for cell in cells:
        assert isinstance(cell, CalibrationConfig)
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
        assert cell.entropy_bins == config.entropy_bins
        assert cell.variants == config.variants


def test_default_grid_contains_issue28_default_cell():
    """The (8, 0.9) cell must coincide with the #28 default config."""

    cells = regime_cells(CalibrationRobustnessConfig())
    default_cells = [
        c for c in cells if c.num_pairs == 8 and c.collision_cosine == 0.9
    ]
    assert len(default_cells) == 1
    cell = default_cells[0]
    reference = CalibrationConfig()
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
        "entropy_bins",
    ):
        assert getattr(cell, field) == getattr(reference, field), field


def test_default_cell_reproduces_issue28_pipeline():
    """Regression: the default cell equals the #28 pipeline bit-for-bit.

    Sample collection is deterministic and bootstrap parameters only enter
    the CI, so the sweep's point estimates at the #28 default operating
    point must equal the #28 pipeline's exactly.
    """

    config = CalibrationRobustnessConfig(
        num_pairs_grid=(8,),
        collision_cosine_grid=(0.9,),
        bootstrap_resamples=8,
    )
    report = run_calibration_robustness(config)
    cell_report = report["cells"][0]["variants"]["precision-gated-kda"]

    reference_samples = collect_variant_samples(
        "precision-gated-kda", CalibrationConfig()
    )
    mask = np.isfinite(reference_samples.signals["precision_along_key"])
    reference_auroc = binary_auroc(
        reference_samples.signals["precision_along_key"][mask],
        reference_samples.labels[mask],
    )
    assert cell_report["signals"]["precision_along_key"]["auroc"]["value"] == reference_auroc
    assert cell_report["failure_rate"]["value"] == float(
        reference_samples.labels.mean()
    )
    assert cell_report["num_items"] == int(reference_samples.labels.size)


def test_report_schema_tiny_grid():
    config = _tiny_config()
    report = run_calibration_robustness(config)
    assert report["experiment"] == "phase4-calibration-robustness"
    assert report["issue"] == 28
    assert report["config"] == config.to_dict()
    assert len(report["cells"]) == 4  # 2 loads x 2 cosines

    for cell in report["cells"]:
        assert set(cell) == {"num_pairs", "collision_cosine", "variants"}
        assert set(cell["variants"]) == set(config.variants)
        for name in config.variants:
            section = cell["variants"][name]
            assert set(section) == {
                "num_sequences",
                "num_items",
                "num_failures",
                "failure_rate",
                "mean_recall_cosine",
                "signals",
            }
            assert set(section["failure_rate"]) == {"value", "ci"}
            assert set(section["signals"]) == set(config.signals)
            for signal, entry in section["signals"].items():
                assert entry["description"] is not None
                assert isinstance(entry["sign"], float)
                if entry["available"]:
                    # AUROC only: ECE/AP/bins are out of scope for the sweep.
                    assert set(entry) == {
                        "available",
                        "description",
                        "sign",
                        "num_items",
                        "auroc",
                    }
                    assert set(entry["auroc"]) == {"value", "ci"}
                else:
                    assert set(entry) == {
                        "available",
                        "description",
                        "sign",
                        "reason",
                    }


def test_signal_availability_by_variant():
    report = run_calibration_robustness(_tiny_config())
    for cell in report["cells"]:
        signals = cell["variants"]["kda"]["signals"]
        assert not signals["evidence_along_key"]["available"]
        assert not signals["precision_along_key"]["available"]
        assert signals["prediction_error_norm"]["available"]

        signals = cell["variants"]["kda-rls-lite"]["signals"]
        assert signals["evidence_along_key"]["available"]
        assert not signals["precision_along_key"]["available"]

        signals = cell["variants"]["precision-gated-kda"]["signals"]
        assert not signals["evidence_along_key"]["available"]
        assert signals["precision_along_key"]["available"]

        for name in config_signals_generic():
            for variant in ("kda", "kda-rls-lite", "precision-gated-kda"):
                assert cell["variants"][variant]["signals"][name]["available"]


def config_signals_generic():
    return (
        "prediction_error_norm",
        "key_collision_score",
        "gate_entropy",
        "state_effective_rank",
    )


def test_item_and_sequence_counts():
    config = _tiny_config()
    report = run_calibration_robustness(config)
    for cell in report["cells"]:
        expected_items = len(config.seeds) * config.batch_size * 2 * cell["num_pairs"]
        for name in config.variants:
            section = cell["variants"][name]
            assert section["num_sequences"] == len(config.seeds) * config.batch_size
            assert section["num_items"] == expected_items
            assert section["failure_rate"]["value"] == pytest.approx(
                section["num_failures"] / section["num_items"]
            )
            assert 0.0 <= section["failure_rate"]["value"] <= 1.0
            assert -1.0 <= section["mean_recall_cosine"] <= 1.0


def test_ci_bounds_within_unit_interval():
    report = run_calibration_robustness(_tiny_config())
    for cell in report["cells"]:
        for section in cell["variants"].values():
            for metric in [section["failure_rate"]] + [
                entry["auroc"]
                for entry in section["signals"].values()
                if entry["available"]
            ]:
                low, high = metric["ci"]
                if math.isnan(low) or math.isnan(high):
                    continue
                assert 0.0 <= low <= high <= 1.0


def test_deterministic_across_runs():
    config = _tiny_config()
    first = json.dumps(run_calibration_robustness(config), sort_keys=True)
    second = json.dumps(run_calibration_robustness(config), sort_keys=True)
    assert first == second


def test_positive_control_native_signals_predict_failure_across_regime():
    """Seeded positive control with wide margins (cf. the #28 defaults).

    At the #28 default operating point the native signals reached AUROC
    0.872 (precision) and 0.786 (evidence); the assertions stay well below
    those values to avoid over-fitting the test to the point estimate.
    """

    config = CalibrationRobustnessConfig(
        num_pairs_grid=(8,),
        collision_cosine_grid=(0.9,),
        bootstrap_resamples=16,
    )
    report = run_calibration_robustness(config)
    cell = report["cells"][0]
    precision = cell["variants"]["precision-gated-kda"]["signals"][
        "precision_along_key"
    ]
    evidence = cell["variants"]["kda-rls-lite"]["signals"]["evidence_along_key"]
    assert precision["available"] and evidence["available"]
    assert precision["auroc"]["value"] > 0.7
    assert evidence["auroc"]["value"] > 0.6
