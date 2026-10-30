"""Tests for the #36 KDA-RLS-lite hyperparameter ablation."""

import dataclasses
import json

import pytest
import torch

from kda.experiments import (
    CapacityConfig,
    RLSLiteAblationConfig,
    is_rls_lite_default_cell,
    rls_lite_cells,
    run_capacity_experiment,
    run_rls_lite_ablation,
)
from kda.variants import create_variant


def _tiny_config(**overrides):
    """Fast deterministic ablation config for end-to-end tests."""
    base = dict(
        evidence_decay_grid=(0.9, 1.0),
        beta_min_grid=(0.0,),
        beta_max_grid=(0.5, 1.0),
        operating_points=((2, 0.5), (3, 0.9)),
        seeds=(0, 1),
        batch_size=2,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=8,
    )
    base.update(overrides)
    return RLSLiteAblationConfig(**base)


def test_config_snapshot_is_json_serializable():
    config = RLSLiteAblationConfig()
    snapshot = json.loads(json.dumps(config.to_dict(), sort_keys=True))
    assert snapshot["evidence_decay_grid"] == [0.8, 0.9, 1.0]
    assert snapshot["beta_min_grid"] == [0.0, 0.1]
    assert snapshot["beta_max_grid"] == [0.5, 1.0]
    assert snapshot["lambda_"] == 1.0
    assert snapshot["operating_points"] == [[8, 0.9], [16, 0.9]]
    assert snapshot["reference_variants"] == ["kda"]
    assert snapshot["seeds"] == [0, 1, 2]
    assert snapshot["failure_threshold"] == 0.5
    assert snapshot["bootstrap_resamples"] == 200
    assert snapshot["ci_level"] == 0.95


def test_config_is_frozen():
    config = _tiny_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.bootstrap_resamples = 4


def test_config_validation():
    with pytest.raises(ValueError, match="unknown reference variants"):
        _tiny_config(reference_variants=("kda", "not-a-variant"))
    with pytest.raises(ValueError, match="evidence_decay_grid"):
        _tiny_config(evidence_decay_grid=())
    with pytest.raises(ValueError, match="evidence_decay_grid"):
        _tiny_config(evidence_decay_grid=(1.0, 0.0))
    with pytest.raises(ValueError, match="evidence_decay_grid"):
        _tiny_config(evidence_decay_grid=(1.0, 1.5))
    with pytest.raises(ValueError, match="beta_min_grid"):
        _tiny_config(beta_min_grid=())
    with pytest.raises(ValueError, match="beta_min_grid"):
        _tiny_config(beta_min_grid=(0.0, -0.1))
    with pytest.raises(ValueError, match="beta_max_grid"):
        _tiny_config(beta_max_grid=())
    with pytest.raises(ValueError, match="beta_max_grid"):
        _tiny_config(beta_max_grid=(0.5, 0.0))
    with pytest.raises(ValueError, match="beta_max_grid"):
        _tiny_config(beta_max_grid=(0.5, 1.5))
    with pytest.raises(ValueError, match="max\\(beta_min_grid\\)"):
        _tiny_config(beta_min_grid=(0.6,), beta_max_grid=(0.5,))
    with pytest.raises(ValueError, match="lambda_"):
        _tiny_config(lambda_=-0.1)
    with pytest.raises(ValueError, match="operating_points"):
        _tiny_config(operating_points=())
    with pytest.raises(ValueError, match="num_pairs"):
        _tiny_config(operating_points=((0, 0.9),))
    with pytest.raises(ValueError, match="collision_cosine"):
        _tiny_config(operating_points=((8, 1.5),))
    with pytest.raises(ValueError, match="seeds"):
        _tiny_config(seeds=())
    with pytest.raises(ValueError, match="batch_size"):
        _tiny_config(batch_size=0)
    with pytest.raises(ValueError, match="key_dim"):
        _tiny_config(key_dim=1)
    with pytest.raises(ValueError, match="failure_threshold"):
        _tiny_config(failure_threshold=2.0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        _tiny_config(bootstrap_resamples=0)
    with pytest.raises(ValueError, match="ci_level"):
        _tiny_config(ci_level=1.0)


def test_rls_lite_cells_cartesian_order():
    config = _tiny_config(
        evidence_decay_grid=(0.9, 1.0),
        beta_min_grid=(0.0, 0.1),
        beta_max_grid=(0.5, 1.0),
    )
    cells = rls_lite_cells(config)
    assert [
        (c["evidence_decay"], c["beta_min"], c["beta_max"]) for c in cells
    ] == [
        (0.9, 0.0, 0.5),
        (0.9, 0.0, 1.0),
        (0.9, 0.1, 0.5),
        (0.9, 0.1, 1.0),
        (1.0, 0.0, 0.5),
        (1.0, 0.0, 1.0),
        (1.0, 0.1, 0.5),
        (1.0, 0.1, 1.0),
    ]


def test_is_rls_lite_default_cell():
    assert is_rls_lite_default_cell(
        {"evidence_decay": 1.0, "beta_min": 0.0, "beta_max": 1.0}
    )
    assert not is_rls_lite_default_cell(
        {"evidence_decay": 0.9, "beta_min": 0.0, "beta_max": 1.0}
    )
    assert not is_rls_lite_default_cell(
        {"evidence_decay": 1.0, "beta_min": 0.1, "beta_max": 1.0}
    )
    assert not is_rls_lite_default_cell(
        {"evidence_decay": 1.0, "beta_min": 0.0, "beta_max": 0.5}
    )


def test_default_cell_reproduces_report_pipeline_bit_for_bit():
    """The default ablation cell at (8, 0.9) must coincide with the #34
    RLS-lite capacity row: identical tasks, gates, variant, and bootstrap,
    so point estimates and CIs match exactly."""
    shared = dict(
        seeds=(0, 1),
        batch_size=2,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=8,
    )
    ablation_report = run_rls_lite_ablation(
        _tiny_config(
            **shared,
            evidence_decay_grid=(1.0,),
            beta_min_grid=(0.0,),
            beta_max_grid=(1.0,),
            operating_points=((8, 0.9),),
        )
    )
    cell = ablation_report["operating_points"][0]["cells"][0]
    assert cell["is_default"]

    capacity_report = run_capacity_experiment(
        CapacityConfig(
            **shared,
            variants=("kda-rls-lite",),
            capacity_loads=(8,),
            capacity_collision_cosine=0.9,
            interference_cosines=(0.9,),
            interference_num_pairs=8,
        )
    )
    point = capacity_report["capacity_sweep"]["kda-rls-lite"]["points"][0]
    assert cell["num_items"] == point["num_items"]
    assert cell["recall_accuracy"] == point["recall_accuracy"]
    assert cell["mean_recall_cosine"] == point["mean_recall_cosine"]


def test_reference_row_reproduces_report_pipeline_bit_for_bit():
    """The fixed-beta KDA reference row runs on identical draws to the #34
    capacity pipeline, so it must match its KDA row exactly."""
    shared = dict(
        seeds=(0, 1),
        batch_size=2,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=8,
    )
    ablation_report = run_rls_lite_ablation(
        _tiny_config(**shared, operating_points=((3, 0.9),))
    )
    reference = ablation_report["operating_points"][0]["reference"]["kda"]

    capacity_report = run_capacity_experiment(
        CapacityConfig(
            **shared,
            variants=("kda",),
            capacity_loads=(3,),
            capacity_collision_cosine=0.9,
            interference_cosines=(0.9,),
            interference_num_pairs=3,
        )
    )
    point = capacity_report["capacity_sweep"]["kda"]["points"][0]
    assert reference["num_items"] == point["num_items"]
    assert reference["recall_accuracy"] == point["recall_accuracy"]
    assert reference["mean_recall_cosine"] == point["mean_recall_cosine"]


def test_report_schema_and_ci_bounds():
    config = _tiny_config()
    report = run_rls_lite_ablation(config)
    assert report["experiment"] == "phase4-rls-lite-ablation"
    assert report["issue"] == 36
    assert report["config"] == config.to_dict()
    assert report["default_cell"] == {
        "evidence_decay": 1.0,
        "beta_min": 0.0,
        "beta_max": 1.0,
    }
    # The report round-trips through JSON without loss.
    assert json.loads(json.dumps(report)) == report

    assert len(report["operating_points"]) == 2
    expected_cells = rls_lite_cells(config)
    for point, (num_pairs, cosine) in zip(
        report["operating_points"], config.operating_points
    ):
        assert point["num_pairs"] == num_pairs
        assert point["collision_cosine"] == cosine
        assert list(point["reference"]) == ["kda"]
        assert len(point["cells"]) == len(expected_cells)
        expected_items = len(config.seeds) * config.batch_size * 2 * num_pairs
        for cell, expected in zip(point["cells"], expected_cells):
            assert cell["evidence_decay"] == expected["evidence_decay"]
            assert cell["beta_min"] == expected["beta_min"]
            assert cell["beta_max"] == expected["beta_max"]
            assert cell["lambda_"] == config.lambda_
            assert cell["is_default"] == is_rls_lite_default_cell(expected)
            assert cell["num_items"] == expected_items
            rate = cell["recall_accuracy"]
            low, high = rate["ci"]
            assert 0.0 <= low <= rate["value"] <= high <= 1.0


def test_run_to_run_determinism():
    config = _tiny_config()
    first = run_rls_lite_ablation(config)
    second = run_rls_lite_ablation(config)
    assert first == second


def test_beta_max_clamp_bounds_effective_gate():
    """Mechanism check: with beta_max < 1 the effective write gate actually
    applied by the variant never exceeds the clamp (trace-level, so the
    sweep's upper-bound cells cannot silently behave like the default)."""
    torch.manual_seed(0)
    beta_max = 0.2
    variant = create_variant(
        "kda-rls-lite", head_dim=8, value_dim=4, beta_max=beta_max
    )
    queries = torch.randn(2, 6, 8)
    keys = torch.nn.functional.normalize(torch.randn(2, 6, 8), dim=-1)
    values = torch.randn(2, 6, 4)
    gates = {
        "alpha": torch.full((2, 6, 8), 0.9),
        "beta": torch.full((2, 6, 1), 0.9),
    }
    result = variant(queries, keys, values, gates=gates, return_trace=True)
    assert float(result.trace["beta_eff"].max()) <= beta_max + 1e-6


def test_easy_operating_point_positive_control():
    """Seeded positive control with wide margins: at low load and moderate
    interference (2 pairs, cosine 0.5), the default cell and the KDA
    reference both recall well above chance."""
    report = run_rls_lite_ablation(
        _tiny_config(
            evidence_decay_grid=(1.0,),
            beta_min_grid=(0.0,),
            beta_max_grid=(1.0,),
            operating_points=((2, 0.5),),
            batch_size=4,
        )
    )
    point = report["operating_points"][0]
    default_cell = point["cells"][0]
    assert default_cell["is_default"]
    assert default_cell["recall_accuracy"]["value"] > 0.5
    assert point["reference"]["kda"]["recall_accuracy"]["value"] > 0.5
