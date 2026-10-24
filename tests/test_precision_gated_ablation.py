"""Tests for the #33 precision-gated KDA hyperparameter ablation."""

import dataclasses
import json

import pytest

from kda.experiments import (
    CapacityConfig,
    PrecisionGatedAblationConfig,
    ablation_cells,
    is_default_cell,
    run_capacity_experiment,
    run_precision_gated_ablation,
)


def _tiny_config(**overrides):
    """Fast deterministic ablation config for end-to-end tests."""
    base = dict(
        lambda_grid=(0.5, 1.0),
        forgetting_grid=(0.9, 1.0),
        process_noise_grid=(0.0,),
        operating_points=((2, 0.5), (3, 0.9)),
        seeds=(0, 1),
        batch_size=2,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=8,
    )
    base.update(overrides)
    return PrecisionGatedAblationConfig(**base)


def test_config_snapshot_is_json_serializable():
    config = PrecisionGatedAblationConfig()
    snapshot = json.loads(json.dumps(config.to_dict(), sort_keys=True))
    assert snapshot["lambda_grid"] == [0.3, 1.0, 3.0]
    assert snapshot["forgetting_grid"] == [0.9, 1.0]
    assert snapshot["process_noise_grid"] == [0.0, 0.05]
    assert snapshot["operating_points"] == [[8, 0.9], [16, 0.9]]
    assert snapshot["reference_variants"] == ["kda"]
    assert snapshot["seeds"] == [0, 1, 2]
    assert snapshot["failure_threshold"] == 0.5
    assert snapshot["precision_init"] == 1.0
    assert snapshot["bootstrap_resamples"] == 200
    assert snapshot["ci_level"] == 0.95


def test_config_is_frozen():
    config = _tiny_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.bootstrap_resamples = 4


def test_config_validation():
    with pytest.raises(ValueError, match="unknown reference variants"):
        _tiny_config(reference_variants=("kda", "not-a-variant"))
    with pytest.raises(ValueError, match="lambda_grid"):
        _tiny_config(lambda_grid=())
    with pytest.raises(ValueError, match="lambda_grid"):
        _tiny_config(lambda_grid=(1.0, 0.0))
    with pytest.raises(ValueError, match="forgetting_grid"):
        _tiny_config(forgetting_grid=())
    with pytest.raises(ValueError, match="forgetting_grid"):
        _tiny_config(forgetting_grid=(1.0, 1.5))
    with pytest.raises(ValueError, match="process_noise_grid"):
        _tiny_config(process_noise_grid=(0.0, -0.1))
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
    with pytest.raises(ValueError, match="p_min"):
        _tiny_config(p_min=2.0, p_max=1.0)
    with pytest.raises(ValueError, match="precision_init"):
        _tiny_config(precision_init=1e6)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        _tiny_config(bootstrap_resamples=0)
    with pytest.raises(ValueError, match="ci_level"):
        _tiny_config(ci_level=1.0)


def test_ablation_cells_cartesian_order():
    config = _tiny_config(
        lambda_grid=(0.5, 1.0),
        forgetting_grid=(0.9, 1.0),
        process_noise_grid=(0.0, 0.05),
    )
    cells = ablation_cells(config)
    assert [
        (c["lambda_"], c["forgetting"], c["process_noise"]) for c in cells
    ] == [
        (0.5, 0.9, 0.0),
        (0.5, 0.9, 0.05),
        (0.5, 1.0, 0.0),
        (0.5, 1.0, 0.05),
        (1.0, 0.9, 0.0),
        (1.0, 0.9, 0.05),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 0.05),
    ]


def test_is_default_cell():
    assert is_default_cell(
        {"lambda_": 1.0, "forgetting": 1.0, "process_noise": 0.0}
    )
    assert not is_default_cell(
        {"lambda_": 0.3, "forgetting": 1.0, "process_noise": 0.0}
    )
    assert not is_default_cell(
        {"lambda_": 1.0, "forgetting": 0.9, "process_noise": 0.0}
    )
    assert not is_default_cell(
        {"lambda_": 1.0, "forgetting": 1.0, "process_noise": 0.05}
    )


def test_default_cell_reproduces_report_pipeline_bit_for_bit():
    """The default ablation cell at (8, 0.9) must coincide with the #34
    precision-gated capacity row: identical tasks, gates, variant, and
    bootstrap, so point estimates and CIs match exactly."""
    shared = dict(
        seeds=(0, 1),
        batch_size=2,
        key_dim=16,
        value_dim=8,
        bootstrap_resamples=8,
    )
    ablation_report = run_precision_gated_ablation(
        _tiny_config(
            **shared,
            lambda_grid=(1.0,),
            forgetting_grid=(1.0,),
            process_noise_grid=(0.0,),
            operating_points=((8, 0.9),),
        )
    )
    cell = ablation_report["operating_points"][0]["cells"][0]
    assert cell["is_default"]

    capacity_report = run_capacity_experiment(
        CapacityConfig(
            **shared,
            variants=("precision-gated-kda",),
            capacity_loads=(8,),
            capacity_collision_cosine=0.9,
            interference_cosines=(0.9,),
            interference_num_pairs=8,
        )
    )
    point = capacity_report["capacity_sweep"]["precision-gated-kda"]["points"][0]
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
    ablation_report = run_precision_gated_ablation(
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
    report = run_precision_gated_ablation(config)
    assert report["experiment"] == "phase4-precision-gated-ablation"
    assert report["issue"] == 33
    assert report["config"] == config.to_dict()
    assert report["default_cell"] == {
        "lambda_": 1.0,
        "forgetting": 1.0,
        "process_noise": 0.0,
    }
    # The report round-trips through JSON without loss.
    assert json.loads(json.dumps(report)) == report

    assert len(report["operating_points"]) == 2
    expected_cells = ablation_cells(config)
    for point, (num_pairs, cosine) in zip(
        report["operating_points"], config.operating_points
    ):
        assert point["num_pairs"] == num_pairs
        assert point["collision_cosine"] == cosine
        assert list(point["reference"]) == ["kda"]
        assert len(point["cells"]) == len(expected_cells)
        expected_items = (
            len(config.seeds) * config.batch_size * 2 * num_pairs
        )
        for cell, expected in zip(point["cells"], expected_cells):
            assert cell["lambda_"] == expected["lambda_"]
            assert cell["forgetting"] == expected["forgetting"]
            assert cell["process_noise"] == expected["process_noise"]
            assert cell["is_default"] == is_default_cell(expected)
            assert cell["num_items"] == expected_items
            rate = cell["recall_accuracy"]
            low, high = rate["ci"]
            assert 0.0 <= low <= rate["value"] <= high <= 1.0


def test_run_to_run_determinism():
    config = _tiny_config()
    first = run_precision_gated_ablation(config)
    second = run_precision_gated_ablation(config)
    assert first == second


def test_easy_operating_point_positive_control():
    """Seeded positive control with wide margins: at low load and moderate
    interference (2 pairs, cosine 0.5), the default cell and the KDA
    reference both recall well above chance."""
    report = run_precision_gated_ablation(
        _tiny_config(
            lambda_grid=(1.0,),
            forgetting_grid=(1.0,),
            process_noise_grid=(0.0,),
            operating_points=((2, 0.5),),
            batch_size=4,
        )
    )
    point = report["operating_points"][0]
    default_cell = point["cells"][0]
    assert default_cell["is_default"]
    assert default_cell["recall_accuracy"]["value"] > 0.5
    assert point["reference"]["kda"]["recall_accuracy"]["value"] > 0.5
