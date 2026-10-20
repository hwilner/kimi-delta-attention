"""Tests for the issue #34 report experiments (capacity, overwrite, locality).

Covers the pre-registered configs (validation, frozen, JSON snapshot), the
edit-locality metric on hand-computed cases, the EDA trace replay against the
variant's own final state, deterministic known cases (tied baselines == KDA,
EDA gamma=0 == KDA, single-key saturated overwrites), small-config end-to-end
schema/determinism, and the report renderer (required sections, determinism,
byte-for-byte reproduction of the checked-in report).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kda.experiments import (
    CalibrationConfig,
    CapacityConfig,
    OverwriteConfig,
    RoutingExperimentConfig,
    edit_locality_metrics,
    run_capacity_experiment,
    run_overwrite_experiment,
)
from kda.experiments.report_experiments import (
    _collect_capacity_point,
    _collect_overwrite_point,
    _eda_state_trajectory,
    _sample_gates,
)
from kda.instrumentation import replay_state_trajectory
from kda.tasks import OverwriteTaskConfig, generate_overwrite_batch
from kda.variants import create_variant

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKED_IN_REPORT = REPO_ROOT / "docs" / "PHASE4_RESULTS.md"


def _load_renderer():
    """Import examples/phase4_results_report.py as a module."""

    spec = importlib.util.spec_from_file_location(
        "phase4_results_report", REPO_ROOT / "examples" / "phase4_results_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_configs():
    """Small, fast configs for renderer/schema tests."""

    return {
        "capacity_config": CapacityConfig(
            seeds=(0,), batch_size=2, capacity_loads=(2,),
            interference_cosines=(0.0,), bootstrap_resamples=10,
        ),
        "overwrite_config": OverwriteConfig(
            seeds=(0,), batch_size=2, update_counts=(2,), bootstrap_resamples=10,
        ),
        "calibration_config": CalibrationConfig(
            seeds=(0,), batch_size=2, num_pairs=4, bootstrap_resamples=10,
        ),
        "routing_config": RoutingExperimentConfig(
            seeds=(0,), batch_size=2, num_pairs=4, threshold_quantiles=(0.5, 0.9),
        ),
    }


# ---------------------------------------------------------------------------
# Config validation / frozen / snapshots
# ---------------------------------------------------------------------------


def test_capacity_config_json_snapshot():
    snapshot = CapacityConfig().to_dict()
    assert json.dumps(snapshot)  # JSON-serializable
    assert snapshot["variants"] == [
        "kda", "kda-rls-lite", "precision-gated-kda", "gdn-2", "eda",
    ]
    assert snapshot["capacity_loads"] == [2, 4, 8, 16]
    assert snapshot["interference_cosines"] == [0.0, 0.5, 0.8, 0.9, 0.99]
    assert snapshot["alpha_bias"] == 2.0 and snapshot["beta_bias"] == 1.0


def test_overwrite_config_json_snapshot():
    snapshot = OverwriteConfig().to_dict()
    assert json.dumps(snapshot)
    assert snapshot["variants"] == ["kda", "kda-rls-lite", "precision-gated-kda", "eda"]
    assert snapshot["update_counts"] == [2, 4, 8]
    assert snapshot["eda_erase_gate"] == 0.5


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CapacityConfig().seeds = (1,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        OverwriteConfig().eda_erase_gate = 0.1


def test_capacity_config_validation():
    with pytest.raises(ValueError, match="unknown variants"):
        CapacityConfig(variants=("nope",))
    with pytest.raises(ValueError, match="non-empty"):
        CapacityConfig(seeds=())
    with pytest.raises(ValueError, match="capacity_loads"):
        CapacityConfig(capacity_loads=(0,))
    with pytest.raises(ValueError, match="capacity_collision_cosine"):
        CapacityConfig(capacity_collision_cosine=1.5)
    with pytest.raises(ValueError, match="interference_cosines"):
        CapacityConfig(interference_cosines=(2.0,))
    with pytest.raises(ValueError, match="failure_threshold"):
        CapacityConfig(failure_threshold=2.0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        CapacityConfig(bootstrap_resamples=0)


def test_overwrite_config_validation():
    with pytest.raises(ValueError, match="unknown variants"):
        OverwriteConfig(variants=("nope",))
    with pytest.raises(ValueError, match="update_counts"):
        OverwriteConfig(update_counts=(1,))
    with pytest.raises(ValueError, match="eda_erase_gate"):
        OverwriteConfig(eda_erase_gate=1.5)
    with pytest.raises(ValueError, match="num_keys"):
        OverwriteConfig(num_keys=0)


# ---------------------------------------------------------------------------
# Edit-locality metric (hand-computed cases)
# ---------------------------------------------------------------------------


def test_edit_locality_hand_computed():
    # Single write into an empty state: relative norm is always 1.
    key = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2), unit norm
    alpha = torch.ones(1, 1, 2)
    value_dir = torch.tensor([[[3.0, 4.0]]])
    on_key = torch.einsum("blh,blv->blhv", key, value_dir)
    metrics = edit_locality_metrics(on_key, key, alpha)
    assert metrics["relative_update_norm"] == pytest.approx(1.0)
    assert metrics["key_axis_alignment"] == pytest.approx(1.0)

    # Edit entirely off the key axis: alignment 0.
    off_key_dir = torch.tensor([[[0.0, 1.0]]])
    off_key = torch.einsum("blh,blv->blhv", off_key_dir, value_dir)
    metrics = edit_locality_metrics(off_key, key, alpha)
    assert metrics["key_axis_alignment"] == pytest.approx(0.0)


def test_edit_locality_validation():
    states = torch.zeros(1, 2, 3, 4)
    keys = torch.zeros(1, 2, 3)
    alpha = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="states"):
        edit_locality_metrics(states[:, :, 0], keys, alpha)
    with pytest.raises(ValueError, match="keys"):
        edit_locality_metrics(states, keys[:, :1], alpha)
    with pytest.raises(ValueError, match="alpha"):
        edit_locality_metrics(states, keys, alpha[:, :1])
    with pytest.raises(ValueError, match="eps"):
        edit_locality_metrics(states, keys, alpha, eps=0.0)


def test_edit_locality_kda_writes_are_key_aligned():
    # KDA's write vector is beta * k, so the replayed edit is exactly
    # parallel to the key axis (alignment 1 up to float error).
    config = OverwriteConfig(seeds=(0,), batch_size=2, update_counts=(2,))
    task = generate_overwrite_batch(
        OverwriteTaskConfig(
            batch_size=2, num_keys=config.num_keys, num_updates=2,
            key_dim=config.key_dim, value_dim=config.value_dim, seed=0,
        )
    )
    gates = _sample_gates(2, task.sequence_keys.shape[1], config.key_dim, 10000, 2.0, 1.0)
    variant = create_variant("kda", head_dim=config.key_dim, value_dim=config.value_dim)
    with torch.no_grad():
        result = variant(
            task.sequence_keys, task.sequence_keys, task.sequence_values,
            gates=gates, return_trace=True,
        )
        states = replay_state_trajectory(
            task.sequence_keys, task.sequence_values, gates["alpha"],
            result.trace["beta"] * task.sequence_keys,
        )
    metrics = edit_locality_metrics(states, task.sequence_keys, gates["alpha"])
    assert metrics["key_axis_alignment"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["relative_update_norm"] > 0.0


# ---------------------------------------------------------------------------
# EDA trace replay
# ---------------------------------------------------------------------------


def test_eda_replay_matches_variant_final_state():
    torch.manual_seed(7)
    batch, seq_len, head_dim, value_dim = 3, 9, 8, 6
    keys = torch.nn.functional.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    values = torch.randn(batch, seq_len, value_dim)
    gates = _sample_gates(batch, seq_len, head_dim, 42, 2.0, 1.0)
    gates["erase_address"] = keys
    gates["erase_gate"] = torch.full((batch, seq_len, 1), 0.5)
    variant = create_variant("eda", head_dim=head_dim, value_dim=value_dim)
    with torch.no_grad():
        result = variant(keys, keys, values, gates=gates, return_trace=True)
        states = _eda_state_trajectory(
            keys, values, gates["alpha"], gates["beta"],
            result.trace["erase_address"], result.trace["erase_gate"],
            result.trace["erased_response"],
        )
    assert torch.allclose(states[:, -1], result.final_state, atol=1e-5)


def test_eda_replay_validation():
    batch, seq_len, head_dim, value_dim = 2, 3, 4, 5
    keys = torch.zeros(batch, seq_len, head_dim)
    values = torch.zeros(batch, seq_len, value_dim)
    alpha = torch.ones(batch, seq_len, head_dim)
    beta = torch.ones(batch, seq_len, 1)
    address = torch.ones(batch, seq_len, head_dim)
    gate = torch.ones(batch, seq_len, 1)
    response = torch.ones(batch, seq_len, value_dim)
    with pytest.raises(ValueError, match="alpha"):
        _eda_state_trajectory(keys, values, alpha[:, :1], beta, address, gate, response)
    with pytest.raises(ValueError, match="beta"):
        _eda_state_trajectory(keys, values, alpha, beta[:, :, :0], address, gate, response)
    with pytest.raises(ValueError, match="erased_response"):
        _eda_state_trajectory(keys, values, alpha, beta, address, gate, response[:, :, :1])


# ---------------------------------------------------------------------------
# Deterministic known cases at the experiment level
# ---------------------------------------------------------------------------


def test_tied_baselines_match_kda_exactly():
    # GDN-2 (tied gates) and EDA (no erase) reduce to KDA on the shared gate
    # path (paper ties); the experiment rows must match exactly.
    config = CapacityConfig(seeds=(0,), batch_size=4, bootstrap_resamples=20)
    reference = _collect_capacity_point("kda", 4, 0.9, config)
    assert _collect_capacity_point("gdn-2", 4, 0.9, config) == reference
    assert _collect_capacity_point("eda", 4, 0.9, config) == reference


def test_eda_zero_erase_gate_matches_kda_on_overwrite():
    config = OverwriteConfig(
        seeds=(0,), batch_size=4, update_counts=(2,),
        eda_erase_gate=0.0, bootstrap_resamples=20,
    )
    reference = _collect_overwrite_point("kda", 2, config)
    point = _collect_overwrite_point("eda", 2, config)
    assert point["fresh_recall_accuracy"] == reference["fresh_recall_accuracy"]
    assert point["stale_dominance_rate"] == reference["stale_dominance_rate"]


def test_single_key_saturated_gates_recall_latest_exactly():
    # With alpha = beta = 1 (saturated biases) and a single key, the delta
    # rule's last write wins exactly: fresh recall is perfect and no stale
    # value dominates. Deterministic margins, not tuned to outcomes.
    config = OverwriteConfig(
        seeds=(0,), batch_size=4, num_keys=1, update_counts=(2, 4),
        alpha_bias=100.0, beta_bias=100.0, bootstrap_resamples=20,
    )
    point = _collect_overwrite_point("kda", 4, config)
    assert point["fresh_recall_accuracy"]["value"] == 1.0
    assert point["stale_dominance_rate"]["value"] == 0.0


def test_interference_ordering():
    # Orthogonal keys must not recall worse than near-identical colliding
    # keys (deterministic given the pre-registered seeds).
    config = CapacityConfig()
    for variant in ("kda", "kda-rls-lite", "precision-gated-kda"):
        easy = _collect_capacity_point(
            variant, config.interference_num_pairs, 0.0, config
        )
        hard = _collect_capacity_point(
            variant, config.interference_num_pairs, 0.99, config
        )
        assert easy["recall_accuracy"]["value"] >= hard["recall_accuracy"]["value"]


# ---------------------------------------------------------------------------
# End-to-end schema, determinism, JSON serializability (small configs)
# ---------------------------------------------------------------------------


def test_capacity_report_schema_and_determinism():
    config = CapacityConfig(
        seeds=(0,), batch_size=2, capacity_loads=(2,),
        interference_cosines=(0.0,), bootstrap_resamples=10,
    )
    report = run_capacity_experiment(config)
    assert report["issue"] == 34
    assert json.dumps(report) == json.dumps(run_capacity_experiment(config))
    for name in config.variants:
        point = report["capacity_sweep"][name]["points"][0]
        assert point["num_pairs"] == 2
        # 1 seed * batch 2 * (2 * num_pairs) items per sequence.
        assert point["num_items"] == 8
        assert 0.0 <= point["recall_accuracy"]["value"] <= 1.0
        low, high = point["recall_accuracy"]["ci"]
        assert low <= high


def test_overwrite_report_schema_and_determinism():
    config = OverwriteConfig(
        seeds=(0,), batch_size=2, update_counts=(2,), bootstrap_resamples=10,
    )
    report = run_overwrite_experiment(config)
    assert report["issue"] == 34
    assert report["config"]["eda_erase_gate"] == 0.5
    assert json.dumps(report) == json.dumps(run_overwrite_experiment(config))
    for name in config.variants:
        point = report["variants"][name]["points"][0]
        assert point["num_updates"] == 2
        assert point["num_items"] == 1 * 2 * config.num_keys
        assert 0.0 <= point["edit_key_axis_alignment"] <= 1.0
        assert point["edit_relative_update_norm"] > 0.0


# ---------------------------------------------------------------------------
# Report renderer (issue #34 acceptance: generated from checked-in configs)
# ---------------------------------------------------------------------------


def test_renderer_has_all_required_sections():
    renderer = _load_renderer()
    markdown = renderer.build_report_markdown(**_tiny_configs())
    # Required sections of issue #34.
    assert "## Hypothesis and related-work boundary" in markdown
    assert "## Exact recurrence definitions" in markdown
    assert "## Task generators and seeds" in markdown
    assert "## Capacity / interference curves" in markdown
    assert "## Overwrite / recency behavior" in markdown
    assert "## Edit locality" in markdown
    assert "Uncertainty calibration" in markdown
    assert "## Computational overhead" in markdown
    assert "## Negative results and limitations" in markdown
    # Prior baselines cited where relevant (acceptance criterion).
    assert "arXiv:2605.22791" in markdown  # GDN-2
    assert "arXiv:2606.26560" in markdown  # EDA
    assert "prior baseline" in markdown
    # Tables present.
    assert "| variant |" in markdown


def test_renderer_deterministic_on_tiny_configs():
    renderer = _load_renderer()
    first = renderer.build_report_markdown(**_tiny_configs())
    second = renderer.build_report_markdown(**_tiny_configs())
    assert first == second


def test_checked_in_report_reproduces_byte_for_byte():
    # Acceptance criterion: every table in docs/PHASE4_RESULTS.md is
    # generated from the checked-in default configs by the checked-in
    # script. Re-running the generator must reproduce the file exactly.
    renderer = _load_renderer()
    regenerated = renderer.build_report_markdown()
    assert CHECKED_IN_REPORT.exists()
    assert regenerated == CHECKED_IN_REPORT.read_text()
