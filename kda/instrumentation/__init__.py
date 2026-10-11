"""Instrumentation utilities for recurrent-memory experiments (issue #27)."""

from .diagnostics import (
    GateStatistics,
    KeyCollisionStatistics,
    PredictionErrorMetrics,
    StateMetrics,
    gate_statistics,
    key_collision_statistics,
    prediction_error_metrics,
    replay_state_trajectory,
    state_metrics,
    summarize_trajectory,
)

__all__ = [
    "GateStatistics",
    "KeyCollisionStatistics",
    "PredictionErrorMetrics",
    "StateMetrics",
    "gate_statistics",
    "key_collision_statistics",
    "prediction_error_metrics",
    "replay_state_trajectory",
    "state_metrics",
    "summarize_trajectory",
]
