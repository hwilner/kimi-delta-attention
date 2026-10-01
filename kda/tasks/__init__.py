"""Synthetic task generators for recurrent-memory research."""

from .generators import (
    KeyCollisionBatch,
    KeyCollisionTaskConfig,
    NoveltyProbeBatch,
    NoveltyProbeConfig,
    OverwriteTaskBatch,
    OverwriteTaskConfig,
    generate_key_collision_batch,
    generate_novelty_probe_batch,
    generate_overwrite_batch,
)

__all__ = [
    "KeyCollisionBatch",
    "KeyCollisionTaskConfig",
    "NoveltyProbeBatch",
    "NoveltyProbeConfig",
    "OverwriteTaskBatch",
    "OverwriteTaskConfig",
    "generate_key_collision_batch",
    "generate_novelty_probe_batch",
    "generate_overwrite_batch",
]
