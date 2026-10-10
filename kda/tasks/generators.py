"""Synthetic associative-memory stress tasks.

These generators are intentionally small and deterministic. They are designed to
probe *how* recurrent memories fail, not to replace real language-modeling data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


DeviceLike = Optional[torch.device | str]


def _make_generator(seed: int, device: DeviceLike = None) -> torch.Generator:
    device = torch.device("cpu") if device is None else torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    device: DeviceLike = None,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, device=device)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _orthogonal_unit_vector(
    key: torch.Tensor,
    noise: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a unit vector orthogonal to ``key`` in the span of ``noise``."""
    orthogonal = noise - (noise * key).sum(dim=-1, keepdim=True) * key
    return _normalize(orthogonal + eps)


def _colliding_key(
    key: torch.Tensor,
    noise: torch.Tensor,
    cosine: float,
) -> torch.Tensor:
    """Construct a normalized key with an exact target cosine to ``key``."""
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"collision cosine must be in [-1, 1], got {cosine}")
    orthogonal = _orthogonal_unit_vector(key, noise)
    scale = (1.0 - cosine**2) ** 0.5
    return _normalize(cosine * key + scale * orthogonal)


@dataclass(frozen=True)
class KeyCollisionTaskConfig:
    """Configuration for controlled key-collision recall."""

    batch_size: int = 8
    num_pairs: int = 16
    key_dim: int = 32
    value_dim: int = 32
    collision_cosine: float = 0.9
    seed: int = 0
    shuffle_queries: bool = True
    device: DeviceLike = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.key_dim < 2:
            raise ValueError("key_dim must be at least 2")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if not -1.0 <= self.collision_cosine <= 1.0:
            raise ValueError("collision_cosine must be in [-1, 1]")


@dataclass
class KeyCollisionBatch:
    """Batch produced by :func:`generate_key_collision_batch`.

    ``keys`` interleaves each base key with its colliding key. The pair
    ``(2*i, 2*i + 1)`` therefore has the configured cosine similarity while
    mapping to independently sampled values.
    """

    keys: torch.Tensor
    values: torch.Tensor
    queries: torch.Tensor
    targets: torch.Tensor
    query_ids: torch.Tensor
    collision_pair_ids: torch.Tensor
    collision_cosines: torch.Tensor
    config: KeyCollisionTaskConfig


def generate_key_collision_batch(
    config: KeyCollisionTaskConfig,
) -> KeyCollisionBatch:
    """Generate key/value pairs with controlled pairwise key collisions."""
    generator = _make_generator(config.seed, config.device)
    base_keys = torch.empty(
        config.batch_size,
        config.num_pairs,
        config.key_dim,
        device=config.device,
    )
    colliding_keys = torch.empty_like(base_keys)

    for batch_idx in range(config.batch_size):
        for pair_idx in range(config.num_pairs):
            key = _normalize(
                _randn((config.key_dim,), generator, config.device)
            )
            noise = _randn((config.key_dim,), generator, config.device)
            base_keys[batch_idx, pair_idx] = key
            colliding_keys[batch_idx, pair_idx] = _colliding_key(
                key,
                noise,
                config.collision_cosine,
            )

    num_items = 2 * config.num_pairs
    keys = torch.empty(
        config.batch_size,
        num_items,
        config.key_dim,
        device=config.device,
    )
    keys[:, 0::2] = base_keys
    keys[:, 1::2] = colliding_keys

    values = _randn(
        (config.batch_size, num_items, config.value_dim),
        generator,
        config.device,
    )

    if config.shuffle_queries:
        query_ids = torch.stack(
            [
                torch.randperm(num_items, generator=generator, device=config.device)
                for _ in range(config.batch_size)
            ]
        )
        queries = torch.gather(
            keys,
            1,
            query_ids.unsqueeze(-1).expand(-1, -1, config.key_dim),
        )
        targets = torch.gather(
            values,
            1,
            query_ids.unsqueeze(-1).expand(-1, -1, config.value_dim),
        )
    else:
        query_ids = torch.arange(num_items, device=config.device).expand(
            config.batch_size,
            num_items,
        )
        queries = keys.clone()
        targets = values.clone()

    collision_pair_ids = torch.stack(
        [
            torch.arange(0, num_items, 2, device=config.device),
            torch.arange(1, num_items, 2, device=config.device),
        ],
        dim=-1,
    ).expand(config.batch_size, config.num_pairs, 2)
    collision_cosines = torch.matmul(keys, keys.transpose(1, 2))

    return KeyCollisionBatch(
        keys=keys,
        values=values,
        queries=queries,
        targets=targets,
        query_ids=query_ids,
        collision_pair_ids=collision_pair_ids,
        collision_cosines=collision_cosines,
        config=config,
    )


@dataclass(frozen=True)
class OverwriteTaskConfig:
    """Configuration for same-key overwrite/recency recall."""

    batch_size: int = 8
    num_keys: int = 8
    num_updates: int = 3
    key_dim: int = 32
    value_dim: int = 32
    seed: int = 0
    device: DeviceLike = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_keys <= 0:
            raise ValueError("num_keys must be positive")
        if self.num_updates < 2:
            raise ValueError("num_updates must be at least 2")
        if self.key_dim <= 0 or self.value_dim <= 0:
            raise ValueError("key_dim and value_dim must be positive")


@dataclass
class OverwriteTaskBatch:
    """Batch produced by :func:`generate_overwrite_batch`.

    ``sequence_keys`` contains every key once per update round. The final target
    is the latest value for each key; ``stale_targets`` stores all earlier
    values to make stale-recall leakage explicit.
    """

    sequence_keys: torch.Tensor
    sequence_values: torch.Tensor
    queries: torch.Tensor
    final_targets: torch.Tensor
    stale_targets: torch.Tensor
    key_ids: torch.Tensor
    update_indices: torch.Tensor
    support_keys: torch.Tensor
    support_values: torch.Tensor
    config: OverwriteTaskConfig


def generate_overwrite_batch(config: OverwriteTaskConfig) -> OverwriteTaskBatch:
    """Generate repeated writes to the same keys with changing values."""
    generator = _make_generator(config.seed, config.device)
    support_keys = _normalize(
        _randn(
            (config.batch_size, config.num_keys, config.key_dim),
            generator,
            config.device,
        )
    )
    support_values = _randn(
        (
            config.batch_size,
            config.num_keys,
            config.num_updates,
            config.value_dim,
        ),
        generator,
        config.device,
    )

    sequence_keys = support_keys.repeat(1, config.num_updates, 1)
    sequence_values = support_values.permute(0, 2, 1, 3).reshape(
        config.batch_size,
        config.num_updates * config.num_keys,
        config.value_dim,
    )
    key_ids = torch.arange(config.num_keys, device=config.device).repeat(
        config.num_updates,
    ).expand(
        config.batch_size,
        config.num_updates * config.num_keys,
    )
    update_indices = torch.arange(
        config.num_updates,
        device=config.device,
    ).repeat_interleave(config.num_keys).expand(
        config.batch_size,
        config.num_updates * config.num_keys,
    )

    return OverwriteTaskBatch(
        sequence_keys=sequence_keys,
        sequence_values=sequence_values,
        queries=support_keys.clone(),
        final_targets=support_values[:, :, -1].clone(),
        stale_targets=support_values[:, :, :-1].clone(),
        key_ids=key_ids,
        update_indices=update_indices,
        support_keys=support_keys,
        support_values=support_values,
        config=config,
    )


@dataclass(frozen=True)
class NoveltyProbeConfig:
    """Configuration for known / colliding / unseen query probes."""

    batch_size: int = 8
    support_size: int = 16
    key_dim: int = 32
    value_dim: int = 32
    collision_cosine: float = 0.9
    seed: int = 0
    device: DeviceLike = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.support_size <= 0:
            raise ValueError("support_size must be positive")
        if self.key_dim < 2:
            raise ValueError("key_dim must be at least 2")
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        if not -1.0 <= self.collision_cosine <= 1.0:
            raise ValueError("collision_cosine must be in [-1, 1]")


@dataclass
class NoveltyProbeBatch:
    """Known, near-collision, and unseen queries against a support set.

    ``query_types`` uses integer labels: ``0`` known, ``1`` collision, and
    ``2`` unseen. Unseen probes have ``support_indices == -1`` and zero targets.
    """

    support_keys: torch.Tensor
    support_values: torch.Tensor
    queries: torch.Tensor
    targets: torch.Tensor
    query_types: torch.Tensor
    support_indices: torch.Tensor
    config: NoveltyProbeConfig


def generate_novelty_probe_batch(
    config: NoveltyProbeConfig,
) -> NoveltyProbeBatch:
    """Generate probes that separate recall from key-collision and novelty."""
    generator = _make_generator(config.seed, config.device)
    support_keys = _normalize(
        _randn(
            (config.batch_size, config.support_size, config.key_dim),
            generator,
            config.device,
        )
    )
    support_values = _randn(
        (config.batch_size, config.support_size, config.value_dim),
        generator,
        config.device,
    )

    colliding_queries = torch.empty_like(support_keys)
    for batch_idx in range(config.batch_size):
        for support_idx in range(config.support_size):
            noise = _randn((config.key_dim,), generator, config.device)
            colliding_queries[batch_idx, support_idx] = _colliding_key(
                support_keys[batch_idx, support_idx],
                noise,
                config.collision_cosine,
            )

    unseen_queries = _normalize(
        _randn(
            (config.batch_size, config.support_size, config.key_dim),
            generator,
            config.device,
        )
    )

    support_indices = torch.arange(
        config.support_size,
        device=config.device,
    ).expand(config.batch_size, config.support_size)
    unseen_indices = torch.full_like(support_indices, -1)

    queries = torch.cat(
        [support_keys, colliding_queries, unseen_queries],
        dim=1,
    )
    targets = torch.cat(
        [
            support_values,
            support_values,
            torch.zeros_like(support_values),
        ],
        dim=1,
    )
    query_types = torch.cat(
        [
            torch.zeros_like(support_indices),
            torch.ones_like(support_indices),
            torch.full_like(support_indices, 2),
        ],
        dim=1,
    )
    repeated_support_indices = torch.cat(
        [support_indices, support_indices, unseen_indices],
        dim=1,
    )

    return NoveltyProbeBatch(
        support_keys=support_keys,
        support_values=support_values,
        queries=queries,
        targets=targets,
        query_types=query_types,
        support_indices=repeated_support_indices,
        config=config,
    )
