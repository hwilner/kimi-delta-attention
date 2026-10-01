"""Tests for synthetic recurrent-memory stress-task generators."""

import torch

from kda.tasks import (
    KeyCollisionTaskConfig,
    NoveltyProbeConfig,
    OverwriteTaskConfig,
    generate_key_collision_batch,
    generate_novelty_probe_batch,
    generate_overwrite_batch,
)


def test_key_collision_batch_shapes_and_exact_pair_cosines():
    config = KeyCollisionTaskConfig(
        batch_size=3,
        num_pairs=5,
        key_dim=16,
        value_dim=7,
        collision_cosine=0.75,
        seed=123,
        shuffle_queries=False,
    )
    batch = generate_key_collision_batch(config)

    assert batch.keys.shape == (3, 10, 16)
    assert batch.values.shape == (3, 10, 7)
    assert batch.queries.shape == (3, 10, 16)
    assert batch.targets.shape == (3, 10, 7)
    assert batch.collision_pair_ids.shape == (3, 5, 2)
    assert batch.collision_cosines.shape == (3, 10, 10)

    for batch_idx in range(config.batch_size):
        for pair_idx in range(config.num_pairs):
            left, right = batch.collision_pair_ids[batch_idx, pair_idx]
            cosine = batch.collision_cosines[batch_idx, left, right]
            assert torch.isclose(
                cosine,
                torch.tensor(config.collision_cosine),
                atol=1e-6,
            )


def test_key_collision_batch_is_deterministic():
    config = KeyCollisionTaskConfig(seed=99, shuffle_queries=True)
    first = generate_key_collision_batch(config)
    second = generate_key_collision_batch(config)

    assert torch.allclose(first.keys, second.keys)
    assert torch.allclose(first.values, second.values)
    assert torch.equal(first.query_ids, second.query_ids)


def test_overwrite_batch_shapes_and_final_targets():
    config = OverwriteTaskConfig(
        batch_size=2,
        num_keys=4,
        num_updates=3,
        key_dim=8,
        value_dim=5,
        seed=7,
    )
    batch = generate_overwrite_batch(config)

    assert batch.sequence_keys.shape == (2, 12, 8)
    assert batch.sequence_values.shape == (2, 12, 5)
    assert batch.queries.shape == (2, 4, 8)
    assert batch.final_targets.shape == (2, 4, 5)
    assert batch.stale_targets.shape == (2, 4, 2, 5)
    assert batch.key_ids.shape == (2, 12)
    assert batch.update_indices.shape == (2, 12)

    assert torch.allclose(
        batch.final_targets,
        batch.support_values[:, :, -1],
    )
    assert torch.equal(batch.update_indices[0, :4], torch.zeros(4, dtype=torch.long))
    assert torch.equal(batch.update_indices[0, -4:], torch.full((4,), 2))


def test_overwrite_batch_is_deterministic():
    config = OverwriteTaskConfig(seed=42)
    first = generate_overwrite_batch(config)
    second = generate_overwrite_batch(config)

    assert torch.allclose(first.sequence_keys, second.sequence_keys)
    assert torch.allclose(first.sequence_values, second.sequence_values)


def test_novelty_probe_labels_and_targets():
    config = NoveltyProbeConfig(
        batch_size=2,
        support_size=6,
        key_dim=12,
        value_dim=4,
        collision_cosine=-0.25,
        seed=5,
    )
    batch = generate_novelty_probe_batch(config)

    assert batch.queries.shape == (2, 18, 12)
    assert batch.targets.shape == (2, 18, 4)
    assert batch.query_types.shape == (2, 18)
    assert batch.support_indices.shape == (2, 18)

    known = batch.query_types == 0
    collision = batch.query_types == 1
    unseen = batch.query_types == 2

    assert known.sum().item() == 12
    assert collision.sum().item() == 12
    assert unseen.sum().item() == 12
    assert torch.all(batch.support_indices[unseen] == -1)
    assert torch.allclose(
        batch.targets[unseen],
        torch.zeros_like(batch.targets[unseen]),
    )

    collision_cosines = torch.sum(
        batch.queries[:, 6:12] * batch.support_keys,
        dim=-1,
    )
    assert torch.allclose(
        collision_cosines,
        torch.full_like(collision_cosines, config.collision_cosine),
        atol=1e-6,
    )
