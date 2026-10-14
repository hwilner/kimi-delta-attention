"""Rule-level tests for the Gated DeltaNet-2 reference baseline (issue #37).

The GDN-2 recurrence under test (reference baseline from arXiv:2605.22791,
not a novel contribution of this repository):

    e_t = b_t ⊙ k_t
    z_t = w_t ⊙ v_t
    S_t = (I − k_t e_tᵀ) Diag(α_t) S_{t−1} + k_t z_tᵀ
    o_t = S_tᵀ q_t

Tests combine hardcoded hand-computed steps, an independent pure-Python
reference implementation of the same recurrence, and the paper's tied-gate
reduction to KDA (Appendix A.5 of arXiv:2605.22791).
"""

import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.tasks import OverwriteTaskConfig, generate_overwrite_batch
from kda.variants import available_variants, create_variant


def _synthetic_inputs(batch_size=2, seq_len=8, head_dim=16, value_dim=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1)
    keys = F.normalize(torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1)
    values = torch.randn(batch_size, seq_len, value_dim, generator=generator)
    gates = {
        "alpha": torch.sigmoid(torch.randn(batch_size, seq_len, head_dim, generator=generator)),
        "beta": torch.sigmoid(torch.randn(batch_size, seq_len, 1, generator=generator)),
        "erase": torch.sigmoid(torch.randn(batch_size, seq_len, head_dim, generator=generator)),
        "write": torch.sigmoid(torch.randn(batch_size, seq_len, value_dim, generator=generator)),
    }
    return queries, keys, values, gates


def _reference_gdn2_step(state, k, v, alpha, b, w):
    """One GDN-2 step in pure Python lists (independent of torch)."""
    d_h = len(state)
    d_v = len(state[0])
    # Decay.
    decayed = [[alpha[i] * state[i][j] for j in range(d_v)] for i in range(d_h)]
    # Erase: e = b ⊙ k; subtract k (eᵀ decayed).
    e = [b[i] * k[i] for i in range(d_h)]
    erased = [sum(e[i] * decayed[i][j] for i in range(d_h)) for j in range(d_v)]
    after_erase = [
        [decayed[i][j] - k[i] * erased[j] for j in range(d_v)] for i in range(d_h)
    ]
    # Write: z = w ⊙ v; add k zᵀ.
    z = [w[j] * v[j] for j in range(d_v)]
    return [
        [after_erase[i][j] + k[i] * z[j] for j in range(d_v)] for i in range(d_h)
    ]


def test_gdn2_is_registered():
    assert "gdn-2" in available_variants()


def test_gdn2_output_shapes_and_trace():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("gdn-2", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert result.outputs.shape == (2, 8, 12)
    assert result.final_state.shape == (2, 16, 12)
    assert result.trace["erase_gate"].shape == (2, 8, 16)
    assert result.trace["write_gate"].shape == (2, 8, 12)
    assert result.trace["tied_from_beta"] is False


def test_gdn2_tied_gates_reduce_to_kda():
    """Paper's reduction: b_t = w_t = β_t·1 recovers the KDA update."""
    for seed in range(3):
        queries, keys, values, gates = _synthetic_inputs(seed=seed)
        tied = {
            "alpha": gates["alpha"],
            "erase": gates["beta"].expand(2, 8, 16),
            "write": gates["beta"].expand(2, 8, 12),
        }

        variant = create_variant("gdn-2", head_dim=16, value_dim=12)
        result = variant(queries, keys, values, gates=tied)

        core = KDACore(head_dim=16, value_dim=12)
        expected_outputs, expected_state = core(
            queries, keys, values, gates["alpha"], gates["beta"]
        )

        assert torch.allclose(result.outputs, expected_outputs, atol=1e-6, rtol=1e-5)
        assert torch.allclose(result.final_state, expected_state, atol=1e-6, rtol=1e-5)


def test_gdn2_beta_fallback_matches_kda():
    """With only alpha/beta supplied, beta is broadcast to both gate axes."""
    queries, keys, values, gates = _synthetic_inputs()
    fallback_gates = {"alpha": gates["alpha"], "beta": gates["beta"]}

    variant = create_variant("gdn-2", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=fallback_gates, return_trace=True)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries, keys, values, gates["alpha"], gates["beta"]
    )

    assert result.trace["tied_from_beta"] is True
    assert torch.allclose(result.outputs, expected_outputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(result.final_state, expected_state, atol=1e-6, rtol=1e-5)


def test_gdn2_hand_computed_single_step():
    """Hardcoded numbers computed by hand from the paper recurrence."""
    state0 = [[1.0, 2.0], [3.0, 4.0]]
    k = [0.6, 0.8]
    q = [0.6, 0.8]
    v = [10.0, 20.0]
    alpha = [0.5, 1.0]
    b = [0.5, 0.25]
    w = [1.0, 0.5]

    # Hand derivation:
    #   S_bar  = Diag(α) S_0          = [[0.5, 1.0], [3.0, 4.0]]
    #   e      = b ⊙ k                = [0.3, 0.2]
    #   eᵀS_bar                       = [0.75, 1.1]
    #   after erase                   = [[0.05, 0.34], [2.4, 3.12]]
    #   z      = w ⊙ v                = [10, 10]
    #   S_1                           = [[6.05, 6.34], [10.4, 11.12]]
    #   o_1    = S_1ᵀ q               = [11.95, 12.7]
    expected_state = torch.tensor([[6.05, 6.34], [10.4, 11.12]])
    expected_output = torch.tensor([11.95, 12.7])

    variant = create_variant("gdn-2", head_dim=2, value_dim=2)
    result = variant(
        queries=torch.tensor([[q]]),
        keys=torch.tensor([[k]]),
        values=torch.tensor([[v]]),
        gates={
            "alpha": torch.tensor([[alpha]]),
            "erase": torch.tensor([[b]]),
            "write": torch.tensor([[w]]),
        },
        initial_state=torch.tensor([state0]),
    )

    assert torch.allclose(result.final_state[0], expected_state, atol=1e-6)
    assert torch.allclose(result.outputs[0, 0], expected_output, atol=1e-6)


def test_gdn2_matches_pure_python_reference_over_short_sequence():
    """Five-step random sequence checked against the list-based reference."""
    torch.manual_seed(7)
    batch, seq_len, head_dim, value_dim = 2, 5, 3, 4
    queries = F.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    keys = F.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    values = torch.randn(batch, seq_len, value_dim)
    gates = {
        "alpha": torch.rand(batch, seq_len, head_dim),
        "erase": torch.rand(batch, seq_len, head_dim),
        "write": torch.rand(batch, seq_len, value_dim),
    }

    variant = create_variant("gdn-2", head_dim=head_dim, value_dim=value_dim)
    result = variant(queries, keys, values, gates=gates)

    for b_idx in range(batch):
        state = [[0.0] * value_dim for _ in range(head_dim)]
        for t in range(seq_len):
            state = _reference_gdn2_step(
                state,
                keys[b_idx, t].tolist(),
                values[b_idx, t].tolist(),
                gates["alpha"][b_idx, t].tolist(),
                gates["erase"][b_idx, t].tolist(),
                gates["write"][b_idx, t].tolist(),
            )
        expected = torch.tensor(state)
        assert torch.allclose(result.final_state[b_idx], expected, atol=1e-5)


def test_gdn2_zero_write_gate_commits_no_new_content():
    """With w = 0 the update adds nothing new; with b = 0 and α = 1 the
    state must be exactly unchanged."""
    torch.manual_seed(3)
    state0 = torch.randn(1, 4, 3)
    queries = F.normalize(torch.randn(1, 2, 4), dim=-1)
    keys = F.normalize(torch.randn(1, 2, 4), dim=-1)
    values = torch.randn(1, 2, 3)
    gates = {
        "alpha": torch.ones(1, 2, 4),
        "erase": torch.zeros(1, 2, 4),
        "write": torch.zeros(1, 2, 3),
    }

    variant = create_variant("gdn-2", head_dim=4, value_dim=3)
    result = variant(queries, keys, values, gates=gates, initial_state=state0)

    assert torch.allclose(result.final_state, state0, atol=1e-7)
    # Readout equals the untouched state read at each query.
    for t in range(2):
        expected = state0[0].T @ queries[0, t]
        assert torch.allclose(result.outputs[0, t], expected, atol=1e-6)


def test_gdn2_zero_erase_gate_leaves_decayed_state_then_writes():
    """With b = 0, one step is S_1 = Diag(α) S_0 + k (w ⊙ v)ᵀ."""
    k = [0.6, 0.8]
    v = [10.0, 20.0]
    alpha = [0.5, 1.0]
    w = [1.0, 0.5]
    state0 = [[1.0, 2.0], [3.0, 4.0]]
    # S_1 = [[0.5, 1.0], [3.0, 4.0]] + [[6, 6], [8, 8]]
    expected_state = torch.tensor([[6.5, 7.0], [11.0, 12.0]])

    variant = create_variant("gdn-2", head_dim=2, value_dim=2)
    result = variant(
        queries=torch.tensor([[k]]),
        keys=torch.tensor([[k]]),
        values=torch.tensor([[v]]),
        gates={
            "alpha": torch.tensor([[alpha]]),
            "erase": torch.zeros(1, 1, 2),
            "write": torch.tensor([[w]]),
        },
        initial_state=torch.tensor([state0]),
    )

    assert torch.allclose(result.final_state[0], expected_state, atol=1e-6)


def test_gdn2_runs_on_shared_overwrite_task_path():
    """Baselines share the KDA task/evaluation path (issue #37 criterion):
    the same overwrite-task batch and alpha/beta gates drive all variants
    through the same write-then-read loop."""
    config = OverwriteTaskConfig(
        batch_size=2, num_keys=4, num_updates=2, key_dim=16, value_dim=12, seed=1
    )
    task = generate_overwrite_batch(config)
    seq_len = task.sequence_keys.shape[1]
    generator = torch.Generator().manual_seed(0)
    write_gates = {
        "alpha": torch.sigmoid(torch.randn(2, seq_len, 16, generator=generator)),
        "beta": torch.sigmoid(torch.randn(2, seq_len, 1, generator=generator)),
    }
    # Readout phase: frozen state (no decay, no write), one step per query.
    read_gates = {
        "alpha": torch.ones(2, config.num_keys, 16),
        "beta": torch.zeros(2, config.num_keys, 1),
    }

    for name in ("kda", "gdn-2"):
        variant = create_variant(name, head_dim=16, value_dim=12)
        written = variant(
            task.sequence_keys,
            task.sequence_keys,
            task.sequence_values,
            gates=write_gates,
        )
        assert written.outputs.shape == (2, seq_len, 12)
        assert written.final_state.shape == (2, 16, 12)

        read = variant(
            task.queries,
            task.queries,
            torch.zeros_like(task.final_targets),
            gates=read_gates,
            initial_state=written.final_state,
        )
        assert read.outputs.shape == task.final_targets.shape


def test_gdn2_gradient_flows_through_gates():
    queries, keys, values, gates = _synthetic_inputs(seq_len=4)
    for tensor in gates.values():
        tensor.requires_grad_(True)

    variant = create_variant("gdn-2", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates)
    result.outputs.sum().backward()

    assert gates["alpha"].grad is not None
    assert gates["erase"].grad is not None
    assert gates["write"].grad is not None
    assert torch.isfinite(gates["erase"].grad).all()
    assert torch.isfinite(gates["write"].grad).all()


def test_gdn2_gate_validation():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("gdn-2", head_dim=16, value_dim=12)

    # Missing alpha entirely.
    try:
        variant(queries, keys, values, gates={"beta": gates["beta"]})
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing alpha")

    # Only one of erase/write supplied.
    try:
        variant(
            queries,
            keys,
            values,
            gates={"alpha": gates["alpha"], "erase": gates["erase"]},
        )
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("expected ValueError for unpaired erase/write gates")

    # No beta to fall back on.
    try:
        variant(queries, keys, values, gates={"alpha": gates["alpha"]})
    except ValueError as exc:
        assert "beta" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing beta fallback")

    # Wrong erase-gate shape.
    try:
        variant(
            queries,
            keys,
            values,
            gates={
                "alpha": gates["alpha"],
                "erase": gates["erase"][..., :4],
                "write": gates["write"],
            },
        )
    except ValueError as exc:
        assert "erase gate" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad erase-gate shape")
