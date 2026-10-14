"""Rule-level tests for the Erase-then-Delta Attention baseline (issue #37).

The EDA recurrence under test (reference baseline from arXiv:2606.26560,
not a novel contribution of this repository):

    S_t = (I − β_t k_t k_tᵀ) (I − γ_t e_t e_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
    o_t = S_tᵀ q_t

Factors apply right to left: decay, then the independently addressed erase,
then the standard delta corrective write. Tests combine hardcoded
hand-computed steps, an independent pure-Python reference implementation,
the γ = 0 / missing-gate reductions to KDA, and the paper's readout-level
erase property (its Eq. 12).
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
        "erase_address": torch.randn(batch_size, seq_len, head_dim, generator=generator),
        "erase_gate": torch.rand(batch_size, seq_len, 1, generator=generator),
    }
    return queries, keys, values, gates


def _reference_eda_step(state, k, v, alpha, beta, e, gamma):
    """One EDA step in pure Python lists (independent of torch).

    ``e`` must already be L2-normalized; ``gamma``/``beta`` are scalars.
    """
    d_h = len(state)
    d_v = len(state[0])
    # 1) Decay.
    decayed = [[alpha[i] * state[i][j] for j in range(d_v)] for i in range(d_h)]
    # 2) Erase: subtract γ e (eᵀ decayed).
    erased = [sum(e[i] * decayed[i][j] for i in range(d_h)) for j in range(d_v)]
    after_erase = [
        [decayed[i][j] - gamma * e[i] * erased[j] for j in range(d_v)]
        for i in range(d_h)
    ]
    # 3) Delta write: subtract β k (kᵀ after_erase), then add β k vᵀ.
    correction = [sum(k[i] * after_erase[i][j] for i in range(d_h)) for j in range(d_v)]
    return [
        [
            after_erase[i][j] - beta * k[i] * correction[j] + beta * k[i] * v[j]
            for j in range(d_v)
        ]
        for i in range(d_h)
    ]


def test_eda_is_registered():
    assert "eda" in available_variants()


def test_eda_output_shapes_and_trace():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("eda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates, return_trace=True)

    assert result.outputs.shape == (2, 8, 12)
    assert result.final_state.shape == (2, 16, 12)
    assert result.trace["erase_address"].shape == (2, 8, 16)
    assert result.trace["erase_gate"].shape == (2, 8, 1)
    assert result.trace["erased_response"].shape == (2, 8, 12)
    # Erase addresses are L2-normalized internally.
    norms = result.trace["erase_address"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_eda_without_erase_gates_matches_kda():
    """Omitting the erase pair recovers the KDA update exactly."""
    queries, keys, values, gates = _synthetic_inputs()
    kda_gates = {"alpha": gates["alpha"], "beta": gates["beta"]}

    variant = create_variant("eda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=kda_gates, return_trace=True)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries, keys, values, gates["alpha"], gates["beta"]
    )

    assert result.trace["erase_address"] is None
    assert result.trace["erase_gate"] is None
    assert torch.allclose(result.outputs, expected_outputs)
    assert torch.allclose(result.final_state, expected_state)


def test_eda_zero_erase_gate_matches_kda():
    """γ = 0 makes the erase factor the identity, recovering KDA."""
    queries, keys, values, gates = _synthetic_inputs()
    gates["erase_gate"] = torch.zeros_like(gates["erase_gate"])

    variant = create_variant("eda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates)

    core = KDACore(head_dim=16, value_dim=12)
    expected_outputs, expected_state = core(
        queries, keys, values, gates["alpha"], gates["beta"]
    )

    assert torch.allclose(result.outputs, expected_outputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(result.final_state, expected_state, atol=1e-6, rtol=1e-5)


def test_eda_hand_computed_single_step():
    """Hardcoded numbers computed by hand from the paper recurrence.

    Erase address e = [1, 0] wipes row 1 of the (undecayed) state while the
    delta write at k = [0, 1] replaces row 2 — erase and write act at
    different addresses in one step, which is the point of the paper.
    """
    state0 = [[1.0, 2.0], [3.0, 4.0]]
    e = [1.0, 0.0]
    k = [0.0, 1.0]
    q = [0.0, 1.0]
    v = [5.0, 7.0]
    # Hand derivation (α = 1, γ = 1, β = 1):
    #   S_hat       = S_0
    #   eᵀS_hat     = [1, 2]          → after erase: [[0, 0], [3, 4]]
    #   kᵀS_erased  = [3, 4]          → after correction: [[0, 0], [0, 0]]
    #   + k vᵀ                        → S_1 = [[0, 0], [5, 7]]
    #   o_1 = S_1ᵀ q = [5, 7]
    expected_state = torch.tensor([[0.0, 0.0], [5.0, 7.0]])
    expected_output = torch.tensor([5.0, 7.0])

    variant = create_variant("eda", head_dim=2, value_dim=2)
    result = variant(
        queries=torch.tensor([[q]]),
        keys=torch.tensor([[k]]),
        values=torch.tensor([[v]]),
        gates={
            "alpha": torch.ones(1, 1, 2),
            "beta": torch.ones(1, 1, 1),
            "erase_address": torch.tensor([[e]]),
            "erase_gate": torch.ones(1, 1, 1),
        },
        initial_state=torch.tensor([state0]),
    )

    assert torch.allclose(result.final_state[0], expected_state, atol=1e-6)
    assert torch.allclose(result.outputs[0, 0], expected_output, atol=1e-6)


def test_eda_matches_pure_python_reference_over_short_sequence():
    """Five-step random sequence checked against the list-based reference."""
    torch.manual_seed(11)
    batch, seq_len, head_dim, value_dim = 2, 5, 3, 4
    queries = F.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    keys = F.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    values = torch.randn(batch, seq_len, value_dim)
    gates = {
        "alpha": torch.rand(batch, seq_len, head_dim),
        "beta": torch.rand(batch, seq_len, 1),
        "erase_address": torch.randn(batch, seq_len, head_dim),
        "erase_gate": torch.rand(batch, seq_len, 1),
    }

    variant = create_variant("eda", head_dim=head_dim, value_dim=value_dim)
    result = variant(queries, keys, values, gates=gates)

    normalized_addresses = F.normalize(gates["erase_address"], p=2, dim=-1)
    for b_idx in range(batch):
        state = [[0.0] * value_dim for _ in range(head_dim)]
        for t in range(seq_len):
            state = _reference_eda_step(
                state,
                keys[b_idx, t].tolist(),
                values[b_idx, t].tolist(),
                gates["alpha"][b_idx, t].tolist(),
                gates["beta"][b_idx, t, 0].item(),
                normalized_addresses[b_idx, t].tolist(),
                gates["erase_gate"][b_idx, t, 0].item(),
            )
        expected = torch.tensor(state)
        assert torch.allclose(result.final_state[b_idx], expected, atol=1e-5)


def test_eda_erases_stale_address_while_writing_elsewhere():
    """The paper's core capability on a tiny synthetic scene: write
    (k1 → v1), then in one step erase at e = k1 (γ = 1) while writing
    (k2 → v2) at an orthogonal key. Afterwards k1 reads back ≈ 0 and
    k2 reads back ≈ v2."""
    k1 = [1.0, 0.0, 0.0, 0.0]
    k2 = [0.0, 1.0, 0.0, 0.0]
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]

    gates = {
        "alpha": torch.ones(1, 2, 4),
        "beta": torch.ones(1, 2, 1),
        "erase_address": torch.tensor([[k2, k1]]),  # step 1: nothing to erase
        "erase_gate": torch.tensor([[[0.0], [1.0]]]),
    }
    variant = create_variant("eda", head_dim=4, value_dim=3)
    result = variant(
        queries=torch.tensor([[k1, k2]]),
        keys=torch.tensor([[k1, k2]]),
        values=torch.tensor([[v1, v2]]),
        gates=gates,
    )

    state = result.final_state[0]
    read_k1 = state.T @ torch.tensor(k1)
    read_k2 = state.T @ torch.tensor(k2)
    assert torch.allclose(read_k1, torch.zeros(3), atol=1e-6)
    assert torch.allclose(read_k2, torch.tensor(v2), atol=1e-6)


def test_eda_erase_leaves_orthogonal_readouts_unchanged():
    """Paper Eq. 12: queries orthogonal to the erase address are unaffected
    by the erase step; the response at the erase address scales by 1 − γ."""
    torch.manual_seed(5)
    state0 = torch.randn(1, 4, 3)
    e = [1.0, 0.0, 0.0, 0.0]
    q_perp = [0.0, 1.0, 0.0, 0.0]
    gamma = 0.7
    base_gates = {"alpha": torch.ones(1, 1, 4), "beta": torch.zeros(1, 1, 1)}
    erase_gates = {
        **base_gates,
        "erase_address": torch.tensor([[e]]),
        "erase_gate": torch.tensor([[[gamma]]]),
    }

    variant = create_variant("eda", head_dim=4, value_dim=3)
    k = [0.0, 0.0, 1.0, 0.0]
    v = torch.zeros(1, 1, 3)

    clean = variant(
        torch.tensor([[q_perp]]), torch.tensor([[k]]), v,
        gates=base_gates, initial_state=state0,
    )
    erased_perp = variant(
        torch.tensor([[q_perp]]), torch.tensor([[k]]), v,
        gates=erase_gates, initial_state=state0,
    )
    erased_along = variant(
        torch.tensor([[e]]), torch.tensor([[k]]), v,
        gates=erase_gates, initial_state=state0,
    )

    assert torch.allclose(erased_perp.outputs, clean.outputs, atol=1e-6)
    expected_along = (1.0 - gamma) * (state0[0].T @ torch.tensor(e))
    assert torch.allclose(erased_along.outputs[0, 0], expected_along, atol=1e-6)


def test_eda_same_address_collapse_matches_closed_form():
    """With e = k, the combined factor is I − (β + γ − βγ) k kᵀ for
    normalized keys, while the write term stays β k vᵀ (paper §3.3)."""
    torch.manual_seed(13)
    batch, seq_len, head_dim, value_dim = 2, 4, 8, 6
    keys = F.normalize(torch.randn(batch, seq_len, head_dim), dim=-1)
    values = torch.randn(batch, seq_len, value_dim)
    alpha = torch.rand(batch, seq_len, head_dim)
    beta = torch.rand(batch, seq_len, 1)
    gamma = torch.rand(batch, seq_len, 1)

    variant = create_variant("eda", head_dim=head_dim, value_dim=value_dim)
    result = variant(
        keys,  # queries unused for the state comparison
        keys,
        values,
        gates={
            "alpha": alpha,
            "beta": beta,
            "erase_address": keys.clone(),
            "erase_gate": gamma,
        },
    )

    # Independent closed-form rollout of the collapsed rule.
    state = torch.zeros(batch, head_dim, value_dim)
    for t in range(seq_len):
        k_t = keys[:, t, :]
        v_t = values[:, t, :]
        beta_eff = beta[:, t, :] + gamma[:, t, :] - beta[:, t, :] * gamma[:, t, :]
        state = alpha[:, t, :].unsqueeze(-1) * state
        k_state = torch.bmm(k_t.unsqueeze(-1), torch.bmm(k_t.unsqueeze(1), state))
        state = state - beta_eff.unsqueeze(-1) * k_state
        state = state + beta[:, t, :].unsqueeze(-1) * torch.bmm(
            k_t.unsqueeze(-1), v_t.unsqueeze(1)
        )

    assert torch.allclose(result.final_state, state, atol=1e-5)


def test_eda_runs_on_shared_overwrite_task_path():
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

    for name in ("kda", "eda"):
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


def test_eda_gradient_flows_through_gates():
    queries, keys, values, gates = _synthetic_inputs(seq_len=4)
    for tensor in gates.values():
        tensor.requires_grad_(True)

    variant = create_variant("eda", head_dim=16, value_dim=12)
    result = variant(queries, keys, values, gates=gates)
    result.outputs.sum().backward()

    assert gates["beta"].grad is not None
    assert gates["erase_address"].grad is not None
    assert gates["erase_gate"].grad is not None
    assert torch.isfinite(gates["erase_address"].grad).all()
    assert torch.isfinite(gates["erase_gate"].grad).all()


def test_eda_gate_validation():
    queries, keys, values, gates = _synthetic_inputs()
    variant = create_variant("eda", head_dim=16, value_dim=12)

    # Missing beta.
    try:
        variant(queries, keys, values, gates={"alpha": gates["alpha"]})
    except ValueError as exc:
        assert "alpha" in str(exc) and "beta" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing beta")

    # Only one of the erase pair supplied.
    try:
        variant(
            queries,
            keys,
            values,
            gates={
                "alpha": gates["alpha"],
                "beta": gates["beta"],
                "erase_gate": gates["erase_gate"],
            },
        )
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("expected ValueError for unpaired erase gates")

    # Wrong erase-address shape.
    try:
        variant(
            queries,
            keys,
            values,
            gates={
                "alpha": gates["alpha"],
                "beta": gates["beta"],
                "erase_address": gates["erase_address"][..., :4],
                "erase_gate": gates["erase_gate"],
            },
        )
    except ValueError as exc:
        assert "erase_address" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad erase-address shape")
