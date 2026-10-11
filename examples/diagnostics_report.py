"""Print a state/gate diagnostics report for the Phase 4 variants (issue #27).

Runs each registered Phase 4 variant once on synthetic data with
``return_trace=True``, reconstructs the state trajectory from the captured
tensors via :func:`replay_state_trajectory`, and prints the JSON summary from
:func:`summarize_trajectory`. Later experiments (#28 calibration, #38 hybrid
routing) can reuse this pattern on their own captured trajectories — the
diagnostics never rerun the model.

Usage:
    python examples/diagnostics_report.py
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from kda.instrumentation import replay_state_trajectory, summarize_trajectory
from kda.variants import create_variant


def _synthetic_inputs(batch_size=2, seq_len=32, head_dim=32, value_dim=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    queries = F.normalize(
        torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1
    )
    keys = F.normalize(
        torch.randn(batch_size, seq_len, head_dim, generator=generator), dim=-1
    )
    values = torch.randn(batch_size, seq_len, value_dim, generator=generator)
    gates = {
        "alpha": torch.sigmoid(
            torch.randn(batch_size, seq_len, head_dim, generator=generator)
        ),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, 1, generator=generator)
        ),
    }
    return queries, keys, values, gates


def _write_vectors(name: str, keys: torch.Tensor, trace: dict) -> torch.Tensor | None:
    """Per-step write vector w_t for state replay, or None if unavailable."""
    if isinstance(trace.get("gain"), torch.Tensor):
        return trace["gain"]  # precision-gated KDA: w_t = g_t
    if isinstance(trace.get("beta_eff"), torch.Tensor):
        return trace["beta_eff"] * keys  # KDA-RLS-lite: w_t = beta_eff,t * k_t
    if isinstance(trace.get("beta_raw"), torch.Tensor):
        return trace["beta_raw"] * keys  # baseline KDA: w_t = beta_t * k_t
    return None


def main() -> None:
    torch.manual_seed(0)
    queries, keys, values, gates = _synthetic_inputs()
    head_dim, value_dim = keys.shape[-1], values.shape[-1]

    for name in ("kda-rls-lite", "precision-gated-kda"):
        variant = create_variant(name, head_dim=head_dim, value_dim=value_dim)
        result = variant(queries, keys, values, gates=gates, return_trace=True)
        write_vectors = _write_vectors(name, keys, result.trace)
        states = (
            replay_state_trajectory(keys, values, gates["alpha"], write_vectors)
            if write_vectors is not None
            else None
        )
        report = summarize_trajectory(
            states=states, trace=result.trace, keys=keys
        )
        print(f"=== {name} ===")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
