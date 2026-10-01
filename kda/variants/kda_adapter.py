"""KDA adapter for the shared delta-memory variant interface."""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from kda.core import KDACore

from .base import DeltaMemoryOutput, DeltaMemoryVariant, register_variant


@register_variant("kda")
class KDAVariant(DeltaMemoryVariant):
    """Compatibility wrapper around the existing :class:`KDACore`.

    This adapter keeps the public benchmark interface stable while leaving the
    educational KDA implementation unchanged.
    """

    def __init__(self, head_dim: int, value_dim: int, eps: float = 1e-6) -> None:
        super().__init__(head_dim=head_dim, value_dim=value_dim)
        self.core = KDACore(head_dim=head_dim, value_dim=value_dim, eps=eps)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: Optional[Mapping[str, torch.Tensor]] = None,
        initial_state: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> DeltaMemoryOutput:
        self._validate_inputs(queries, keys, values)
        if gates is None:
            raise ValueError("KDAVariant requires gates containing 'alpha' and 'beta'")
        if "alpha" not in gates or "beta" not in gates:
            raise ValueError("KDAVariant requires gates containing 'alpha' and 'beta'")

        alpha = gates["alpha"]
        beta = gates["beta"]
        outputs, final_state = self.core(
            queries=queries,
            keys=keys,
            values=values,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
        )
        trace = {"alpha": alpha, "beta": beta} if return_trace else {}
        return DeltaMemoryOutput(
            outputs=outputs,
            final_state=final_state,
            trace=trace,
        )
