"""Shared interface for delta-memory research variants.

The interface is intentionally small: each variant consumes normalized query/key
vectors, values, a mapping of gate tensors, and an optional recurrent state. It
returns a :class:`DeltaMemoryOutput` so experiments can collect auxiliary traces
without changing the core benchmark loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


@dataclass
class DeltaMemoryOutput:
    """Standard output container for recurrent memory variants.

    Attributes:
        outputs: Tensor of shape ``(batch, seq_len, value_dim)``.
        final_state: Tensor of shape ``(batch, head_dim, value_dim)``.
        trace: Optional auxiliary tensors or metrics captured during the pass.
    """

    outputs: torch.Tensor
    final_state: torch.Tensor
    trace: dict[str, Any] = field(default_factory=dict)


class DeltaMemoryVariant(nn.Module, ABC):
    """Common API for associative recurrent-memory mechanisms."""

    def __init__(self, head_dim: int, value_dim: int) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        self.head_dim = head_dim
        self.value_dim = value_dim

    def reset(self) -> None:
        """Reset optional internal evidence buffers.

        Most variants are stateless between calls and use ``initial_state`` for
        continuation. Variants with internal diagnostics may override this hook.
        """

    @abstractmethod
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: Optional[Mapping[str, torch.Tensor]] = None,
        initial_state: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> DeltaMemoryOutput:
        """Run the recurrent memory over a sequence.

        Args:
            queries: Tensor ``(batch, seq_len, head_dim)``.
            keys: Tensor ``(batch, seq_len, head_dim)``.
            values: Tensor ``(batch, seq_len, value_dim)``.
            gates: Mapping of variant-specific gate tensors. KDA expects
                ``alpha`` with shape ``(batch, seq_len, head_dim)`` and
                ``beta`` with shape ``(batch, seq_len, 1)``.
            initial_state: Optional tensor ``(batch, head_dim, value_dim)``.
            return_trace: Whether to populate :attr:`DeltaMemoryOutput.trace`.

        Returns:
            A standardized :class:`DeltaMemoryOutput`.
        """

    def _validate_inputs(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Validate common input dimensions."""
        if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
            raise ValueError("queries, keys, and values must be rank-3 tensors")
        if queries.shape != keys.shape:
            raise ValueError(
                f"queries and keys must have identical shapes, got "
                f"{queries.shape} and {keys.shape}"
            )
        if queries.shape[:2] != values.shape[:2]:
            raise ValueError(
                f"queries/keys and values must agree on batch and sequence dims, "
                f"got {queries.shape[:2]} and {values.shape[:2]}"
            )
        if queries.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected query/key dim {self.head_dim}, got {queries.shape[-1]}"
            )
        if values.shape[-1] != self.value_dim:
            raise ValueError(
                f"expected value dim {self.value_dim}, got {values.shape[-1]}"
            )


_VARIANT_REGISTRY: dict[str, type[DeltaMemoryVariant]] = {}


class VariantNotFoundError(KeyError):
    """Raised when an unknown delta-memory variant is requested."""


def register_variant(name: str):
    """Class decorator used to register a delta-memory variant."""

    if not name or not name.strip():
        raise ValueError("variant name must be a non-empty string")
    normalized = name.strip().lower()

    def decorator(cls: type[DeltaMemoryVariant]) -> type[DeltaMemoryVariant]:
        if not issubclass(cls, DeltaMemoryVariant):
            raise TypeError("registered variants must subclass DeltaMemoryVariant")
        if normalized in _VARIANT_REGISTRY and _VARIANT_REGISTRY[normalized] is not cls:
            raise ValueError(f"variant '{normalized}' is already registered")
        _VARIANT_REGISTRY[normalized] = cls
        return cls

    return decorator


def available_variants() -> tuple[str, ...]:
    """Return registered variant names in deterministic order."""

    return tuple(sorted(_VARIANT_REGISTRY))


def create_variant(
    name: str,
    head_dim: int,
    value_dim: int,
    **kwargs: Any,
) -> DeltaMemoryVariant:
    """Instantiate a registered delta-memory variant."""

    normalized = name.strip().lower()
    try:
        variant_cls = _VARIANT_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(available_variants()) or "<none>"
        raise VariantNotFoundError(
            f"unknown delta-memory variant '{name}'. Available: {available}"
        ) from exc
    return variant_cls(head_dim=head_dim, value_dim=value_dim, **kwargs)
