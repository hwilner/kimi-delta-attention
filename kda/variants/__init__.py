"""Research variants sharing the delta-memory benchmark interface."""

from .base import (
    DeltaMemoryOutput,
    DeltaMemoryVariant,
    VariantNotFoundError,
    available_variants,
    create_variant,
    register_variant,
)
from .kda_adapter import KDAVariant
from .kda_rls_lite import KDARLSLiteVariant

__all__ = [
    "DeltaMemoryOutput",
    "DeltaMemoryVariant",
    "VariantNotFoundError",
    "available_variants",
    "create_variant",
    "register_variant",
    "KDAVariant",
    "KDARLSLiteVariant",
]
