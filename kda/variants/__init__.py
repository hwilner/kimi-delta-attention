"""Research variants sharing the delta-memory benchmark interface."""

from .base import (
    DeltaMemoryOutput,
    DeltaMemoryVariant,
    VariantNotFoundError,
    available_variants,
    create_variant,
    register_variant,
)
from .erase_then_delta import EraseThenDeltaVariant
from .gated_deltanet2 import GatedDeltaNet2Variant
from .kda_adapter import KDAVariant
from .kda_rls_lite import KDARLSLiteVariant
from .precision_gated_kda import PrecisionGatedKDAVariant

__all__ = [
    "DeltaMemoryOutput",
    "DeltaMemoryVariant",
    "VariantNotFoundError",
    "available_variants",
    "create_variant",
    "register_variant",
    "EraseThenDeltaVariant",
    "GatedDeltaNet2Variant",
    "KDAVariant",
    "KDARLSLiteVariant",
    "PrecisionGatedKDAVariant",
]
