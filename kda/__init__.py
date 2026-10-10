"""
Kimi Delta Attention (KDA) Implementation

An educational implementation of the Kimi Delta Attention mechanism from:
"Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)

This package provides:
- KDACore: Core recurrent state update mechanism
- KDAChunkwise: Chunkwise parallel implementation
- KDADPLRChunkwise: Exact chunkwise DPLR (WY-representation) parallel form
- KimiDeltaAttention: Complete multi-head attention layer with neural parameterization
- BlockScaledFP8State / stochastic_round: low-precision state utilities
"""

from .core import KDACore, KDAChunkwise
from .chunkwise import KDADPLRChunkwise, chunkwise_dplr_forward
from .precision import BlockScaledFP8State, stochastic_round
from .attention import KimiDeltaAttention, ShortConv1d
from .variants import (
    DeltaMemoryOutput,
    DeltaMemoryVariant,
    available_variants,
    create_variant,
)

__version__ = "0.2.0"

__all__ = [
    "KDACore",
    "KDAChunkwise",
    "KDADPLRChunkwise",
    "chunkwise_dplr_forward",
    "BlockScaledFP8State",
    "stochastic_round",
    "KimiDeltaAttention",
    "ShortConv1d",
    "DeltaMemoryOutput",
    "DeltaMemoryVariant",
    "available_variants",
    "create_variant",
]
