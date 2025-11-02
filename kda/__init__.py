"""
Kimi Delta Attention (KDA) Implementation

An educational implementation of the Kimi Delta Attention mechanism from:
"Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)

This package provides:
- KDACore: Core recurrent state update mechanism
- KDAChunkwise: Chunkwise parallel implementation
- KimiDeltaAttention: Complete multi-head attention layer with neural parameterization
"""

from .core import KDACore, KDAChunkwise
from .attention import KimiDeltaAttention, ShortConv1d

__version__ = "0.1.0"

__all__ = [
    "KDACore",
    "KDAChunkwise",
    "KimiDeltaAttention",
    "ShortConv1d",
]
