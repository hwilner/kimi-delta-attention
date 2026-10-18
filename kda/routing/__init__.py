"""Adaptive hybrid routing between compressed memory and exact attention (#38)."""

from .hybrid import (
    ROUTING_SIGNALS,
    HybridRoutingResult,
    RoutingPolicy,
    compute_routing_signals,
    exact_attention_readout,
    fixed_ratio_decisions,
    route_hybrid_readout,
)

__all__ = [
    "ROUTING_SIGNALS",
    "HybridRoutingResult",
    "RoutingPolicy",
    "compute_routing_signals",
    "exact_attention_readout",
    "fixed_ratio_decisions",
    "route_hybrid_readout",
]
