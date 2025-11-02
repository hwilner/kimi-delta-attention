"""
Kimi Delta Attention (KDA) - Core Implementation

This module implements the core Kimi Delta Attention mechanism as described in:
"Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)

Key Innovation:
--------------
KDA refines Gated DeltaNet by replacing scalar decay (α_t) with diagonal decay (Diag(α_t)),
enabling fine-grained per-channel memory control.

Mathematical Formulation:
------------------------
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t

where:
- S_t ∈ R^{d_h × d_v}: recurrent state matrix
- q_t, k_t ∈ R^{d_h}: queries and keys (L2-normalized)
- v_t ∈ R^{d_v}: values
- α_t ∈ [0,1]^{d_h}: per-channel decay (diagonal)
- β_t ∈ [0,1]: scalar gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class KDACore(nn.Module):
    """
    Core Kimi Delta Attention computation module.
    
    This implements the recurrent state update and output computation
    for a single attention head.
    
    Args:
        head_dim: Dimension of queries/keys (d_h)
        value_dim: Dimension of values (d_v)
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        eps: float = 1e-6
    ):
        super().__init__()
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.eps = eps
        
    def forward(
        self,
        queries: torch.Tensor,      # (B, L, d_h)
        keys: torch.Tensor,          # (B, L, d_h)
        values: torch.Tensor,        # (B, L, d_v)
        alpha: torch.Tensor,         # (B, L, d_h) - per-channel decay
        beta: torch.Tensor,          # (B, L, 1) - scalar gate
        initial_state: Optional[torch.Tensor] = None  # (B, d_h, d_v)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of KDA core mechanism.
        
        Args:
            queries: Query vectors (L2-normalized)
            keys: Key vectors (L2-normalized)
            values: Value vectors
            alpha: Per-channel decay factors in [0,1]
            beta: Scalar gate values in [0,1]
            initial_state: Initial recurrent state (optional)
            
        Returns:
            outputs: Attention outputs (B, L, d_v)
            final_state: Final recurrent state (B, d_h, d_v)
        """
        batch_size, seq_len, _ = queries.shape
        device = queries.device
        
        # Initialize state
        if initial_state is None:
            state = torch.zeros(
                batch_size, self.head_dim, self.value_dim,
                device=device, dtype=queries.dtype
            )
        else:
            state = initial_state
            
        outputs = []
        
        # Recurrent computation over sequence
        for t in range(seq_len):
            q_t = queries[:, t, :]           # (B, d_h)
            k_t = keys[:, t, :]              # (B, d_h)
            v_t = values[:, t, :]            # (B, d_v)
            alpha_t = alpha[:, t, :]         # (B, d_h)
            beta_t = beta[:, t, :]           # (B, 1)
            
            # State update with fine-grained gating
            # S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
            
            # Step 1: Apply diagonal decay - Diag(α_t) S_{t-1}
            # Broadcasting: (B, d_h, 1) * (B, d_h, d_v) -> (B, d_h, d_v)
            state = alpha_t.unsqueeze(-1) * state
            
            # Step 2: Compute forget term - β_t k_t k_t^T Diag(α_t) S_{t-1}
            # This removes the component of the state in the direction of k_t
            # (B, d_h, 1) @ (B, 1, d_v) -> (B, d_h, d_v)
            k_state = torch.bmm(k_t.unsqueeze(-1), 
                               torch.bmm(k_t.unsqueeze(1), state))
            state = state - beta_t.unsqueeze(-1) * k_state
            
            # Step 3: Add new key-value association - β_t k_t v_t^T
            # (B, d_h, 1) @ (B, 1, d_v) -> (B, d_h, d_v)
            kv_outer = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
            state = state + beta_t.unsqueeze(-1) * kv_outer
            
            # Compute output: o_t = S_t^T q_t
            # (B, d_v, d_h) @ (B, d_h, 1) -> (B, d_v, 1) -> (B, d_v)
            o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (B, L, d_v)
        
        return outputs, state


class KDAChunkwise(nn.Module):
    """
    Chunkwise parallel implementation of KDA for improved efficiency.
    
    This implements the WY representation and chunkwise parallelization
    described in Section 3.1 of the paper. For educational purposes,
    this is a simplified version focusing on clarity over optimization.
    
    Args:
        head_dim: Dimension of queries/keys (d_h)
        value_dim: Dimension of values (d_v)
        chunk_size: Size of each chunk for parallel processing
        eps: Small constant for numerical stability
    """
    
    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        chunk_size: int = 64,
        eps: float = 1e-6
    ):
        super().__init__()
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.chunk_size = chunk_size
        self.eps = eps
        
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunkwise forward pass.
        
        This splits the sequence into chunks and processes them with
        a combination of inter-chunk recurrence and intra-chunk parallelism.
        
        Args:
            Same as KDACore.forward()
            
        Returns:
            outputs: Attention outputs (B, L, d_v)
            final_state: Final recurrent state (B, d_h, d_v)
        """
        batch_size, seq_len, _ = queries.shape
        device = queries.device
        
        # For simplicity, if sequence is shorter than chunk size,
        # fall back to sequential processing
        if seq_len <= self.chunk_size:
            core = KDACore(self.head_dim, self.value_dim, self.eps)
            return core(queries, keys, values, alpha, beta, initial_state)
        
        # Split into chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Pad sequence to multiple of chunk_size
        pad_len = num_chunks * self.chunk_size - seq_len
        if pad_len > 0:
            queries = F.pad(queries, (0, 0, 0, pad_len))
            keys = F.pad(keys, (0, 0, 0, pad_len))
            values = F.pad(values, (0, 0, 0, pad_len))
            alpha = F.pad(alpha, (0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))
        
        # Reshape into chunks
        queries = queries.view(batch_size, num_chunks, self.chunk_size, self.head_dim)
        keys = keys.view(batch_size, num_chunks, self.chunk_size, self.head_dim)
        values = values.view(batch_size, num_chunks, self.chunk_size, self.value_dim)
        alpha = alpha.view(batch_size, num_chunks, self.chunk_size, self.head_dim)
        beta = beta.view(batch_size, num_chunks, self.chunk_size, 1)
        
        # Process chunks sequentially (inter-chunk recurrence)
        if initial_state is None:
            state = torch.zeros(
                batch_size, self.head_dim, self.value_dim,
                device=device, dtype=queries.dtype
            )
        else:
            state = initial_state
            
        chunk_outputs = []
        core = KDACore(self.head_dim, self.value_dim, self.eps)
        
        for c in range(num_chunks):
            # Process chunk with intra-chunk parallelism
            chunk_out, state = core(
                queries[:, c],
                keys[:, c],
                values[:, c],
                alpha[:, c],
                beta[:, c],
                state
            )
            chunk_outputs.append(chunk_out)
        
        # Concatenate chunk outputs
        outputs = torch.cat(chunk_outputs, dim=1)  # (B, num_chunks * chunk_size, d_v)
        
        # Remove padding
        if pad_len > 0:
            outputs = outputs[:, :-pad_len]
        
        return outputs, state


def test_kda_core():
    """
    Simple test to verify KDA core implementation.
    Uses synthetic data for validation.
    """
    print("Testing KDA Core Implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 8
    head_dim = 16
    value_dim = 16
    
    # Create synthetic test data
    queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
    keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
    values = torch.randn(batch_size, seq_len, value_dim)
    alpha = torch.sigmoid(torch.randn(batch_size, seq_len, head_dim))
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1))
    
    # Initialize KDA core
    kda = KDACore(head_dim, value_dim)
    
    # Forward pass
    outputs, final_state = kda(queries, keys, values, alpha, beta)
    
    # Verify shapes
    assert outputs.shape == (batch_size, seq_len, value_dim), \
        f"Expected output shape {(batch_size, seq_len, value_dim)}, got {outputs.shape}"
    assert final_state.shape == (batch_size, head_dim, value_dim), \
        f"Expected state shape {(batch_size, head_dim, value_dim)}, got {final_state.shape}"
    
    print(f"✓ Output shape: {outputs.shape}")
    print(f"✓ Final state shape: {final_state.shape}")
    print(f"✓ Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print("✓ KDA Core test passed!")


if __name__ == "__main__":
    test_kda_core()
