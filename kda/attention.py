"""
Kimi Delta Attention Layer

This module implements the complete KDA attention layer with neural parameterization
as described in Section 4 of the paper.

Neural Parameterization:
-----------------------
q^h, k^h = L2Norm(Swish(ShortConv(W_q^h x_t)))
v^h = Swish(ShortConv(W_v^h x_t))
α_t^h = f(W_α^+ W_α^↓ x_t) ∈ [0,1]^{d_h}  # Low-rank projection
β_t^h = Sigmoid(W_β^h x_t) ∈ [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .core import KDACore, KDAChunkwise


class ShortConv1d(nn.Module):
    """
    Short convolution for local context modeling.
    
    Applies a 1D convolution with a small kernel size (typically 4)
    to capture local dependencies before attention.
    
    Args:
        d_model: Input dimension
        kernel_size: Convolution kernel size (default: 4)
    """
    
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model  # Depthwise convolution
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, D)
        Returns:
            Convolved tensor (B, L, D)
        """
        # Conv1d expects (B, D, L)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Remove extra padding
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.transpose(1, 2)
        return x


class KimiDeltaAttention(nn.Module):
    """
    Complete Kimi Delta Attention layer with multi-head support.
    
    This implements the full KDA mechanism including:
    - Neural parameterization of queries, keys, values
    - Fine-grained diagonal gating (α)
    - Scalar gating (β)
    - Multi-head attention
    - Optional short convolution for local context
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (default: d_model // num_heads)
        use_short_conv: Whether to use short convolution (default: True)
        conv_kernel_size: Kernel size for short convolution (default: 4)
        chunk_size: Chunk size for parallel processing (default: 64)
        use_chunkwise: Whether to use chunkwise implementation (default: False)
        dropout: Dropout probability (default: 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
        chunk_size: int = 64,
        use_chunkwise: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim or (d_model // num_heads)
        self.value_dim = self.head_dim  # For simplicity, use same dimension
        self.use_short_conv = use_short_conv
        self.use_chunkwise = use_chunkwise
        
        # Short convolution for local context
        if use_short_conv:
            self.short_conv = ShortConv1d(d_model, conv_kernel_size)
        
        # Projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * self.value_dim, bias=False)
        
        # Gating parameters
        # α: per-channel decay (low-rank parameterization)
        alpha_rank = self.head_dim  # Rank equal to head dimension
        self.alpha_down = nn.Linear(d_model, alpha_rank, bias=False)
        self.alpha_up = nn.Linear(alpha_rank, num_heads * self.head_dim, bias=False)
        
        # β: scalar gate
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.value_dim, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # KDA core
        if use_chunkwise:
            self.kda_cores = nn.ModuleList([
                KDAChunkwise(self.head_dim, self.value_dim, chunk_size)
                for _ in range(num_heads)
            ])
        else:
            self.kda_cores = nn.ModuleList([
                KDACore(self.head_dim, self.value_dim)
                for _ in range(num_heads)
            ])
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters following the paper's recommendations."""
        # Xavier uniform initialization for projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize gating parameters
        nn.init.xavier_uniform_(self.alpha_down.weight)
        nn.init.xavier_uniform_(self.alpha_up.weight)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        
    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
        return_states: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of KDA attention.
        
        Args:
            x: Input tensor (B, L, D)
            initial_states: Initial recurrent states for each head (B, H, d_h, d_v)
            return_states: Whether to return final states
            
        Returns:
            Output tensor (B, L, D)
            If return_states=True, also returns final states (B, H, d_h, d_v)
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply short convolution for local context
        if self.use_short_conv:
            x_conv = self.short_conv(x)
        else:
            x_conv = x
        
        # Project to queries, keys, values
        # (B, L, D) -> (B, L, H * d_h)
        queries = self.q_proj(x_conv)
        keys = self.k_proj(x_conv)
        values = self.v_proj(x_conv)
        
        # Apply Swish activation
        queries = F.silu(queries)
        keys = F.silu(keys)
        values = F.silu(values)
        
        # L2 normalization for queries and keys (for stability)
        queries = F.normalize(queries, p=2, dim=-1)
        keys = F.normalize(keys, p=2, dim=-1)
        
        # Reshape to separate heads
        # (B, L, H * d_h) -> (B, L, H, d_h) -> (B, H, L, d_h)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.value_dim)
        
        # Compute gating parameters
        # α: per-channel decay with low-rank parameterization
        alpha = self.alpha_down(x)  # (B, L, rank)
        alpha = self.alpha_up(alpha)  # (B, L, H * d_h)
        alpha = torch.sigmoid(alpha)  # Constrain to [0, 1]
        alpha = alpha.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # β: scalar gate
        beta = self.beta_proj(x)  # (B, L, H)
        beta = torch.sigmoid(beta)  # Constrain to [0, 1]
        beta = beta.view(batch_size, seq_len, self.num_heads, 1)
        
        # Process each head independently
        head_outputs = []
        final_states = []
        
        for h in range(self.num_heads):
            # Extract head-specific tensors
            q_h = queries[:, :, h, :]  # (B, L, d_h)
            k_h = keys[:, :, h, :]     # (B, L, d_h)
            v_h = values[:, :, h, :]   # (B, L, d_v)
            alpha_h = alpha[:, :, h, :]  # (B, L, d_h)
            beta_h = beta[:, :, h, :]    # (B, L, 1)
            
            # Initial state for this head
            init_state_h = None
            if initial_states is not None:
                init_state_h = initial_states[:, h, :, :]
            
            # Apply KDA core
            out_h, state_h = self.kda_cores[h](
                q_h, k_h, v_h, alpha_h, beta_h, init_state_h
            )
            
            head_outputs.append(out_h)
            final_states.append(state_h)
        
        # Concatenate head outputs
        # List of (B, L, d_v) -> (B, L, H, d_v) -> (B, L, H * d_v)
        outputs = torch.stack(head_outputs, dim=2)
        outputs = outputs.view(batch_size, seq_len, self.num_heads * self.value_dim)
        
        # Output projection
        outputs = self.out_proj(outputs)
        outputs = self.dropout(outputs)
        
        if return_states:
            # Stack final states: List of (B, d_h, d_v) -> (B, H, d_h, d_v)
            final_states = torch.stack(final_states, dim=1)
            return outputs, final_states
        
        return outputs


def test_kda_attention():
    """
    Test the complete KDA attention layer.
    Uses synthetic data for validation.
    """
    print("Testing KDA Attention Layer...")
    
    # Test parameters
    batch_size = 2
    seq_len = 32
    d_model = 256
    num_heads = 4
    
    # Create synthetic test data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize KDA attention
    kda_attn = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_short_conv=True,
        use_chunkwise=False
    )
    
    # Forward pass without state
    output = kda_attn(x)
    assert output.shape == x.shape, \
        f"Expected output shape {x.shape}, got {output.shape}"
    print(f"✓ Output shape (no state): {output.shape}")
    
    # Forward pass with state
    output, final_states = kda_attn(x, return_states=True)
    assert output.shape == x.shape
    assert final_states.shape == (batch_size, num_heads, 
                                   kda_attn.head_dim, kda_attn.value_dim)
    print(f"✓ Output shape (with state): {output.shape}")
    print(f"✓ Final states shape: {final_states.shape}")
    
    # Test with chunkwise implementation
    kda_attn_chunk = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_short_conv=True,
        use_chunkwise=True,
        chunk_size=16
    )
    
    output_chunk = kda_attn_chunk(x)
    assert output_chunk.shape == x.shape
    print(f"✓ Chunkwise output shape: {output_chunk.shape}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    assert kda_attn.q_proj.weight.grad is not None
    print("✓ Gradient flow verified")
    
    print(f"✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
    print("✓ KDA Attention test passed!")


if __name__ == "__main__":
    test_kda_attention()
