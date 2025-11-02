"""
Integration tests for KDA attention layer.

Note: All tests use synthetic/random data for validation purposes.
This is standard practice in ML research implementations to verify
correctness without requiring real datasets.
"""

import torch
import pytest
from kda import KimiDeltaAttention


class TestKimiDeltaAttention:
    """Test suite for KimiDeltaAttention layer."""
    
    @pytest.fixture
    def setup_params(self):
        """Setup common test parameters."""
        return {
            'batch_size': 2,
            'seq_len': 32,
            'd_model': 256,
            'num_heads': 4
        }
    
    @pytest.fixture
    def create_synthetic_input(self, setup_params):
        """Create synthetic input data."""
        return torch.randn(
            setup_params['batch_size'],
            setup_params['seq_len'],
            setup_params['d_model']
        )
    
    def test_basic_forward(self, setup_params, create_synthetic_input):
        """Test basic forward pass."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads']
        )
        
        output = kda_attn(x)
        
        assert output.shape == x.shape
    
    def test_with_states(self, setup_params, create_synthetic_input):
        """Test forward pass with state management."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads']
        )
        
        output, final_states = kda_attn(x, return_states=True)
        
        assert output.shape == x.shape
        assert final_states.shape == (
            setup_params['batch_size'],
            setup_params['num_heads'],
            kda_attn.head_dim,
            kda_attn.value_dim
        )
    
    def test_with_short_conv(self, setup_params, create_synthetic_input):
        """Test with short convolution enabled."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads'],
            use_short_conv=True,
            conv_kernel_size=4
        )
        
        output = kda_attn(x)
        assert output.shape == x.shape
    
    def test_without_short_conv(self, setup_params, create_synthetic_input):
        """Test without short convolution."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads'],
            use_short_conv=False
        )
        
        output = kda_attn(x)
        assert output.shape == x.shape
    
    def test_chunkwise_mode(self, setup_params, create_synthetic_input):
        """Test chunkwise implementation."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads'],
            use_chunkwise=True,
            chunk_size=16
        )
        
        output = kda_attn(x)
        assert output.shape == x.shape
    
    def test_gradient_flow(self, setup_params, create_synthetic_input):
        """Test that gradients flow through the layer."""
        x = create_synthetic_input
        x.requires_grad = True
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads']
        )
        
        output = kda_attn(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        assert x.grad is not None
        assert kda_attn.q_proj.weight.grad is not None
        assert kda_attn.k_proj.weight.grad is not None
        assert kda_attn.v_proj.weight.grad is not None
        assert kda_attn.out_proj.weight.grad is not None
    
    def test_with_dropout(self, setup_params, create_synthetic_input):
        """Test with dropout enabled."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads'],
            dropout=0.1
        )
        
        # Training mode
        kda_attn.train()
        output_train = kda_attn(x)
        
        # Eval mode
        kda_attn.eval()
        output_eval = kda_attn(x)
        
        assert output_train.shape == x.shape
        assert output_eval.shape == x.shape
    
    def test_different_head_dims(self, setup_params, create_synthetic_input):
        """Test with custom head dimension."""
        x = create_synthetic_input
        
        kda_attn = KimiDeltaAttention(
            d_model=setup_params['d_model'],
            num_heads=setup_params['num_heads'],
            head_dim=32  # Custom head dimension
        )
        
        output = kda_attn(x)
        assert output.shape == x.shape
    
    def test_state_continuity(self, setup_params):
        """Test that states can be passed between forward calls."""
        d_model = setup_params['d_model']
        num_heads = setup_params['num_heads']
        
        kda_attn = KimiDeltaAttention(d_model=d_model, num_heads=num_heads)
        
        # First sequence
        x1 = torch.randn(1, 16, d_model)
        output1, state1 = kda_attn(x1, return_states=True)
        
        # Second sequence with state from first
        x2 = torch.randn(1, 16, d_model)
        output2, state2 = kda_attn(x2, initial_states=state1, return_states=True)
        
        assert output1.shape == (1, 16, d_model)
        assert output2.shape == (1, 16, d_model)
        assert state2.shape == state1.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
