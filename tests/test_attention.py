"""
Integration tests for KDA attention layer.

Note: All tests use synthetic/random data for validation purposes.
This is standard practice in ML research implementations to verify
correctness without requiring real datasets.
"""

import torch
import pytest
from kda import KimiDeltaAttention
from kda.attention import ShortConv1d


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

    def test_chunkwise_state_not_corrupted_by_padding(self, setup_params):
        """Chunkwise mode must not zero the state when the length pads.

        A length that is not a multiple of the chunk size forces the
        chunkwise path to pad. The returned state is what a streaming caller
        feeds into the next call, so a collapsed state would silently destroy
        all memory carried across calls.
        """
        batch_size = setup_params['batch_size']
        d_model = setup_params['d_model']
        num_heads = setup_params['num_heads']
        chunk_size = 32
        seq_len = 50  # Pads to two 32-wide chunks.

        torch.manual_seed(0)
        x = torch.randn(batch_size, seq_len, d_model)

        chunkwise = KimiDeltaAttention(
            d_model=d_model, num_heads=num_heads, use_short_conv=False,
            use_chunkwise=True, chunk_size=chunk_size,
        ).eval()
        sequential = KimiDeltaAttention(
            d_model=d_model, num_heads=num_heads, use_short_conv=False,
            use_chunkwise=False, chunk_size=chunk_size,
        ).eval()
        sequential.load_state_dict(chunkwise.state_dict())

        with torch.no_grad():
            _, chunk_state = chunkwise(x, return_states=True)
            _, seq_state = sequential(x, return_states=True)

        assert chunk_state.shape == seq_state.shape
        assert chunk_state.abs().max() > 1e-6, (
            "chunkwise returned an all-zero state; padded steps are not no-ops"
        )
        assert torch.allclose(chunk_state, seq_state, atol=1e-4), (
            f"max state deviation: {(chunk_state - seq_state).abs().max().item():.2e}"
        )


class TestShortConv:
    """Test suite for the short convolution used by the attention layer."""

    @pytest.mark.parametrize("kernel_size", [1, 2, 3, 4])
    def test_output_length_preserved(self, kernel_size):
        """The convolution must not change the sequence length.

        A width-1 kernel adds no causal padding, and naively trimming the
        padding with a ``-0`` slice would drop the entire sequence to zero
        length instead of leaving it untouched.
        """
        conv = ShortConv1d(d_model=16, kernel_size=kernel_size)
        x = torch.randn(2, 12, 16)
        assert conv(x).shape == x.shape

    def test_kernel_size_one_end_to_end(self):
        """A width-1 short convolution is usable in the full attention layer."""
        layer = KimiDeltaAttention(
            d_model=32, num_heads=2, use_short_conv=True, conv_kernel_size=1
        )
        x = torch.randn(2, 10, 32)
        assert layer(x).shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
