"""
Unit tests for KDA core implementation.

Note: All tests use synthetic/random data for validation purposes.
This is standard practice in ML research implementations to verify
correctness without requiring real datasets.
"""

import torch
import torch.nn.functional as F
import pytest
from kda.core import KDACore, KDAChunkwise


class TestKDACore:
    """Test suite for KDACore implementation."""
    
    @pytest.fixture
    def setup_params(self):
        """Setup common test parameters."""
        return {
            'batch_size': 2,
            'seq_len': 16,
            'head_dim': 32,
            'value_dim': 32
        }
    
    @pytest.fixture
    def create_synthetic_data(self, setup_params):
        """Create synthetic test data."""
        batch_size = setup_params['batch_size']
        seq_len = setup_params['seq_len']
        head_dim = setup_params['head_dim']
        value_dim = setup_params['value_dim']
        
        # L2-normalized queries and keys
        queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        values = torch.randn(batch_size, seq_len, value_dim)
        
        # Gating parameters in [0, 1]
        alpha = torch.sigmoid(torch.randn(batch_size, seq_len, head_dim))
        beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1))
        
        return queries, keys, values, alpha, beta
    
    def test_output_shape(self, setup_params, create_synthetic_data):
        """Test that output shape is correct."""
        queries, keys, values, alpha, beta = create_synthetic_data
        
        kda = KDACore(setup_params['head_dim'], setup_params['value_dim'])
        outputs, final_state = kda(queries, keys, values, alpha, beta)
        
        expected_output_shape = (setup_params['batch_size'], 
                                setup_params['seq_len'], 
                                setup_params['value_dim'])
        expected_state_shape = (setup_params['batch_size'],
                               setup_params['head_dim'],
                               setup_params['value_dim'])
        
        assert outputs.shape == expected_output_shape
        assert final_state.shape == expected_state_shape
    
    def test_with_initial_state(self, setup_params, create_synthetic_data):
        """Test KDA with initial state."""
        queries, keys, values, alpha, beta = create_synthetic_data
        
        # Create initial state
        initial_state = torch.randn(
            setup_params['batch_size'],
            setup_params['head_dim'],
            setup_params['value_dim']
        )
        
        kda = KDACore(setup_params['head_dim'], setup_params['value_dim'])
        outputs, final_state = kda(queries, keys, values, alpha, beta, initial_state)
        
        assert outputs.shape[0] == setup_params['batch_size']
        assert final_state.shape == initial_state.shape
    
    def test_gradient_flow(self, setup_params, create_synthetic_data):
        """Test that gradients flow through the module."""
        queries, keys, values, alpha, beta = create_synthetic_data
        
        # Require gradients
        queries.requires_grad = True
        keys.requires_grad = True
        values.requires_grad = True
        
        kda = KDACore(setup_params['head_dim'], setup_params['value_dim'])
        outputs, _ = kda(queries, keys, values, alpha, beta)
        
        # Compute loss and backward
        loss = outputs.sum()
        loss.backward()
        
        assert queries.grad is not None
        assert keys.grad is not None
        assert values.grad is not None
    
    def test_deterministic(self, setup_params, create_synthetic_data):
        """Test that forward pass is deterministic."""
        queries, keys, values, alpha, beta = create_synthetic_data
        
        kda = KDACore(setup_params['head_dim'], setup_params['value_dim'])
        
        # Run twice with same inputs
        outputs1, state1 = kda(queries, keys, values, alpha, beta)
        outputs2, state2 = kda(queries, keys, values, alpha, beta)
        
        assert torch.allclose(outputs1, outputs2)
        assert torch.allclose(state1, state2)


class TestKDAChunkwise:
    """Test suite for KDAChunkwise implementation."""
    
    @pytest.fixture
    def setup_params(self):
        """Setup common test parameters."""
        return {
            'batch_size': 2,
            'seq_len': 64,  # Multiple of chunk_size
            'head_dim': 32,
            'value_dim': 32,
            'chunk_size': 16
        }
    
    @pytest.fixture
    def create_synthetic_data(self, setup_params):
        """Create synthetic test data."""
        batch_size = setup_params['batch_size']
        seq_len = setup_params['seq_len']
        head_dim = setup_params['head_dim']
        value_dim = setup_params['value_dim']
        
        queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        values = torch.randn(batch_size, seq_len, value_dim)
        alpha = torch.sigmoid(torch.randn(batch_size, seq_len, head_dim))
        beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1))
        
        return queries, keys, values, alpha, beta
    
    def test_chunkwise_output_shape(self, setup_params, create_synthetic_data):
        """Test that chunkwise output shape is correct."""
        queries, keys, values, alpha, beta = create_synthetic_data
        
        kda_chunk = KDAChunkwise(
            setup_params['head_dim'],
            setup_params['value_dim'],
            setup_params['chunk_size']
        )
        
        outputs, final_state = kda_chunk(queries, keys, values, alpha, beta)
        
        expected_output_shape = (setup_params['batch_size'],
                                setup_params['seq_len'],
                                setup_params['value_dim'])
        
        assert outputs.shape == expected_output_shape
    
    def test_chunkwise_with_padding(self, setup_params):
        """Test chunkwise with sequence length that requires padding."""
        batch_size = setup_params['batch_size']
        seq_len = 50  # Not a multiple of chunk_size (16)
        head_dim = setup_params['head_dim']
        value_dim = setup_params['value_dim']
        
        queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        values = torch.randn(batch_size, seq_len, value_dim)
        alpha = torch.sigmoid(torch.randn(batch_size, seq_len, head_dim))
        beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1))
        
        kda_chunk = KDAChunkwise(head_dim, value_dim, setup_params['chunk_size'])
        outputs, _ = kda_chunk(queries, keys, values, alpha, beta)
        
        # Output should have original sequence length (padding removed)
        assert outputs.shape == (batch_size, seq_len, value_dim)

    @pytest.mark.parametrize("seq_len", [17, 50, 100, 129])
    def test_padded_run_matches_sequential_state(self, setup_params, seq_len):
        """Padded chunkwise runs return the same state as the exact recurrence.

        Chunked execution pads the sequence up to a whole number of chunks.
        Those padded steps must be no-ops of the recurrence: padding the
        decay gate with 0.0 (rather than 1.0) would decay the state to zero,
        which silently corrupts the returned state even though the output
        tensors still look correct.
        """
        head_dim = setup_params['head_dim']
        value_dim = setup_params['value_dim']
        batch_size = setup_params['batch_size']

        queries = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        keys = F.normalize(torch.randn(batch_size, seq_len, head_dim), dim=-1)
        values = torch.randn(batch_size, seq_len, value_dim)
        alpha = torch.sigmoid(torch.randn(batch_size, seq_len, head_dim))
        beta = torch.sigmoid(torch.randn(batch_size, seq_len, 1))

        sequential_out, sequential_state = KDACore(head_dim, value_dim)(
            queries, keys, values, alpha, beta
        )
        chunked_out, chunked_state = KDAChunkwise(
            head_dim, value_dim, setup_params['chunk_size']
        )(queries, keys, values, alpha, beta)

        assert torch.allclose(chunked_out, sequential_out, atol=1e-5), (
            f"max output deviation: {(chunked_out - sequential_out).abs().max().item():.2e}"
        )
        assert chunked_state.abs().max() > 1e-6, (
            "chunkwise state collapsed to zero; padded steps are not no-ops"
        )
        assert torch.allclose(chunked_state, sequential_state, atol=1e-5), (
            f"max state deviation: {(chunked_state - sequential_state).abs().max().item():.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
