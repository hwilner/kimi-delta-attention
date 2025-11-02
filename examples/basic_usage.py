"""
Basic Usage Example for Kimi Delta Attention

This example demonstrates how to use the KDA layer in a simple setting.

Note: This example uses synthetic data for demonstration purposes.
"""

import torch
from kda import KimiDeltaAttention


def basic_example():
    """Basic usage of KDA attention layer."""
    print("=" * 60)
    print("Basic KDA Usage Example")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 128
    d_model = 512
    num_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    
    # Create KDA layer
    kda_layer = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_short_conv=True,
        dropout=0.1
    )
    
    print(f"\nKDA Layer created with:")
    print(f"  Head dimension: {kda_layer.head_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in kda_layer.parameters()):,}")
    
    # Create synthetic input (in practice, this would be your embeddings)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = kda_layer(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n✓ Basic forward pass successful!")


def stateful_example():
    """Example with state management for sequential processing."""
    print("\n" + "=" * 60)
    print("Stateful Processing Example")
    print("=" * 60)
    
    # Configuration
    batch_size = 1
    chunk_len = 64
    d_model = 256
    num_heads = 4
    
    print(f"\nProcessing sequence in chunks of {chunk_len} tokens")
    
    # Create KDA layer
    kda_layer = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads
    )
    
    # Process first chunk
    chunk1 = torch.randn(batch_size, chunk_len, d_model)
    output1, state1 = kda_layer(chunk1, return_states=True)
    print(f"\nChunk 1:")
    print(f"  Input shape: {chunk1.shape}")
    print(f"  Output shape: {output1.shape}")
    print(f"  State shape: {state1.shape}")
    
    # Process second chunk with state from first chunk
    chunk2 = torch.randn(batch_size, chunk_len, d_model)
    output2, state2 = kda_layer(chunk2, initial_states=state1, return_states=True)
    print(f"\nChunk 2 (with state from chunk 1):")
    print(f"  Input shape: {chunk2.shape}")
    print(f"  Output shape: {output2.shape}")
    print(f"  State shape: {state2.shape}")
    
    print("\n✓ Stateful processing successful!")
    print("  States allow the model to maintain context across chunks")


def chunkwise_example():
    """Example using chunkwise parallel processing."""
    print("\n" + "=" * 60)
    print("Chunkwise Parallel Processing Example")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 256
    d_model = 512
    num_heads = 8
    chunk_size = 64
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Number of chunks: {seq_len // chunk_size}")
    
    # Create KDA layer with chunkwise processing
    kda_layer = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_chunkwise=True,
        chunk_size=chunk_size
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass with chunkwise processing
    output = kda_layer(x)
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Chunkwise processing successful!")
    print("  Chunkwise mode enables efficient parallel computation")


def comparison_example():
    """Compare sequential vs chunkwise implementations."""
    print("\n" + "=" * 60)
    print("Sequential vs Chunkwise Comparison")
    print("=" * 60)
    
    # Configuration
    batch_size = 1
    seq_len = 128
    d_model = 256
    num_heads = 4
    
    # Create both versions
    kda_sequential = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_chunkwise=False
    )
    
    kda_chunkwise = KimiDeltaAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_chunkwise=True,
        chunk_size=32
    )
    
    # Copy weights to ensure same initialization
    kda_chunkwise.load_state_dict(kda_sequential.state_dict())
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    with torch.no_grad():
        output_seq = kda_sequential(x)
        output_chunk = kda_chunkwise(x)
    
    print(f"\nSequential output shape: {output_seq.shape}")
    print(f"Chunkwise output shape: {output_chunk.shape}")
    
    # Note: Outputs may differ slightly due to numerical precision
    # in the chunkwise computation
    print(f"\nOutput statistics:")
    print(f"  Sequential - mean: {output_seq.mean():.6f}, std: {output_seq.std():.6f}")
    print(f"  Chunkwise  - mean: {output_chunk.mean():.6f}, std: {output_chunk.std():.6f}")
    
    print("\n✓ Both implementations produce valid outputs!")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    stateful_example()
    chunkwise_example()
    comparison_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
