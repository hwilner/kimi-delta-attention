# Kimi Delta Attention - Architecture Deep Dive

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Key Innovation: Fine-Grained Gating](#key-innovation-fine-grained-gating)
4. [Implementation Details](#implementation-details)
5. [Chunkwise Parallelization](#chunkwise-parallelization)
6. [Neural Parameterization](#neural-parameterization)

## Overview

Kimi Delta Attention (KDA) is a linear attention mechanism that achieves **sub-quadratic complexity** while maintaining or exceeding the quality of full softmax attention. It accomplishes this through a refined recurrent state update with fine-grained gating.

### Complexity Comparison

| Mechanism | Time Complexity | Space Complexity (KV Cache) |
|-----------|----------------|----------------------------|
| Softmax Attention | O(L²) | O(L × d) |
| Linear Attention | O(L) | O(d²) |
| **Kimi Delta Attention** | **O(L)** | **O(d²)** |

Where L is sequence length and d is dimension.

## Mathematical Foundation

### Linear Attention as Online Learning

Linear attention can be viewed as maintaining a recurrent state that accumulates key-value associations:

```
S_t = S_{t-1} + k_t v_t^T
o_t = S_t^T q_t
```

where:
- `S_t ∈ R^{d_h × d_v}`: recurrent state matrix
- `k_t, q_t ∈ R^{d_h}`: keys and queries
- `v_t ∈ R^{d_v}`: values

**Problem**: This accumulates all associations without forgetting, leading to interference over long contexts.

### DeltaNet: Adding the Delta Rule

DeltaNet introduces the **delta rule** from associative memory, which corrects the state by removing outdated associations:

```
S_t = S_{t-1} - β_t k_t (k_t^T S_{t-1}) + β_t k_t v_t^T
    = (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
```

This is equivalent to a rank-1 update that removes the component of S in the direction of k_t before adding the new association.

### Gated DeltaNet: Adding Decay

Gated DeltaNet adds a **scalar decay** factor α_t to implement forgetting:

```
S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
```

where `α_t ∈ [0,1]` controls how much of the previous state to retain.

## Key Innovation: Fine-Grained Gating

### The Problem with Scalar Decay

Scalar decay `α_t` applies **uniform forgetting** across all channels of the state. This limits expressivity because different features may need different decay rates.

### Kimi Delta Attention Solution

KDA replaces scalar decay with **diagonal decay** `Diag(α_t)`:

```
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
```

where `α_t ∈ [0,1]^{d_h}` is a **vector** of per-channel decay factors.

### Why This Matters

1. **Fine-grained control**: Each channel can have its own decay rate
2. **Improved expressivity**: The model can learn which features to retain longer
3. **Better memory management**: More efficient use of the limited finite-state RNN memory
4. **Position awareness**: Different channels can encode different positional information

### Visual Comparison

```
Gated DeltaNet:
[α  0  0]   [s₁₁ s₁₂ s₁₃]
[0  α  0] × [s₂₁ s₂₂ s₂₃]  ← Same α for all rows
[0  0  α]   [s₃₁ s₃₂ s₃₃]

Kimi Delta Attention:
[α₁  0   0 ]   [s₁₁ s₁₂ s₁₃]
[0   α₂  0 ] × [s₂₁ s₂₂ s₂₃]  ← Different α for each row
[0   0   α₃]   [s₃₁ s₃₂ s₃₃]
```

## Implementation Details

### Core Recurrent Update

The state update consists of three steps:

```python
# Step 1: Apply diagonal decay
state = alpha_t.unsqueeze(-1) * state

# Step 2: Remove outdated association (delta rule)
k_state = torch.bmm(k_t.unsqueeze(-1), 
                   torch.bmm(k_t.unsqueeze(1), state))
state = state - beta_t.unsqueeze(-1) * k_state

# Step 3: Add new association
kv_outer = torch.bmm(k_t.unsqueeze(-1), v_t.unsqueeze(1))
state = state + beta_t.unsqueeze(-1) * kv_outer
```

### Output Computation

```python
# Compute output: o_t = S_t^T q_t
o_t = torch.bmm(state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
```

## Chunkwise Parallelization

### Motivation

Sequential recurrence is slow. KDA uses **chunkwise parallelization** to process multiple tokens in parallel while maintaining the recurrent structure across chunks.

### Approach

1. **Split sequence into chunks** of size C
2. **Inter-chunk recurrence**: Process chunks sequentially, passing state between them
3. **Intra-chunk parallelism**: Within each chunk, use WY representation to pack rank-1 updates

### WY Representation

The WY representation allows expressing a sequence of rank-1 updates as:

```
S_[t+1] = Diag(γ) S_[t] + (U - W S_[t]) V^T
```

where U, W, V are matrices that pack the chunk's updates. This enables:
- Fewer matrix multiplications
- Better hardware utilization
- Maintained numerical stability

### Efficiency Gains

Compared to standard DPLR (Diagonal-Plus-Low-Rank) formulation:
- **~100% efficiency improvement** through specialized DPLR variant
- **Reduced secondary chunk matrix computations**
- **Better consistency with classical delta rule**

## Neural Parameterization

### Input Processing

For each attention head h:

```python
# Short convolution for local context
x_conv = ShortConv(x)

# Project to queries, keys, values
q^h = L2Norm(Swish(W_q^h x_conv))
k^h = L2Norm(Swish(W_k^h x_conv))
v^h = Swish(W_v^h x_conv)
```

**Key design choices:**
- **ShortConv**: Captures local dependencies (kernel size ~4)
- **Swish activation**: Smooth, non-monotonic activation
- **L2 normalization**: Ensures numerical stability

### Gating Parameters

#### Per-Channel Decay (α)

```python
# Low-rank parameterization for efficiency
α_t = Sigmoid(W_α^+ W_α^↓ x)  # ∈ [0,1]^{d_h}
```

**Low-rank projection**:
- Down-project: `d_model → rank` (rank = head_dim)
- Up-project: `rank → num_heads × head_dim`
- Reduces parameters while maintaining expressivity

#### Scalar Gate (β)

```python
β_t = Sigmoid(W_β x)  # ∈ [0,1]
```

Controls the strength of the delta rule update.

### Multi-Head Attention

KDA uses **independent heads** that are processed separately and concatenated:

```python
for h in range(num_heads):
    out_h, state_h = KDA_core(q_h, k_h, v_h, α_h, β_h)
    
outputs = Concat([out_1, ..., out_H])
outputs = W_out outputs
```

## Hybrid Architecture

The full Kimi Linear model uses a **3:1 ratio** of KDA to global attention:

```
[KDA] → [KDA] → [KDA] → [MLA] → [KDA] → [KDA] → [KDA] → [MLA] → ...
```

where MLA is Multi-Head Latent Attention (full attention).

**Benefits:**
- KDA handles most processing efficiently
- MLA provides global context every 4 layers
- 75% reduction in KV cache
- 6× faster decoding at 1M context

## Performance Characteristics

### Advantages

1. **Linear complexity**: O(L) time, O(d²) space
2. **Long-context efficiency**: No quadratic bottleneck
3. **Stateful processing**: Can process sequences in chunks
4. **Hardware efficient**: Chunkwise algorithm optimized for GPUs

### Trade-offs

1. **Recurrent bottleneck**: Cannot fully parallelize across sequence length during training
2. **State size**: O(d²) state per head (though much smaller than O(L×d) KV cache)
3. **Approximation**: Linear attention is an approximation of softmax attention

## Comparison with Other Linear Attentions

| Method | Gating | Complexity | Key Feature |
|--------|--------|-----------|-------------|
| Linear Attention | None | O(L) | Simple, but limited quality |
| GLA | Scalar gate | O(L) | Data-dependent gating |
| DeltaNet | None | O(L) | Delta rule for correction |
| Gated DeltaNet | Scalar decay | O(L) | Combines gating + delta rule |
| **KDA** | **Diagonal decay** | **O(L)** | **Fine-grained per-channel control** |

## Conclusion

Kimi Delta Attention achieves state-of-the-art performance among linear attention mechanisms through its key innovation: **fine-grained diagonal gating**. This simple but effective modification enables more expressive memory management, leading to the first linear attention that outperforms full attention under fair comparisons.

The combination of:
- Fine-grained gating
- Efficient chunkwise parallelization  
- Careful neural parameterization
- Hybrid architecture with periodic global attention

makes KDA a compelling choice for long-context language modeling and agentic applications.
