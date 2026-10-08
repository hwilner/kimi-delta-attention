# Kimi Delta Attention (KDA) - Educational Implementation

[![arXiv](https://img.shields.io/badge/arXiv-2510.26692-b31b1b.svg)](https://arxiv.org/abs/2510.26692)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CI](https://github.com/hwilner/kimi-delta-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/hwilner/kimi-delta-attention/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, educational implementation of **Kimi Delta Attention (KDA)** from the paper ["Kimi Linear: An Expressive, Efficient Attention Architecture"](https://arxiv.org/abs/2510.26692) by Moonshot AI.

## 🌟 What is Kimi Delta Attention?

Kimi Delta Attention (KDA) is a novel linear attention mechanism that refines the Gated DeltaNet architecture with **fine-grained diagonal gating**. Unlike traditional Gated DeltaNet which uses scalar decay, KDA introduces **per-channel decay control**, enabling more precise memory management and improved expressivity.

### Key Innovation

**Gated DeltaNet (Baseline):**
```
S_t = α_t(I - β_t k_t k_t^T)S_{t-1} + β_t k_t v_t^T
```
where α_t is a **scalar** decay factor.

**Kimi Delta Attention (KDA):**
```
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
```
where `Diag(α_t)` is a **diagonal matrix** enabling channel-wise gating.

### Performance Highlights

- ✅ **First linear attention to outperform full attention** under fair comparisons
- ✅ **75% reduction in KV cache usage** for long-context scenarios
- ✅ **6× faster decoding** at 1M token contexts
- ✅ **Superior performance** on MMLU-Pro, RULER, and RL-style benchmarks

## 📁 Repository Structure

```
kimi-delta-attention/
├── kda/
│   ├── __init__.py
│   ├── core.py              # Core KDA implementation
│   ├── chunkwise.py         # Chunkwise DPLR parallel form (exact WY representation)
│   ├── precision.py         # Block-scaled FP8 state cache & stochastic rounding
│   └── attention.py         # KDA attention layer
├── tests/
│   ├── test_kda_core.py     # Unit tests for KDA
│   ├── test_chunkwise.py    # DPLR parity + low-precision numerics tests
│   ├── test_attention.py    # Integration tests
│   └── test_packaging.py    # Version + doc/tree consistency checks
├── examples/
│   ├── basic_usage.py       # Simple usage example
│   └── benchmark.py         # Performance benchmarking
├── docs/
│   ├── ARCHITECTURE.md      # Detailed architecture explanation
│   ├── CHUNKWISE.md         # Chunkwise DPLR derivation & stability analysis
│   └── TESTING.md           # Testing methodology
├── README.md
├── requirements.txt
└── setup.py
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Install from source

```bash
git clone https://github.com/hwilner/kimi-delta-attention.git
cd kimi-delta-attention
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 💡 Quick Start

```python
import torch
from kda import KimiDeltaAttention

# Initialize KDA layer
batch_size, seq_len, d_model = 2, 128, 512
num_heads, head_dim = 8, 64

kda = KimiDeltaAttention(
    d_model=d_model,
    num_heads=num_heads,
    head_dim=head_dim
)

# Forward pass
x = torch.randn(batch_size, seq_len, d_model)
output = kda(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## ⚡ Chunkwise DPLR Parallel Form

`kda/chunkwise.py` implements the chunkwise **Diagonal-Plus-Low-Rank (DPLR)** reformulation of the KDA recurrence (WY representation, chunk size `C = 64`). It replaces the sequential, memory-bandwidth-bound state loop with batched matmuls and one triangular solve per chunk, and is **algebraically identical** to `KDACore` — parity is enforced by `tests/test_chunkwise.py` across multi-chunk, ragged-length, streaming-state, and gradient-flow configurations. All cumulative decay ratios are computed in log space, so the parallel form cannot overflow/underflow for any decay setting. See [docs/CHUNKWISE.md](docs/CHUNKWISE.md) for the full derivation.

```python
from kda import KDADPLRChunkwise

kda = KDADPLRChunkwise(head_dim=64, value_dim=64, chunk_size=64)
out, state = kda(q, k, v, alpha, beta)                            # same interface as KDACore
out2, state = kda(q2, k2, v2, alpha2, beta2, initial_state=state) # streaming
```

`kda/precision.py` adds the Phase-1 numerics utilities: a block-scaled FP8 (e4m3) state cache and unbiased stochastic rounding, as correctness oracles for the planned fused Triton kernel.

## 🧪 Testing

**Note:** All tests use **synthetic/random data** for validation purposes only. This is standard practice in ML research implementations to verify correctness without requiring real datasets.

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_kda_core.py -v

# Run with coverage
pytest tests/ --cov=kda --cov-report=html
```

See [TESTING.md](docs/TESTING.md) for detailed information about the testing methodology.

## 📊 Benchmarks

Compare the KDA implementations against PyTorch's scaled-dot-product attention:

```bash
python examples/benchmark.py --seq-len 1024 --batch-size 4 --num-heads 8
```

The script times the sequential and chunkwise KDA paths plus an SDPA reference on
your own hardware. These are locally-measured timings, not published benchmark
numbers: KDA is a recurrent linear-attention formulation and SDPA is quadratic
full attention, so the table is a scale check rather than a head-to-head claim.

## 🎓 Educational Focus

This implementation prioritizes **clarity and educational value** over production-level optimization. Key features:

- ✅ **Clean, readable code** with extensive comments
- ✅ **Mathematical formulations** documented inline
- ✅ **Step-by-step explanations** of the algorithm
- ✅ **Comprehensive tests** with synthetic data
- ✅ **Detailed architecture documentation**

For production use, consider the official [FLA implementation](https://github.com/fla-org/flash-linear-attention) with optimized Triton kernels.

## 📚 Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Detailed explanation of KDA mechanism
- [Chunkwise DPLR Formulation](docs/CHUNKWISE.md) - Parallel scan derivation and numerical stability
- [Testing Methodology](docs/TESTING.md) - How we validate the implementation
- [Project Board](docs/PROJECT.md) - Roadmap, issue cards, dependencies, and contribution paths
- [Live Contribution Board](https://github.com/users/hwilner/projects/8) - Ready, blocked, in-progress, and completed work
- [Examples](examples/) - Usage examples and benchmarks

## 🔗 Related Resources

- **Paper:** [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- **Official Implementation:** [FLA - Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- **Model Checkpoints:** [Kimi-Linear-48B on Hugging Face](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- **Original Gated DeltaNet:** [Paper](https://arxiv.org/abs/2412.06464)

## 🤝 Contributing

Contributions are welcome from implementers, researchers, educators, and first-time
open-source contributors. Help extending this educational implementation of the
original Kimi Linear work is especially welcome: roadmap cards include core
implementation, tests, baselines, experiments, examples, documentation, and API
improvements. See the [project board mirror](docs/PROJECT.md) and
[contribution guide](CONTRIBUTING.md) to choose a task; for a substantial new
direction, open an issue first so the scope can be agreed together.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Moonshot AI** for the original Kimi Linear paper and architecture
- **FLA Team** for the reference implementation and Triton kernels
- The broader research community working on efficient attention mechanisms

## 📧 Contact

For questions or discussions, please open an issue on GitHub.

---

**Disclaimer:** This is an educational implementation created for learning purposes. The official implementation with optimized kernels is available at [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
