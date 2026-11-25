# API Reference

This page documents the stable, import-facing API of the `kda` package. It is
verified against the actual signatures in `kda/core.py`, `kda/attention.py`,
`kda/__init__.py`, and `kda/variants/base.py`.

## Scope

The stable top-level package exports are:

```python
from kda import (
    KDACore,
    KDAChunkwise,
    KimiDeltaAttention,
    ShortConv1d,
    DeltaMemoryOutput,
    DeltaMemoryVariant,
    available_variants,
    create_variant,
)
```

The registry decorator and exception are available from `kda.variants`:

```python
from kda.variants import VariantNotFoundError, register_variant
```

There is currently **no `kda/utils.py` module** in the repository. The public
utilities are the variant-registry helpers documented below. Research
subpackages under `kda/tasks`, `kda/instrumentation`, `kda/routing`, and
`kda/experiments` are experiment-facing rather than stable top-level API; their
entry points are indexed in `examples/README.md` and summarized at the end of
this page.

## Shape conventions

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size |
| `L` | Sequence length |
| `D` | Model dimension (`d_model`) |
| `H` | Number of attention heads |
| `d_h` | Per-head query/key dimension (`head_dim`) |
| `d_v` | Per-head value dimension (`value_dim`) |
| `C` | Chunk size for `KDAChunkwise` |
| `S_t` | Recurrent state after token `t` |

Unless stated otherwise, tensors are PyTorch tensors with batch-first layout.
Direct `KDACore` and `KDAChunkwise` callers should L2-normalize `queries` and
`keys` along the last dimension. `KimiDeltaAttention` performs this
normalization internally.

## State-update equations

For one head at time step `t`:

```text
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t
```

Where:

- `S_t ∈ R^{d_h × d_v}` is the recurrent state matrix.
- `q_t, k_t ∈ R^{d_h}` are the query and key vectors.
- `v_t ∈ R^{d_v}` is the value vector.
- `α_t ∈ [0, 1]^{d_h}` is the per-channel diagonal decay vector.
- `β_t ∈ [0, 1]` is the scalar write/erase gate.

The implementation applies the update in three steps:

```python
state = alpha_t.unsqueeze(-1) * state
state = state - beta_t.unsqueeze(-1) * k_t @ (k_t.T @ state)
state = state + beta_t.unsqueeze(-1) * k_t @ v_t.T
output_t = state.T @ q_t
```

This is the same equation shown in `README.md` and `docs/ARCHITECTURE.md`.

## `KDACore`

Core sequential recurrent computation for one attention head.

### Constructor

```python
KDACore(head_dim: int, value_dim: int, eps: float = 1e-6)
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `head_dim` | `int` | required | Query/key dimension `d_h`. |
| `value_dim` | `int` | required | Value dimension `d_v`. |
| `eps` | `float` | `1e-6` | Reserved numerical-stability constant. |

### Forward

```python
outputs, final_state = core(
    queries,
    keys,
    values,
    alpha,
    beta,
    initial_state=None,
)
```

| Argument | Shape | Description |
| --- | --- | --- |
| `queries` | `(B, L, d_h)` | L2-normalized query vectors. |
| `keys` | `(B, L, d_h)` | L2-normalized key vectors. |
| `values` | `(B, L, d_v)` | Value vectors. |
| `alpha` | `(B, L, d_h)` | Per-channel decay values in `[0, 1]`. |
| `beta` | `(B, L, 1)` | Scalar gate values in `[0, 1]`. |
| `initial_state` | `(B, d_h, d_v)` or `None` | Optional recurrent state. Zeros are used when omitted. |

| Return | Shape | Description |
| --- | --- | --- |
| `outputs` | `(B, L, d_v)` | Per-token readouts `o_t = S_t^T q_t`. |
| `final_state` | `(B, d_h, d_v)` | Recurrent state after the final token. |

### Minimal example

```python
import torch
import torch.nn.functional as F
from kda import KDACore

batch, length, head_dim, value_dim = 2, 8, 16, 12
queries = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
keys = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
values = torch.randn(batch, length, value_dim)
alpha = torch.sigmoid(torch.randn(batch, length, head_dim))
beta = torch.sigmoid(torch.randn(batch, length, 1))

core = KDACore(head_dim=head_dim, value_dim=value_dim)
outputs, final_state = core(queries, keys, values, alpha, beta)

assert outputs.shape == (batch, length, value_dim)
assert final_state.shape == (batch, head_dim, value_dim)
```

## `KDAChunkwise`

Chunked version of the same recurrence. The current educational implementation
splits the sequence into chunks, processes each chunk with `KDACore`, and
carries the state across chunks. It pads inputs whose sequence length is not a
multiple of `chunk_size` and removes padding from the returned outputs.

### Constructor

```python
KDAChunkwise(
    head_dim: int,
    value_dim: int,
    chunk_size: int = 64,
    eps: float = 1e-6,
)
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `head_dim` | `int` | required | Query/key dimension `d_h`. |
| `value_dim` | `int` | required | Value dimension `d_v`. |
| `chunk_size` | `int` | `64` | Number of tokens processed per chunk `C`. |
| `eps` | `float` | `1e-6` | Reserved numerical-stability constant. |

### Forward

The forward signature, shapes, and return values match `KDACore.forward`.
If `L <= chunk_size`, the module falls back directly to `KDACore`.

### Minimal example

```python
import torch
import torch.nn.functional as F
from kda import KDAChunkwise

batch, length, head_dim, value_dim = 1, 20, 8, 8
queries = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
keys = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
values = torch.randn(batch, length, value_dim)
alpha = torch.sigmoid(torch.randn(batch, length, head_dim))
beta = torch.sigmoid(torch.randn(batch, length, 1))

chunked = KDAChunkwise(head_dim, value_dim, chunk_size=16)
outputs, final_state = chunked(queries, keys, values, alpha, beta)

assert outputs.shape == (batch, length, value_dim)
assert final_state.shape == (batch, head_dim, value_dim)
```

## `ShortConv1d`

Depthwise short convolution used by `KimiDeltaAttention` for local context.
It applies a causal-style trim after convolution so the output length matches
the input length.

### Constructor

```python
ShortConv1d(d_model: int, kernel_size: int = 4)
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `d_model` | `int` | required | Input and output channel dimension `D`. |
| `kernel_size` | `int` | `4` | Temporal convolution kernel size. |

### Forward

| Argument | Shape | Description |
| --- | --- | --- |
| `x` | `(B, L, D)` | Input sequence. |

| Return | Shape | Description |
| --- | --- | --- |
| `output` | `(B, L, D)` | Locally convolved sequence with unchanged length. |

### Minimal example

```python
import torch
from kda import ShortConv1d

conv = ShortConv1d(d_model=32, kernel_size=4)
x = torch.randn(2, 16, 32)
y = conv(x)
assert y.shape == x.shape
```

## `KimiDeltaAttention`

Complete multi-head KDA layer with neural parameterization of `q`, `k`, `v`,
`α`, and `β`.

### Constructor

```python
KimiDeltaAttention(
    d_model: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    use_short_conv: bool = True,
    conv_kernel_size: int = 4,
    chunk_size: int = 64,
    use_chunkwise: bool = False,
    dropout: float = 0.0,
)
```

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `d_model` | `int` | required | Model dimension `D`. |
| `num_heads` | `int` | required | Number of independent KDA heads `H`. |
| `head_dim` | `int` or `None` | `d_model // num_heads` | Per-head query/key dimension `d_h`. |
| `use_short_conv` | `bool` | `True` | Apply `ShortConv1d` before projections. |
| `conv_kernel_size` | `int` | `4` | Kernel size for the short convolution. |
| `chunk_size` | `int` | `64` | Chunk size used when `use_chunkwise=True`. |
| `use_chunkwise` | `bool` | `False` | Use `KDAChunkwise` instead of `KDACore`. |
| `dropout` | `float` | `0.0` | Dropout probability after output projection. |

`value_dim` is currently set equal to `head_dim`.

### Forward

```python
output = layer(x)
output, final_states = layer(
    x,
    initial_states=initial_states,
    return_states=True,
)
```

| Argument | Shape | Description |
| --- | --- | --- |
| `x` | `(B, L, D)` | Input token representations. |
| `initial_states` | `(B, H, d_h, d_v)` or `None` | Optional per-head recurrent states. |
| `return_states` | `bool` | If `True`, return final states in addition to outputs. |

| Return | Shape | Description |
| --- | --- | --- |
| `output` | `(B, L, D)` | Projected multi-head output. |
| `final_states` | `(B, H, d_h, d_v)` | Returned only when `return_states=True`. |

### Parameterization

For each head `h`, the layer computes:

```text
q^h = L2Norm(Swish(W_q^h x_conv))
k^h = L2Norm(Swish(W_k^h x_conv))
v^h = Swish(W_v^h x_conv)
α^h = Sigmoid(W_α^up W_α^down x)
β^h = Sigmoid(W_β^h x)
```

`x_conv` is the short-convolution output when `use_short_conv=True`, otherwise
`x`. The α parameterization uses a low-rank projection whose intermediate rank
is `head_dim`.

### Minimal examples

Basic forward pass:

```python
import torch
from kda import KimiDeltaAttention

layer = KimiDeltaAttention(d_model=64, num_heads=4)
x = torch.randn(2, 16, 64)
output = layer(x)
assert output.shape == x.shape
```

Stateful chunked decoding:

```python
layer = KimiDeltaAttention(d_model=64, num_heads=4)
chunk_1 = torch.randn(2, 8, 64)
chunk_2 = torch.randn(2, 8, 64)

out_1, state_1 = layer(chunk_1, return_states=True)
out_2, state_2 = layer(chunk_2, initial_states=state_1, return_states=True)

assert state_1.shape == (2, 4, layer.head_dim, layer.value_dim)
assert state_2.shape == state_1.shape
```

## Delta-memory variant API

Phase 4 research prototypes share a small recurrent-memory interface. The
classes and helpers below are public through `kda` or `kda.variants`.

### `DeltaMemoryOutput`

```python
@dataclass
class DeltaMemoryOutput:
    outputs: torch.Tensor
    final_state: torch.Tensor
    trace: dict[str, Any] = field(default_factory=dict)
```

| Field | Shape/type | Description |
| --- | --- | --- |
| `outputs` | `(B, L, d_v)` | Per-token variant outputs. |
| `final_state` | `(B, d_h, d_v)` | Final recurrent state. |
| `trace` | `dict[str, Any]` | Optional diagnostics collected when `return_trace=True`. |

### `DeltaMemoryVariant`

Abstract base class for recurrent delta-memory mechanisms.

```python
class DeltaMemoryVariant(nn.Module, ABC):
    def __init__(self, head_dim: int, value_dim: int) -> None: ...
    def reset(self) -> None: ...
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        gates: Optional[Mapping[str, torch.Tensor]] = None,
        initial_state: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> DeltaMemoryOutput: ...
```

Common input shapes:

| Argument | Shape | Description |
| --- | --- | --- |
| `queries` | `(B, L, d_h)` | Query vectors. |
| `keys` | `(B, L, d_h)` | Key vectors. |
| `values` | `(B, L, d_v)` | Value vectors. |
| `gates` | mapping | Variant-specific gate tensors. KDA-style variants use `alpha: (B, L, d_h)` and `beta: (B, L, 1)`. |
| `initial_state` | `(B, d_h, d_v)` or `None` | Optional recurrent state. |
| `return_trace` | `bool` | Populate `DeltaMemoryOutput.trace` when supported. |

`reset()` clears optional internal evidence buffers for variants that maintain
diagnostics between calls. The base implementation is a no-op.

### Registry helpers

```python
available_variants() -> tuple[str, ...]
create_variant(
    name: str,
    head_dim: int,
    value_dim: int,
    **kwargs,
) -> DeltaMemoryVariant
register_variant(name: str)
```

- `available_variants()` returns registered names in deterministic sorted
  order.
- `create_variant()` instantiates a registered class. Names are normalized by
  stripping whitespace and lowercasing. Unknown names raise
  `VariantNotFoundError` with the available names in the error message.
- `register_variant()` is a class decorator for custom
  `DeltaMemoryVariant` subclasses. Re-registering a different class under the
  same normalized name raises `ValueError`.

### Registered variants

| Registry name | Class | Role |
| --- | --- | --- |
| `kda` | `KDAVariant` | Adapter around `KDACore`. |
| `gdn-2` | `GatedDeltaNet2Variant` | Gated DeltaNet-2 reference baseline. |
| `eda` | `EraseThenDeltaVariant` | Erase-then-Delta Attention reference baseline. |
| `kda-rls-lite` | `KDARLSLiteVariant` | Phase 4 evidence-modulated prototype. |
| `precision-gated-kda` | `PrecisionGatedKDAVariant` | Phase 4 diagonal precision-gated prototype. |

GDN-2 and EDA are included as comparison baselines, not as novel
contributions of this repository.

Common constructor arguments for all registered variants are
`head_dim: int` and `value_dim: int`. Additional prototype arguments are:

| Class | Additional arguments |
| --- | --- |
| `KDAVariant` | `eps: float = 1e-6` |
| `GatedDeltaNet2Variant` | `eps: float = 1e-6` |
| `EraseThenDeltaVariant` | `eps: float = 1e-6` |
| `KDARLSLiteVariant` | `eps: float = 1e-6`, `evidence_enabled: bool = True`, `evidence_decay: float = 1.0`, `lambda_: float = 1.0`, `beta_min: float = 0.0`, `beta_max: float = 1.0` |
| `PrecisionGatedKDAVariant` | `eps: float = 1e-6`, `precision_enabled: bool = True`, `precision_init: float = 1.0`, `lambda_: float = 1.0`, `forgetting: float = 1.0`, `process_noise: float = 0.0`, `p_min: float = 0.001`, `p_max: float = 1000.0` |

### Minimal variant example

```python
import torch
import torch.nn.functional as F
from kda import create_variant

batch, length, head_dim, value_dim = 2, 8, 16, 12
queries = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
keys = F.normalize(torch.randn(batch, length, head_dim), dim=-1)
values = torch.randn(batch, length, value_dim)
gates = {
    "alpha": torch.sigmoid(torch.randn(batch, length, head_dim)),
    "beta": torch.sigmoid(torch.randn(batch, length, 1)),
}

variant = create_variant("kda-rls-lite", head_dim=head_dim, value_dim=value_dim)
result = variant(queries, keys, values, gates=gates, return_trace=True)

assert result.outputs.shape == (batch, length, value_dim)
assert result.final_state.shape == (batch, head_dim, value_dim)
assert isinstance(result.trace, dict)
```

## Research module map

These modules support the Phase 4 experiments and may evolve with the research
track. They are intentionally not re-exported as stable top-level API.

| Module | Public entry points |
| --- | --- |
| `kda.tasks` | Key-collision, overwrite, and novelty-probe synthetic task generators. |
| `kda.instrumentation` | State, gate, prediction-error, and key-collision diagnostics. |
| `kda.routing` | Exact-attention readout and hybrid-routing helpers. |
| `kda.experiments` | Calibration, capacity, overwrite, ablation, robustness, and report-consistency runners. |

For executable usage, see the scripts indexed in `examples/README.md`. For
experimental conclusions and known negative/flat results, see
`docs/PHASE4_RESULTS.md`.
