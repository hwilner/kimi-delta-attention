# Chunkwise DPLR Formulation

This document describes the chunkwise Diagonal-Plus-Low-Rank (DPLR) parallel
form of Kimi Delta Attention implemented in `kda/chunkwise.py`
(roadmap Phase 1: Systems & Numerical Modernization).

## Motivation

The sequential recurrence

```
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
o_t = S_t^T q_t
```

loads and writes the `d_h x d_v` state matrix from/to high-bandwidth memory at
every token, making evaluation strictly memory-bandwidth-bound. The chunkwise
form partitions the sequence into blocks of length `C = 64` and replaces the
intra-chunk loop with batched matrix multiplications plus one `C x C`
triangular solve per chunk, while inter-chunk states propagate through a
prefix hand-off.

## Derivation sketch

Define the cumulative diagonal decay `Gamma_t = prod_{s<=t} Diag(alpha_s)`.
Rescaling the state as `S̃_t = Gamma_t^{-1} S_t` turns the recurrence into a
plain delta rule in transformed coordinates:

```
S̃_t = S̃_{t-1} + k̃_t u_t^T,
u_t = beta_t (v_t - k̄_t^T S̃_{t-1}),
k̃_t = Gamma_t^{-1} k_t,   k̄_t = Gamma_t k_t
```

Collecting the `u_t` over a chunk gives the triangular system

```
(I + tril(A, -1)) U = beta (V - K_bar S_0),
A_{ts} = beta_t k̄_t^T k̃_s = beta_t k_t^T (Gamma_t / Gamma_s) k_s   (s < t)
```

and outputs/state hand-off follow as

```
O     = (Gamma Q) S_0 + tril(B) U,   B_{ts} = q_t^T (Gamma_t / Gamma_s) k_s  (s <= t)
S_end = Gamma_C S_0 + [k_s (Gamma_C / Gamma_s)]^T U
```

## Numerical stability

Every decay ratio in the algorithm has the form `Gamma_t / Gamma_s` with
`s <= t`, hence is bounded in `(0, 1]`. All ratios are computed in log space
as `exp(log Gamma_t - log Gamma_s)`, where `log Gamma` is a cumulative sum of
`log alpha`. No intermediate quantity can overflow or underflow, regardless of
how small the per-channel decays become; this is what allows block-scaled FP8
or stochastic-rounding state caches (`kda/precision.py`) to be layered on top
without fighting conditioning issues in the parallel form itself.

## Correctness contract

`KDADPLRChunkwise` is algebraically identical to the sequential `KDACore`:
`tests/test_chunkwise.py` asserts output and final-state parity to
`atol = 1e-4` across single-chunk, multi-chunk, ragged-length, sub-chunk,
rectangular-state, non-zero-initial-state, and gradient-flow configurations.
This reference implementation is the oracle for the planned fused Triton
kernel (benchmark parity against FLA is tracked on the project board).
