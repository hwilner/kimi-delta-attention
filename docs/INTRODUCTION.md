# Introduction: From Softmax Attention to Kimi Delta Attention

This document gives a technical newcomer the full background needed to understand
what this repository implements and why it matters. It assumes familiarity with
transformers at the level of "I know attention has queries, keys, and values,"
and builds everything else up from there.

![Concept figure: Kimi Delta Attention — key, value, write strength beta, and per-channel gate alpha feed the delta-rule update (erase old entry, then write new) of a fixed-size recurrent state S, read out by the query; contrasted with a KV cache that grows with sequence length.](figures/concept_figure.svg)

*Figure 1. KDA maintains a fixed-size state S updated by the delta rule with per-channel gating, so memory stays constant while a standard KV cache grows linearly with the sequence.*

## 1. The quadratic-cost problem of softmax attention

Standard transformer attention computes, for every token, a weighted average over
**all** previous token values, with weights given by softmax similarities between
queries and keys. This means every forward pass materializes an `L × L` attention
matrix, where `L` is the sequence length. Compute and memory grow **quadratically**
with context length: doubling the context quadruples the attention work.

At training time this is painful but parallelizable. At inference time it is worse
in a subtler way: to generate each new token you must keep around the key and
value vectors of every previous token — the **KV cache** — whose size grows linearly
with the sequence and is often the binding memory constraint for long-context
deployment.

The central question of the "efficient attention" literature is: can we get the
expressivity of attention without the quadratic matrix and the growing cache?

## 2. Linear attention: attention as a recurrence

The key algebraic observation is that softmax attention can be approximated —
or replaced — by a form where the softmax normalization is dropped and the
attention weights factorize. **Katharopoulos et al. ("Transformers are RNNs",
ICML 2020)** showed that if attention is written with a kernel feature map
`φ`, the computation can be reorganized so that instead of comparing each query
to every key, you maintain a running summary and update it token by token [4].
Attention becomes a recurrent neural network: constant memory per token at
inference, no KV cache.

**The Performer (Choromanski et al., ICLR 2021)** took a complementary approach:
FAVOR+ random features approximate the softmax kernel itself, giving unbiased or
low-variance estimates of true softmax attention in linear time [5].

Both lines share the same fundamental idea: replace explicit token-to-token
comparison with a **fixed-size state** that accumulates information. The question
then becomes: what exactly should that state be, and how should it be updated?

## 3. The fast-weight / associative-memory view

A powerful reinterpretation comes from **Schlag, Irie, and Schmidhuber, "Linear
Transformers Are Secretly Fast Weight Programmers" (ICML 2021)** [3]. In this view,
the linear-attention state `S` is a **fast weight matrix** — an associative memory
that maps keys to values. Writing a token means adding an outer product `k vᵀ` to
the state; reading means multiplying the state by a query.

The naive additive update has a problem: if two keys collide (are similar), their
values interfere, and there is no way to *correct* or *overwrite* a stored
association. The same authors' **DeltaNet** update fixes this with the **delta
rule** — a classic error-correcting learning rule. Before writing `k_t v_tᵀ`, the
state first *removes* whatever value it currently associates with `k_t`, then
writes the new one:

```
S_t = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
```

The `(I − β k kᵀ)` term erases the old association along the key direction; the
`β k vᵀ` term writes the new one. This is literally online least-squares
regression of values onto keys, a connection made precise by **Wang, Shi, and Fox,
"Test-Time Regression" (JMLR 2025)**, which shows that linear-attention state
updates are online regression steps, with gating playing the role of weight
decay [13]. Relatedly, **von Oswald et al. (ICML 2023)** showed that linear
attention layers can implement gradient descent in-context, reinforcing the view
of the state as a learned, optimizable memory [12].

The practical blocker was parallelism: the delta rule looks inherently sequential.
**Yang, Wang, Zhang, Shen, and Kim (NeurIPS 2024)** solved this with a **chunkwise**
parallel form of DeltaNet, making delta-rule models trainable at scale on modern
hardware [6].

## 4. State-space rivals

In parallel, the state-space model (SSM) lineage attacked the same problem from a
different direction:

- **RetNet (Sun et al., 2023)** — retention instead of attention: a recurrent
  formulation with a fixed scalar decay, plus a parallel training form [10].
- **RWKV (Peng et al., EMNLP 2023 Findings)** — an RNN with linear attention-like
  updates, trained at scale and competitive with transformers [9].
- **Mamba (Gu & Dao, COLM 2024)** — selective state spaces: the SSM parameters
  (including a decay/gate) become input-dependent, restoring content-based
  selectivity that plain SSMs lack [7].
- **Mamba-2 / SSD (Dao & Gu, ICML 2024)** — the state-space duality framework,
  unifying SSMs and linear attention as two views of the same computation [8].
- **GLA — Gated Linear Attention (Yang et al., ICML 2024)** — linear attention
  with data-dependent gating, showing gates are crucial for recall and length
  generalization [11].

The convergent lesson of all of these: a fixed-size recurrent state can match
attention **if** it has (a) input-dependent gating to decide what to forget, and
(b) a smart write rule to decide what to store.

## 5. Gated DeltaNet → KDA: per-channel gating

**Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025)** combined the two ideas:
the delta-rule write with a scalar forget gate `α_t` [2]:

```
S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ        (Gated DeltaNet)
```

Here `α_t` is a **scalar** — at every step, the *entire* memory is decayed by the
same amount. Forgetting is all-or-nothing across channels.

**Kimi Delta Attention (KDA), from "Kimi Linear: An Expressive, Efficient
Attention Architecture" (Kimi Team, 2025)** makes one surgical change: replace
the scalar `α_t` with a **diagonal matrix** `Diag(α_t)`, where `α_t ∈ [0,1]^{d_h}`
is a vector of per-channel decay factors [1]:

```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ   (KDA)
```

Each **channel** (row) of the state matrix now has its own decay rate. One channel
can hold a long-lived fact nearly forever (`α ≈ 1`) while another channel forgets
aggressively (`α ≈ 0`) — all decided per token, per channel, by the network.
This fine-grained memory management is the core expressivity upgrade: scalar
gating forces a single global timescale, whereas diagonal gating gives the model
a *spectrum* of timescales simultaneously, under the same delta-rule error
correction.

Crucially, the diagonal structure keeps this cheap: `Diag(α)` costs `O(d)` per
step, and the whole update remains chunkwise-parallelizable (the paper uses a
chunkwise DPLR — diagonal plus low-rank — form) [1].

## 6. Reported gains

From the Kimi Linear paper [1]:

- **75% reduction in KV-cache usage** for long-context scenarios (a hybrid layout
  keeps only a fraction of full-attention layers).
- **6× faster decoding** at 1M-token contexts.
- Reported as **the first linear-attention model to outperform full attention**
  under fair comparisons, with superior results on benchmarks including MMLU-Pro
  and RULER [16].

## 7. Honest limits of this educational implementation

This repository is a **clean-room, clarity-first PyTorch implementation** of the
KDA recurrence — `kda/core.py` implements the sequential state update;
`kda/attention.py` wraps it in a standard attention-layer interface. It is
deliberately **not**:

- Fast. The recurrent loop is sequential Python; the paper's performance comes
  from chunkwise DPLR kernels. For production, use the official
  [FLA implementation](https://github.com/fla-org/flash-linear-attention)
  with optimized Triton kernels.
- Trained. All tests use **synthetic/random data** to verify numerical
  correctness, not learned behavior.
- The full hybrid Kimi Linear model (which interleaves KDA with periodic full
  attention); we implement the KDA layer itself.

What it *is* good for: reading the math as code, experimenting with the state
update, and serving as the base for the state-inspection and editing research
described in `docs/ROADMAP.md`.

## References

1. Kimi Team. **Kimi Linear: An Expressive, Efficient Attention Architecture.** 2025. arXiv:2510.26692.
2. Yang, Kautz, Hatamizadeh. **Gated Delta Networks.** ICLR 2025. arXiv:2412.06464.
3. Schlag, Irie, Schmidhuber. **Linear Transformers Are Secretly Fast Weight Programmers.** ICML 2021. arXiv:2102.11174.
4. Katharopoulos et al. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.** ICML 2020. arXiv:2006.16236.
5. Choromanski et al. **Rethinking Attention with Performers.** ICLR 2021. arXiv:2009.14794.
6. Yang, Wang, Zhang, Shen, Kim. **Parallelizing Linear Transformations with the Delta Rule over Sequence Length.** NeurIPS 2024. arXiv:2406.06484.
7. Gu, Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** COLM 2024. arXiv:2312.00752.
8. Dao, Gu. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2).** ICML 2024. arXiv:2405.21060.
9. Peng et al. **RWKV: Reinventing RNNs for the Transformer Era.** EMNLP 2023 Findings. arXiv:2305.13048.
10. Sun et al. **Retentive Network: A Successor to Transformer for Large Language Models (RetNet).** 2023. arXiv:2307.08621.
11. Yang et al. **Gated Linear Attention Transformers with Hardware-Efficient Training.** ICML 2024. arXiv:2312.06635.
12. von Oswald et al. **Transformers Learn In-Context by Gradient Descent.** ICML 2023. arXiv:2212.07677.
13. Wang, Shi, Fox. **Test-Time Regression: A Unifying Framework for Sequence Models.** JMLR 2025. arXiv:2501.12352.
14. Meng, Bau et al. **Locating and Editing Factual Associations in GPT (ROME).** NeurIPS 2022. arXiv:2202.05262.
15. Sharma, Atkinson, Bau. **Locating and Editing Factual Associations in Mamba.** NeurIPS 2024. arXiv:2404.03646.
16. Hsieh et al. **RULER: What's the Real Context Size of Your Long-Context Language Models?** COLM 2024. arXiv:2404.06654.
