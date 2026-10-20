# Phase 4 results report — Precision-Gated / Kalman Delta Attention

This report is **generated from checked-in pre-registered configs** by
`examples/phase4_results_report.py` (issue #34, epic #25). Do not edit the
tables by hand; regenerate with:

```
python examples/phase4_results_report.py
```

All experiments are deterministic given the seeds in the embedded configs.
Claims are limited to the tested synthetic regime (acceptance criterion).

## Hypothesis and related-work boundary

**Hypotheses (fixed in the Phase 4 epic #25 before these experiments ran).**

- **H1 — mechanism.** Diagonal precision/evidence gating (a Kalman-flavored,
  per-channel write-strength control) improves recall over fixed-beta KDA
  under key-collision and overwrite stress.
- **H2 — measurement.** The variants' *native* uncertainty signals
  (precision along the key, evidence along the key) predict recall failure
  well enough to route hard queries to an exact fallback, buying recall per
  unit of fallback rate that a fixed-ratio router cannot; generic proxies
  (gate entropy, prediction-error norm, state effective rank) were expected
  to be weaker and are reported regardless of outcome.

**Related-work boundary.** KDA itself comes from *Kimi Linear* (Yang et al.,
arXiv:2510.26692) and is this repository's base architecture. **GDN-2**
(Gated DeltaNet-2, Hatamizadeh, Choi, Kautz, arXiv:2605.22791) and **EDA**
(Erase-then-Delta Attention, Li, Zhang, Luo et al., arXiv:2606.26560) are
published **prior baselines**, implemented in #37 strictly as references —
they are not contributions of this repository, and their sequential reference
recurrences intentionally omit the papers' chunkwise/kernel algorithms.
Phase 4's own contribution is measurement infrastructure (diagnostics #27,
calibration #28, routing #38) plus two small educational prototypes
(KDA-RLS-lite #36, diagonal precision-gated KDA #33). Nothing here is a
trained model or a production kernel.

## Exact recurrence definitions

All variants share the readout `o_t = S_t^T q_t` and differ only in the state
update (`S_bar = Diag(alpha_t) S_{t-1}` is the common decay step):

- **KDA** (base architecture, arXiv:2510.26692):
  `S_t = S_bar + beta_t k_t (v_t - S_bar^T k_t)^T` — scalar delta gate
  `beta_t` ties erase and write strength together.
- **KDA-RLS-lite** (#36 prototype): the KDA update with
  `beta_eff,t = clamp(beta_t / (lambda + e_t), beta_min, beta_max)`, where
  `e_t = sum_c n_{t,c} k_{t,c}^2` is the occupancy of a decaying evidence
  accumulator `n_t = rho alpha_t ⊙ n_{t-1} + k_t^2`. Repeated writes along a
  direction damp later writes there.
- **Precision-gated KDA** (#33 prototype):
  `g_t = (P_t^- ⊙ k_t) / (lambda + sum_c P^-_{t,c} k_{t,c}^2)`,
  `S_t = S_bar + g_t (v_t - S_bar^T k_t)^T`, with a diagonal precision
  time update (`P / forgetting + process_noise`) and an RLS-style downdate
  (`P ⊙ (1 - g ⊙ k)`), clamped to `[p_min, p_max]`.
- **GDN-2** (prior baseline, arXiv:2605.22791):
  `S_t = (I - k_t (b_t ⊙ k_t)^T) S_bar + k_t (w_t ⊙ v_t)^T` with
  channel-wise erase/write gates; tying `b_t = w_t = beta_t` recovers KDA
  (paper Appendix A.5) — the configuration used on the shared gate path.
- **EDA** (prior baseline, arXiv:2606.26560):
  `S_t = (I - beta_t k_t k_t^T)(I - gamma_t e_t e_t^T) S_bar
  + beta_t k_t v_t^T` with an independently addressed L2-normalized erase
  direction; `gamma = 0` recovers KDA, and `e_t = k_t` collapses to a
  stronger same-address correction — the configuration used here, since the
  shared task path has no learned erase addresses.

## Task generators and seeds

All tasks come from the #35 generators (`kda/tasks/generators.py`);
gate distributions mirror the #28 pre-registration
(`alpha = sigmoid(N(0,1) + 2.0)`, `beta = sigmoid(N(0,1) + 1.0)`, drawn
from `seed + 10000` independently of task content). Seeds and sizes are
embedded verbatim in the configs below.


## Capacity / interference curves

Recall accuracy (fraction of queries whose readout from the final state reaches cosine 0.500 with the target) on the key-collision task, with cluster-bootstrap 95% CIs over sequences. GDN-2/EDA run on the shared alpha/beta gate path, where their tied/zero-erase configurations reduce to KDA exactly (the paper ties asserted by the #37 tests) — their rows are regression checks of those reductions, not independent mechanisms.

### Capacity sweep (collision cosine 0.900)

| variant | pairs=2 | pairs=4 | pairs=8 | pairs=16 |
|---|---|---|---|---|
| kda | 0.646 [0.583, 0.688] | 0.635 [0.599, 0.667] | 0.411 [0.380, 0.456] | 0.220 [0.202, 0.238] |
| kda-rls-lite | 0.646 [0.583, 0.677] | 0.641 [0.599, 0.672] | 0.469 [0.430, 0.518] | 0.246 [0.228, 0.263] |
| precision-gated-kda | 0.719 [0.667, 0.760] | 0.714 [0.656, 0.760] | 0.534 [0.513, 0.555] | 0.271 [0.255, 0.286] |
| gdn-2 | 0.646 [0.583, 0.688] | 0.635 [0.599, 0.667] | 0.411 [0.380, 0.456] | 0.220 [0.202, 0.238] |
| eda | 0.646 [0.583, 0.688] | 0.635 [0.599, 0.667] | 0.411 [0.380, 0.456] | 0.220 [0.202, 0.238] |

### Interference sweep (fixed load 8 pairs)

| variant | cos=0.000 | cos=0.500 | cos=0.800 | cos=0.900 | cos=0.990 |
|---|---|---|---|---|---|
| kda | 0.625 [0.594, 0.654] | 0.576 [0.536, 0.628] | 0.458 [0.422, 0.508] | 0.411 [0.380, 0.456] | 0.396 [0.375, 0.430] |
| kda-rls-lite | 0.651 [0.615, 0.688] | 0.633 [0.596, 0.672] | 0.531 [0.495, 0.581] | 0.469 [0.430, 0.518] | 0.432 [0.401, 0.479] |
| precision-gated-kda | 0.688 [0.661, 0.711] | 0.672 [0.641, 0.698] | 0.596 [0.576, 0.620] | 0.534 [0.513, 0.555] | 0.497 [0.471, 0.521] |
| gdn-2 | 0.625 [0.594, 0.654] | 0.576 [0.536, 0.628] | 0.458 [0.422, 0.508] | 0.411 [0.380, 0.456] | 0.396 [0.375, 0.430] |
| eda | 0.625 [0.594, 0.654] | 0.576 [0.536, 0.628] | 0.458 [0.422, 0.508] | 0.411 [0.380, 0.456] | 0.396 [0.375, 0.430] |


## Overwrite / recency behavior

Same-key overwrite task: each key is re-written with a fresh value every round. **Fresh recall accuracy** measures the readout of the *latest* value per key (cosine 0.500 criterion, bootstrap 95% CIs); the **stale-dominance rate** is the fraction of queries whose readout is closer to an overwritten value than to the latest one. The EDA arm erases at the current key with constant gate 0.500 (the paper's `e_t = k_t` collapse — the only erase-address choice available on the shared task path; a pre-registered baseline configuration, not a tuned design).

### Fresh recall accuracy

| variant | updates=2 | updates=4 | updates=8 |
|---|---|---|---|
| kda | 0.885 [0.844, 0.927] | 0.938 [0.901, 0.969] | 0.932 [0.896, 0.969] |
| kda-rls-lite | 0.906 [0.870, 0.938] | 0.943 [0.911, 0.974] | 0.938 [0.901, 0.974] |
| precision-gated-kda | 0.979 [0.958, 0.995] | 0.995 [0.984, 1.000] | 0.990 [0.974, 1.000] |
| eda | 0.885 [0.844, 0.927] | 0.932 [0.901, 0.958] | 0.927 [0.885, 0.958] |

### Stale-dominance rate

| variant | updates=2 | updates=4 | updates=8 |
|---|---|---|---|
| kda | 0.021 [0.005, 0.036] | 0.010 [0.000, 0.026] | 0.010 [0.000, 0.026] |
| kda-rls-lite | 0.036 [0.010, 0.063] | 0.010 [0.000, 0.026] | 0.010 [0.000, 0.026] |
| precision-gated-kda | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.016] |
| eda | 0.010 [0.000, 0.026] | 0.005 [0.000, 0.016] | 0.005 [0.000, 0.016] |


## Edit locality

Each write's *edit* is `DeltaS_t = S_t - Diag(alpha_t) S_{t-1}`, computed from replayed state trajectories (captured traces only — no model reruns). **Relative update norm** `||DeltaS_t||_F / ||S_t||_F` measures edit magnitude (the first write into an empty state always scores 1). **Key-axis alignment** `||k_t k_t^T DeltaS_t||_F / ||DeltaS_t||_F` measures how concentrated the edit is along the write key: exactly 1 for KDA/RLS-lite (`beta k`, `beta_eff k` write vectors) and for EDA's `e_t = k_t` collapse, smaller when the edit spreads off-key (the precision-gated gain `g_t`).

### Relative update norm

| variant | updates=2 | updates=4 | updates=8 |
|---|---|---|---|
| kda | 0.600 | 0.572 | 0.559 |
| kda-rls-lite | 0.586 | 0.560 | 0.548 |
| precision-gated-kda | 0.599 | 0.568 | 0.549 |
| eda | 0.608 | 0.582 | 0.570 |

### Key-axis alignment

| variant | updates=2 | updates=4 | updates=8 |
|---|---|---|---|
| kda | 1.000 | 1.000 | 1.000 |
| kda-rls-lite | 1.000 | 1.000 | 1.000 |
| precision-gated-kda | 0.998 | 0.995 | 0.986 |
| eda | 1.000 | 1.000 | 1.000 |


## Uncertainty calibration (#28 experiment, default config)

Do internal uncertainty signals predict recall failure? AUROC of each pre-registered signal against the failure label on the key-collision task (cluster-bootstrap 95% CIs; 0.5 = chance, < 0.5 = anti-correlated, reported as-is). Full details and pre-registration in PR #47.

### Failure rates

| variant | failure rate | mean recall cosine |
|---|---|---|
| kda | 0.589 [0.544, 0.620] | 0.468 |
| kda-rls-lite | 0.531 [0.482, 0.570] | 0.502 |
| precision-gated-kda | 0.466 [0.445, 0.487] | 0.531 |

### AUROC of each uncertainty signal

| signal | kda | kda-rls-lite | precision-gated-kda |
|---|---|---|---|
| evidence_along_key | unavailable | 0.786 [0.753, 0.814] | unavailable |
| precision_along_key | unavailable | unavailable | 0.872 [0.843, 0.897] |
| prediction_error_norm | 0.373 [0.321, 0.428] | 0.425 [0.368, 0.482] | 0.406 [0.351, 0.459] |
| key_collision_score | 0.564 [0.518, 0.605] | 0.607 [0.556, 0.648] | 0.573 [0.518, 0.616] |
| gate_entropy | 0.459 [0.397, 0.520] | 0.479 [0.427, 0.526] | 0.486 [0.435, 0.548] |
| state_effective_rank | 0.245 [0.217, 0.274] | 0.241 [0.209, 0.276] | 0.154 [0.121, 0.188] |


## Consequence: hybrid routing (#38 experiment, default config)

Uncertainty-aware routing between the compressed memory and an exact-attention fallback. Each cell shows the best and worst operating points over the pre-registered threshold grid, as `gain @ fallback rate` (gain = recall improvement over a matched-budget fixed-ratio router at that operating point; fallback rate = cost proxy, no wall-clock measured). Flat and negative signals are shown in full, not filtered. Full grid in PR #48.

### Endpoints

| variant | compressed accuracy | exact accuracy | headroom |
|---|---|---|---|
| kda | 0.411 | 1.000 | 0.589 |
| kda-rls-lite | 0.469 | 1.000 | 0.531 |
| precision-gated-kda | 0.534 | 1.000 | 0.466 |

### Best / worst adaptive gain over matched fixed-ratio routing

| signal | kda | kda-rls-lite | precision-gated-kda |
|---|---|---|---|
| prediction_error_norm | 0.005 @ 0.010 / -0.047 @ 0.500 | 0.005 @ 0.052 / -0.034 @ 0.500 | 0.003 @ 0.010 / -0.036 @ 0.201 |
| precision_along_key | unavailable | unavailable | 0.146 @ 0.500 / 0.008 @ 0.010 |
| evidence_along_key | unavailable | 0.094 @ 0.299 / 0.010 @ 0.010 | unavailable |
| key_collision_score | 0.023 @ 0.344 / 0.005 @ 0.625 | 0.029 @ 0.156 / 0.008 @ 0.031 | 0.021 @ 0.625 / 0.003 @ 0.031 |
| gate_entropy | 0.005 @ 0.010 / -0.016 @ 0.102 | 0.005 @ 0.010 / -0.016 @ 0.500 | 0.003 @ 0.010 / -0.018 @ 0.102 |


## Computational overhead

Asymptotic per-step costs of the **sequential reference recurrences** in this
repository (no chunkwise WY forms, no fused kernels). No wall-clock latency
is measured anywhere in Phase 4; the hybrid-routing experiments use the
*fallback rate* (fraction of queries answered exactly) as the only cost
proxy.

| mechanism | state memory | per-step update cost | notes |
|---|---|---|---|
| KDA | `O(d_h * d_v)` | `O(d_h * d_v)` | one rank-1 update |
| KDA-RLS-lite (#36) | `O(d_h * d_v)` + `O(d_h)` evidence | `O(d_h * d_v)` + `O(d_h)` | evidence accumulator + scalar damping |
| precision-gated KDA (#33) | `O(d_h * d_v)` + `O(d_h)` precision | `O(d_h * d_v)` + `O(d_h)` | diagonal precision update/downdate |
| GDN-2 (prior baseline) | `O(d_h * d_v)` | `O(d_h * d_v)` | tied config identical to KDA |
| EDA (prior baseline) | `O(d_h * d_v)` | `O(d_h * d_v)` | + one rank-1 erase when enabled |
| exact-attention fallback (#38) | `O(L * (d_h + d_v))` raw cache | `O(L * (d_h + d_v))` per query | the cost the router avoids on easy queries |

The educational prototypes deliberately stay sequential and readable; no
production-performance or kernel-level speedup is claimed.

## Negative results and limitations

Reported per the repository's honest-negatives rule; numbers are in the
tables above.

- **Generic uncertainty proxies are flat or anti-correlated** in this
  regime: `gate_entropy` sits at chance in the calibration table and never
  beats fixed-ratio routing; `prediction_error_norm` is *anti*-correlated
  with failure (AUROC < 0.5, negative routing gains at most operating
  points); `state_effective_rank` is strongly anti-correlated — a recency
  artifact (the state fills with time while failures concentrate on early,
  decayed writes), not a usable signal here.
- **`key_collision_score` is only weakly positive** — directionally useful
  but far below the native precision/evidence signals it is a proxy for.
- **The baselines do not differentiate on the shared gate path**: GDN-2
  (tied) and EDA (`gamma = 0`) coincide with KDA by construction, so the
  capacity/interference tables contain duplicate rows by design (they
  double as regression checks of the paper ties). The EDA overwrite arm uses
  the `e_t = k_t` collapse with a pre-registered constant gate — the paper's
  learned, independently addressed erase is *not* evaluated.
- **Mechanism gains are modest, not transformative**: precision/evidence
  gating buys a few accuracy points at high collision cosines and reduces
  stale-value dominance at these defaults; it does not remove the capacity
  cliff.
- **Scope**: synthetic tasks only; gates are sampled, not learned; no
  training runs; sequential reference recurrences only; ECE uses a
  deterministic min-max score-to-probability mapping (no fitted
  recalibration); bootstrap CIs are percentile intervals; claims are limited
  to the tested regime (acceptance criterion).

## Pre-registered configs (embedded verbatim)


<details><summary>Capacity/interference config (issue #34 sweeps)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "capacity_collision_cosine": 0.9,
  "capacity_loads": [
    2,
    4,
    8,
    16
  ],
  "ci_level": 0.95,
  "failure_threshold": 0.5,
  "gate_seed_offset": 10000,
  "interference_cosines": [
    0.0,
    0.5,
    0.8,
    0.9,
    0.99
  ],
  "interference_num_pairs": 8,
  "key_dim": 32,
  "seeds": [
    0,
    1,
    2
  ],
  "value_dim": 32,
  "variants": [
    "kda",
    "kda-rls-lite",
    "precision-gated-kda",
    "gdn-2",
    "eda"
  ]
}
```

</details>

<details><summary>Overwrite/recency config (issue #34 sweeps)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "ci_level": 0.95,
  "eda_erase_gate": 0.5,
  "failure_threshold": 0.5,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "num_keys": 8,
  "seeds": [
    0,
    1,
    2
  ],
  "update_counts": [
    2,
    4,
    8
  ],
  "value_dim": 32,
  "variants": [
    "kda",
    "kda-rls-lite",
    "precision-gated-kda",
    "eda"
  ]
}
```

</details>

<details><summary>Calibration config (#28)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "ci_level": 0.95,
  "collision_cosine": 0.9,
  "ece_bins": 10,
  "entropy_bins": 20,
  "failure_threshold": 0.5,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "metrics": [
    "auroc",
    "average_precision",
    "ece",
    "calibration_bins"
  ],
  "num_pairs": 8,
  "recall_bins": 10,
  "seeds": [
    0,
    1,
    2
  ],
  "signals": [
    "evidence_along_key",
    "precision_along_key",
    "prediction_error_norm",
    "key_collision_score",
    "gate_entropy",
    "state_effective_rank"
  ],
  "value_dim": 32,
  "variants": [
    "kda",
    "kda-rls-lite",
    "precision-gated-kda"
  ]
}
```

</details>

<details><summary>Hybrid-routing config (#38)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "collision_cosine": 0.9,
  "entropy_bins": 20,
  "failure_threshold": 0.5,
  "fallback_temperature": 0.1,
  "fixed_ratio_seed": 0,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "num_pairs": 8,
  "seeds": [
    0,
    1,
    2
  ],
  "signals": [
    "prediction_error_norm",
    "precision_along_key",
    "evidence_along_key",
    "key_collision_score",
    "gate_entropy"
  ],
  "threshold_quantiles": [
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99
  ],
  "value_dim": 32,
  "variants": [
    "kda",
    "kda-rls-lite",
    "precision-gated-kda"
  ]
}
```

</details>
