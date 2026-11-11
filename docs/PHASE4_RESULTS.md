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


## Robustness: hyperparameter ablations (#51, #54)

The capacity numbers above are measured at each prototype's default hyperparameters. The two ablations below ask whether those results survive natural perturbations of the mechanisms' own knobs: every grid cell runs on the identical task/gate draws as the capacity pipeline, so the default cell (marked `*`) reproduces the corresponding capacity-table row above exactly. No cell is filtered — collapsed or flat settings are shown as-is. Full grids and analysis in PRs #51/#54; these tables are cross-checked cell-by-cell against a fresh recomputation by `examples/report_consistency.py`.

### Precision-gated KDA ablation (pairs=8, collision cosine 0.900)

| lambda | forgetting | process_noise | default | recall accuracy | mean recall cosine |
|---|---|---|---|---|---|
| 0.300 | 0.900 | 0.000 |  | 0.388 [0.365, 0.414] | 0.443 |
| 0.300 | 0.900 | 0.050 |  | 0.383 [0.362, 0.409] | 0.436 |
| 0.300 | 1.000 | 0.000 |  | 0.448 [0.430, 0.469] | 0.493 |
| 0.300 | 1.000 | 0.050 |  | 0.396 [0.370, 0.427] | 0.465 |
| 1.000 | 0.900 | 0.000 |  | 0.401 [0.372, 0.430] | 0.461 |
| 1.000 | 0.900 | 0.050 |  | 0.383 [0.352, 0.414] | 0.451 |
| 1.000 | 1.000 | 0.000 | * | 0.534 [0.513, 0.555] | 0.531 |
| 1.000 | 1.000 | 0.050 |  | 0.466 [0.440, 0.497] | 0.499 |
| 3.000 | 0.900 | 0.000 |  | 0.419 [0.396, 0.445] | 0.468 |
| 3.000 | 0.900 | 0.050 |  | 0.404 [0.378, 0.435] | 0.459 |
| 3.000 | 1.000 | 0.000 |  | 0.604 [0.573, 0.638] | 0.546 |
| 3.000 | 1.000 | 0.050 |  | 0.539 [0.510, 0.568] | 0.514 |

Fixed-beta KDA reference on identical draws: **0.411 [0.380, 0.456]** (mean recall cosine 0.468). Across-cell recall-accuracy spread (max - min over the 12 cells): **0.221**.

### Precision-gated KDA ablation (pairs=16, collision cosine 0.900)

| lambda | forgetting | process_noise | default | recall accuracy | mean recall cosine |
|---|---|---|---|---|---|
| 0.300 | 0.900 | 0.000 |  | 0.194 [0.177, 0.211] | 0.237 |
| 0.300 | 0.900 | 0.050 |  | 0.194 [0.177, 0.210] | 0.236 |
| 0.300 | 1.000 | 0.000 |  | 0.227 [0.211, 0.247] | 0.280 |
| 0.300 | 1.000 | 0.050 |  | 0.197 [0.181, 0.214] | 0.253 |
| 1.000 | 0.900 | 0.000 |  | 0.191 [0.174, 0.208] | 0.241 |
| 1.000 | 0.900 | 0.050 |  | 0.194 [0.177, 0.211] | 0.239 |
| 1.000 | 1.000 | 0.000 | * | 0.271 [0.255, 0.286] | 0.299 |
| 1.000 | 1.000 | 0.050 |  | 0.219 [0.201, 0.240] | 0.271 |
| 3.000 | 0.900 | 0.000 |  | 0.188 [0.171, 0.203] | 0.244 |
| 3.000 | 0.900 | 0.050 |  | 0.190 [0.173, 0.207] | 0.242 |
| 3.000 | 1.000 | 0.000 |  | 0.307 [0.292, 0.324] | 0.307 |
| 3.000 | 1.000 | 0.050 |  | 0.258 [0.238, 0.275] | 0.282 |

Fixed-beta KDA reference on identical draws: **0.220 [0.202, 0.238]** (mean recall cosine 0.259). Across-cell recall-accuracy spread (max - min over the 12 cells): **0.120**.

### KDA-RLS-lite ablation (pairs=8, collision cosine 0.900)

| evidence_decay | beta_min | beta_max | default | recall accuracy | mean recall cosine |
|---|---|---|---|---|---|
| 0.800 | 0.000 | 0.500 |  | 0.479 [0.445, 0.526] | 0.508 |
| 0.800 | 0.000 | 1.000 |  | 0.451 [0.411, 0.497] | 0.490 |
| 0.800 | 0.100 | 0.500 |  | 0.479 [0.445, 0.526] | 0.508 |
| 0.800 | 0.100 | 1.000 |  | 0.451 [0.411, 0.497] | 0.490 |
| 0.900 | 0.000 | 0.500 |  | 0.477 [0.443, 0.523] | 0.509 |
| 0.900 | 0.000 | 1.000 |  | 0.456 [0.417, 0.500] | 0.495 |
| 0.900 | 0.100 | 0.500 |  | 0.477 [0.443, 0.523] | 0.509 |
| 0.900 | 0.100 | 1.000 |  | 0.456 [0.417, 0.500] | 0.495 |
| 1.000 | 0.000 | 0.500 |  | 0.482 [0.448, 0.526] | 0.509 |
| 1.000 | 0.000 | 1.000 | * | 0.469 [0.430, 0.518] | 0.502 |
| 1.000 | 0.100 | 0.500 |  | 0.482 [0.448, 0.526] | 0.510 |
| 1.000 | 0.100 | 1.000 |  | 0.469 [0.430, 0.518] | 0.502 |

Fixed-beta KDA reference on identical draws: **0.411 [0.380, 0.456]** (mean recall cosine 0.468). Across-cell recall-accuracy spread (max - min over the 12 cells): **0.031**.

### KDA-RLS-lite ablation (pairs=16, collision cosine 0.900)

| evidence_decay | beta_min | beta_max | default | recall accuracy | mean recall cosine |
|---|---|---|---|---|---|
| 0.800 | 0.000 | 0.500 |  | 0.241 [0.225, 0.258] | 0.281 |
| 0.800 | 0.000 | 1.000 |  | 0.240 [0.221, 0.255] | 0.268 |
| 0.800 | 0.100 | 0.500 |  | 0.241 [0.225, 0.258] | 0.281 |
| 0.800 | 0.100 | 1.000 |  | 0.240 [0.221, 0.255] | 0.268 |
| 0.900 | 0.000 | 0.500 |  | 0.243 [0.228, 0.262] | 0.281 |
| 0.900 | 0.000 | 1.000 |  | 0.241 [0.223, 0.258] | 0.270 |
| 0.900 | 0.100 | 0.500 |  | 0.243 [0.228, 0.262] | 0.281 |
| 0.900 | 0.100 | 1.000 |  | 0.241 [0.223, 0.258] | 0.270 |
| 1.000 | 0.000 | 0.500 |  | 0.243 [0.227, 0.262] | 0.281 |
| 1.000 | 0.000 | 1.000 | * | 0.246 [0.228, 0.263] | 0.273 |
| 1.000 | 0.100 | 0.500 |  | 0.243 [0.227, 0.262] | 0.281 |
| 1.000 | 0.100 | 1.000 |  | 0.246 [0.228, 0.263] | 0.273 |

Fixed-beta KDA reference on identical draws: **0.220 [0.202, 0.238]** (mean recall cosine 0.259). Across-cell recall-accuracy spread (max - min over the 12 cells): **0.007**.

Reading (every claim is visible in the tables above):

- **The capacity-table advantage is not a lucky default**: the default cell sits mid-grid in both ablations, and the across-cell spread is small where the mechanism is well-behaved.
- **Precision-gated failure modes (honest negatives)**: forgetting `0.900` degrades recall at every grid point — at 16 pairs it erases the default cell's advantage over the KDA reference — and process noise `0.050` never helps materially (at 8 pairs every paired cell drops; at 16 pairs the effect is flat to negative). At these gate distributions, precision inflation from the forgetting/drift terms is counterproductive on this task family.
- **The default is not tuned**: `lambda = 3.000` (forgetting `1.000`, process noise `0.000`) scores slightly above the default at both operating points; the default is mid-grid, and the #34 "modest, not transformative" reading survives the ablation.
- **RLS-lite clamps are mostly inert here (honest negative)**: `beta_min = 0.100` leaves recall accuracy bit-identical at both operating points (the floor essentially never binds; mean recall cosine shifts by at most 1e-4); `beta_max = 0.500` slightly *raises* recall at 8 pairs with overlapping CIs — a direction, not a claim.


## Robustness across task regimes (#50, #52)

The calibration (#28) and routing (#38) tables above are measured at one pre-registered operating point (8 pairs, collision cosine 0.900). The two sweeps below re-run both pipelines on a pre-registered grid of task regimes (sequence load x collision cosine); every grid cell is rendered, no cell is filtered. Rows at the default operating point (marked `*`) reproduce the corresponding tables above exactly, since every grid cell runs on the identical task/gate draws as the base pipeline. Full grids and analysis in PRs #50/#52; these tables are cross-checked cell-by-cell against a fresh recomputation by `examples/report_consistency.py`.

### Calibration failure rate by regime (#50 grid)

| num_pairs | collision_cosine | variant | default | failure rate | mean recall cosine |
|---|---|---|---|---|---|
| 4 | 0.500 | kda |  | 0.130 [0.088, 0.177] | 0.753 |
| 4 | 0.500 | kda-rls-lite |  | 0.089 [0.057, 0.130] | 0.768 |
| 4 | 0.500 | precision-gated-kda |  | 0.062 [0.026, 0.109] | 0.793 |
| 4 | 0.900 | kda |  | 0.365 [0.333, 0.401] | 0.586 |
| 4 | 0.900 | kda-rls-lite |  | 0.359 [0.328, 0.401] | 0.615 |
| 4 | 0.900 | precision-gated-kda |  | 0.286 [0.240, 0.344] | 0.647 |
| 4 | 0.990 | kda |  | 0.432 [0.406, 0.464] | 0.545 |
| 4 | 0.990 | kda-rls-lite |  | 0.375 [0.344, 0.406] | 0.577 |
| 4 | 0.990 | precision-gated-kda |  | 0.385 [0.344, 0.427] | 0.609 |
| 8 | 0.500 | kda |  | 0.424 [0.372, 0.464] | 0.568 |
| 8 | 0.500 | kda-rls-lite |  | 0.367 [0.328, 0.404] | 0.588 |
| 8 | 0.500 | precision-gated-kda |  | 0.328 [0.302, 0.359] | 0.614 |
| 8 | 0.900 | kda | * | 0.589 [0.544, 0.620] | 0.468 |
| 8 | 0.900 | kda-rls-lite | * | 0.531 [0.482, 0.570] | 0.502 |
| 8 | 0.900 | precision-gated-kda | * | 0.466 [0.445, 0.487] | 0.531 |
| 8 | 0.990 | kda |  | 0.604 [0.570, 0.625] | 0.440 |
| 8 | 0.990 | kda-rls-lite |  | 0.568 [0.521, 0.599] | 0.478 |
| 8 | 0.990 | precision-gated-kda |  | 0.503 [0.479, 0.529] | 0.507 |
| 16 | 0.500 | kda |  | 0.695 [0.678, 0.714] | 0.300 |
| 16 | 0.500 | kda-rls-lite |  | 0.685 [0.667, 0.704] | 0.308 |
| 16 | 0.500 | precision-gated-kda |  | 0.652 [0.638, 0.667] | 0.334 |
| 16 | 0.900 | kda |  | 0.780 [0.762, 0.798] | 0.259 |
| 16 | 0.900 | kda-rls-lite |  | 0.754 [0.737, 0.772] | 0.273 |
| 16 | 0.900 | precision-gated-kda |  | 0.729 [0.714, 0.745] | 0.299 |
| 16 | 0.990 | kda |  | 0.779 [0.764, 0.797] | 0.251 |
| 16 | 0.990 | kda-rls-lite |  | 0.773 [0.758, 0.789] | 0.268 |
| 16 | 0.990 | precision-gated-kda |  | 0.753 [0.736, 0.771] | 0.293 |

### Calibration AUROC by regime (#50 grid)

| num_pairs | collision_cosine | signal | default | kda | kda-rls-lite | precision-gated-kda |
|---|---|---|---|---|---|---|
| 4 | 0.500 | evidence_along_key |  | unavailable | 0.781 [0.662, 0.885] | unavailable |
| 4 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.790 [0.694, 0.890] |
| 4 | 0.500 | prediction_error_norm |  | 0.469 [0.373, 0.555] | 0.495 [0.362, 0.610] | 0.300 [0.184, 0.432] |
| 4 | 0.500 | key_collision_score |  | 0.535 [0.404, 0.627] | 0.503 [0.348, 0.628] | 0.561 [0.386, 0.700] |
| 4 | 0.500 | gate_entropy |  | 0.581 [0.483, 0.658] | 0.541 [0.427, 0.642] | 0.566 [0.453, 0.674] |
| 4 | 0.500 | state_effective_rank |  | 0.314 [0.217, 0.417] | 0.265 [0.172, 0.351] | 0.208 [0.140, 0.273] |
| 4 | 0.900 | evidence_along_key |  | unavailable | 0.688 [0.614, 0.764] | unavailable |
| 4 | 0.900 | precision_along_key |  | unavailable | unavailable | 0.666 [0.601, 0.717] |
| 4 | 0.900 | prediction_error_norm |  | 0.284 [0.217, 0.355] | 0.340 [0.263, 0.420] | 0.300 [0.224, 0.362] |
| 4 | 0.900 | key_collision_score |  | 0.451 [0.392, 0.514] | 0.458 [0.403, 0.518] | 0.493 [0.422, 0.575] |
| 4 | 0.900 | gate_entropy |  | 0.550 [0.465, 0.633] | 0.520 [0.425, 0.612] | 0.574 [0.480, 0.669] |
| 4 | 0.900 | state_effective_rank |  | 0.405 [0.349, 0.475] | 0.415 [0.349, 0.472] | 0.405 [0.345, 0.474] |
| 4 | 0.990 | evidence_along_key |  | unavailable | 0.698 [0.623, 0.760] | unavailable |
| 4 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.746 [0.696, 0.788] |
| 4 | 0.990 | prediction_error_norm |  | 0.281 [0.230, 0.335] | 0.310 [0.244, 0.385] | 0.327 [0.253, 0.391] |
| 4 | 0.990 | key_collision_score |  | 0.498 [0.462, 0.533] | 0.499 [0.458, 0.532] | 0.469 [0.422, 0.508] |
| 4 | 0.990 | gate_entropy |  | 0.528 [0.451, 0.615] | 0.518 [0.435, 0.594] | 0.550 [0.459, 0.659] |
| 4 | 0.990 | state_effective_rank |  | 0.404 [0.365, 0.438] | 0.437 [0.380, 0.490] | 0.361 [0.316, 0.415] |
| 8 | 0.500 | evidence_along_key |  | unavailable | 0.840 [0.794, 0.887] | unavailable |
| 8 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.923 [0.904, 0.941] |
| 8 | 0.500 | prediction_error_norm |  | 0.465 [0.403, 0.524] | 0.479 [0.394, 0.547] | 0.481 [0.418, 0.529] |
| 8 | 0.500 | key_collision_score |  | 0.532 [0.478, 0.594] | 0.529 [0.478, 0.581] | 0.504 [0.446, 0.578] |
| 8 | 0.500 | gate_entropy |  | 0.498 [0.434, 0.576] | 0.491 [0.438, 0.548] | 0.499 [0.443, 0.579] |
| 8 | 0.500 | state_effective_rank |  | 0.121 [0.091, 0.151] | 0.123 [0.091, 0.163] | 0.073 [0.058, 0.089] |
| 8 | 0.900 | evidence_along_key | * | unavailable | 0.786 [0.753, 0.814] | unavailable |
| 8 | 0.900 | precision_along_key | * | unavailable | unavailable | 0.872 [0.843, 0.897] |
| 8 | 0.900 | prediction_error_norm | * | 0.373 [0.321, 0.428] | 0.425 [0.368, 0.482] | 0.406 [0.351, 0.459] |
| 8 | 0.900 | key_collision_score | * | 0.564 [0.518, 0.605] | 0.607 [0.556, 0.648] | 0.573 [0.518, 0.616] |
| 8 | 0.900 | gate_entropy | * | 0.459 [0.397, 0.520] | 0.479 [0.427, 0.526] | 0.486 [0.435, 0.548] |
| 8 | 0.900 | state_effective_rank | * | 0.245 [0.217, 0.274] | 0.241 [0.209, 0.276] | 0.154 [0.121, 0.188] |
| 8 | 0.990 | evidence_along_key |  | unavailable | 0.760 [0.733, 0.789] | unavailable |
| 8 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.812 [0.780, 0.840] |
| 8 | 0.990 | prediction_error_norm |  | 0.308 [0.261, 0.350] | 0.379 [0.318, 0.432] | 0.377 [0.316, 0.434] |
| 8 | 0.990 | key_collision_score |  | 0.511 [0.466, 0.551] | 0.512 [0.470, 0.552] | 0.507 [0.443, 0.570] |
| 8 | 0.990 | gate_entropy |  | 0.459 [0.397, 0.515] | 0.451 [0.393, 0.509] | 0.499 [0.443, 0.565] |
| 8 | 0.990 | state_effective_rank |  | 0.284 [0.252, 0.321] | 0.286 [0.247, 0.326] | 0.239 [0.209, 0.273] |
| 16 | 0.500 | evidence_along_key |  | unavailable | 0.768 [0.730, 0.799] | unavailable |
| 16 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.976 [0.969, 0.984] |
| 16 | 0.500 | prediction_error_norm |  | 0.493 [0.462, 0.530] | 0.508 [0.481, 0.541] | 0.508 [0.473, 0.549] |
| 16 | 0.500 | key_collision_score |  | 0.508 [0.453, 0.551] | 0.513 [0.455, 0.558] | 0.502 [0.452, 0.542] |
| 16 | 0.500 | gate_entropy |  | 0.499 [0.450, 0.548] | 0.487 [0.444, 0.530] | 0.495 [0.460, 0.534] |
| 16 | 0.500 | state_effective_rank |  | 0.129 [0.099, 0.167] | 0.130 [0.100, 0.167] | 0.068 [0.050, 0.091] |
| 16 | 0.900 | evidence_along_key |  | unavailable | 0.757 [0.724, 0.785] | unavailable |
| 16 | 0.900 | precision_along_key |  | unavailable | unavailable | 0.927 [0.913, 0.943] |
| 16 | 0.900 | prediction_error_norm |  | 0.361 [0.325, 0.409] | 0.426 [0.395, 0.466] | 0.451 [0.425, 0.494] |
| 16 | 0.900 | key_collision_score |  | 0.544 [0.501, 0.587] | 0.535 [0.483, 0.586] | 0.517 [0.477, 0.567] |
| 16 | 0.900 | gate_entropy |  | 0.534 [0.490, 0.580] | 0.531 [0.488, 0.570] | 0.512 [0.472, 0.563] |
| 16 | 0.900 | state_effective_rank |  | 0.174 [0.146, 0.202] | 0.174 [0.141, 0.208] | 0.123 [0.103, 0.143] |
| 16 | 0.990 | evidence_along_key |  | unavailable | 0.770 [0.730, 0.802] | unavailable |
| 16 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.907 [0.885, 0.924] |
| 16 | 0.990 | prediction_error_norm |  | 0.363 [0.328, 0.406] | 0.415 [0.376, 0.458] | 0.465 [0.427, 0.511] |
| 16 | 0.990 | key_collision_score |  | 0.506 [0.451, 0.569] | 0.504 [0.453, 0.565] | 0.519 [0.473, 0.567] |
| 16 | 0.990 | gate_entropy |  | 0.506 [0.460, 0.560] | 0.513 [0.466, 0.561] | 0.500 [0.460, 0.544] |
| 16 | 0.990 | state_effective_rank |  | 0.190 [0.164, 0.219] | 0.192 [0.164, 0.222] | 0.158 [0.140, 0.175] |

### Routing endpoints by regime (#52 grid)

| num_pairs | collision_cosine | variant | default | compressed accuracy | exact accuracy | headroom |
|---|---|---|---|---|---|---|
| 4 | 0.500 | kda |  | 0.870 | 1.000 | 0.130 |
| 4 | 0.500 | kda-rls-lite |  | 0.911 | 1.000 | 0.089 |
| 4 | 0.500 | precision-gated-kda |  | 0.938 | 1.000 | 0.062 |
| 4 | 0.900 | kda |  | 0.635 | 1.000 | 0.365 |
| 4 | 0.900 | kda-rls-lite |  | 0.641 | 1.000 | 0.359 |
| 4 | 0.900 | precision-gated-kda |  | 0.714 | 1.000 | 0.286 |
| 4 | 0.990 | kda |  | 0.568 | 0.990 | 0.422 |
| 4 | 0.990 | kda-rls-lite |  | 0.625 | 0.990 | 0.365 |
| 4 | 0.990 | precision-gated-kda |  | 0.615 | 0.990 | 0.375 |
| 8 | 0.500 | kda |  | 0.576 | 1.000 | 0.424 |
| 8 | 0.500 | kda-rls-lite |  | 0.633 | 1.000 | 0.367 |
| 8 | 0.500 | precision-gated-kda |  | 0.672 | 1.000 | 0.328 |
| 8 | 0.900 | kda | * | 0.411 | 1.000 | 0.589 |
| 8 | 0.900 | kda-rls-lite | * | 0.469 | 1.000 | 0.531 |
| 8 | 0.900 | precision-gated-kda | * | 0.534 | 1.000 | 0.466 |
| 8 | 0.990 | kda |  | 0.396 | 0.992 | 0.596 |
| 8 | 0.990 | kda-rls-lite |  | 0.432 | 0.992 | 0.560 |
| 8 | 0.990 | precision-gated-kda |  | 0.497 | 0.992 | 0.495 |
| 16 | 0.500 | kda |  | 0.305 | 1.000 | 0.695 |
| 16 | 0.500 | kda-rls-lite |  | 0.315 | 1.000 | 0.685 |
| 16 | 0.500 | precision-gated-kda |  | 0.348 | 1.000 | 0.652 |
| 16 | 0.900 | kda |  | 0.220 | 1.000 | 0.780 |
| 16 | 0.900 | kda-rls-lite |  | 0.246 | 1.000 | 0.754 |
| 16 | 0.900 | precision-gated-kda |  | 0.271 | 1.000 | 0.729 |
| 16 | 0.990 | kda |  | 0.221 | 0.988 | 0.767 |
| 16 | 0.990 | kda-rls-lite |  | 0.227 | 0.988 | 0.762 |
| 16 | 0.990 | precision-gated-kda |  | 0.247 | 0.988 | 0.741 |

### Routing best/worst adaptive gain by regime (#52 grid)

| num_pairs | collision_cosine | signal | default | kda | kda-rls-lite | precision-gated-kda |
|---|---|---|---|---|---|---|
| 4 | 0.500 | prediction_error_norm |  | 0.005 @ 0.302 / -0.010 @ 0.104 | 0.005 @ 0.052 / -0.010 @ 0.104 | 0.000 @ 0.052 / -0.026 @ 0.500 |
| 4 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.021 @ 0.500 / 0.000 @ 0.010 |
| 4 | 0.500 | evidence_along_key |  | unavailable | 0.031 @ 0.500 / 0.005 @ 0.010 | unavailable |
| 4 | 0.500 | key_collision_score |  | 0.010 @ 0.161 / 0.000 @ 0.807 | 0.005 @ 0.161 / -0.005 @ 0.443 | 0.000 @ 0.807 / 0.000 @ 0.807 |
| 4 | 0.500 | gate_entropy |  | 0.016 @ 0.500 / 0.000 @ 0.104 | 0.010 @ 0.500 / -0.010 @ 0.104 | 0.005 @ 0.500 / -0.005 @ 0.302 |
| 4 | 0.900 | prediction_error_norm |  | 0.000 @ 0.010 / -0.062 @ 0.500 | 0.005 @ 0.010 / -0.047 @ 0.500 | 0.005 @ 0.052 / -0.094 @ 0.500 |
| 4 | 0.900 | precision_along_key |  | unavailable | unavailable | 0.047 @ 0.302 / 0.005 @ 0.010 |
| 4 | 0.900 | evidence_along_key |  | unavailable | 0.068 @ 0.500 / 0.005 @ 0.010 | unavailable |
| 4 | 0.900 | key_collision_score |  | 0.005 @ 0.344 / -0.026 @ 0.135 | 0.005 @ 0.031 / -0.026 @ 0.135 | 0.010 @ 0.031 / -0.021 @ 0.615 |
| 4 | 0.900 | gate_entropy |  | 0.031 @ 0.302 / -0.016 @ 0.104 | 0.016 @ 0.302 / -0.026 @ 0.104 | 0.026 @ 0.302 / -0.005 @ 0.052 |
| 4 | 0.990 | prediction_error_norm |  | 0.000 @ 0.010 / -0.078 @ 0.500 | 0.000 @ 0.010 / -0.062 @ 0.500 | 0.000 @ 0.010 / -0.073 @ 0.500 |
| 4 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.083 @ 0.302 / 0.010 @ 0.010 |
| 4 | 0.990 | evidence_along_key |  | unavailable | 0.073 @ 0.500 / 0.005 @ 0.010 | unavailable |
| 4 | 0.990 | key_collision_score |  | 0.010 @ 0.333 / -0.016 @ 0.104 | 0.016 @ 0.333 / -0.016 @ 0.104 | 0.005 @ 0.010 / -0.021 @ 0.573 |
| 4 | 0.990 | gate_entropy |  | 0.031 @ 0.302 / -0.016 @ 0.104 | 0.016 @ 0.500 / -0.021 @ 0.104 | 0.026 @ 0.302 / 0.000 @ 0.104 |
| 8 | 0.500 | prediction_error_norm |  | 0.008 @ 0.052 / -0.016 @ 0.201 | 0.013 @ 0.052 / -0.005 @ 0.201 | 0.010 @ 0.500 / -0.005 @ 0.201 |
| 8 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.159 @ 0.500 / 0.010 @ 0.010 |
| 8 | 0.500 | evidence_along_key |  | unavailable | 0.117 @ 0.299 / 0.010 @ 0.010 | unavailable |
| 8 | 0.500 | key_collision_score |  | 0.016 @ 0.471 / -0.013 @ 0.799 | 0.018 @ 0.471 / -0.010 @ 0.799 | 0.026 @ 0.471 / -0.008 @ 0.133 |
| 8 | 0.500 | gate_entropy |  | 0.005 @ 0.010 / -0.021 @ 0.203 | 0.005 @ 0.500 / -0.021 @ 0.203 | 0.013 @ 0.500 / -0.010 @ 0.102 |
| 8 | 0.900 | prediction_error_norm | * | 0.005 @ 0.010 / -0.047 @ 0.500 | 0.005 @ 0.052 / -0.034 @ 0.500 | 0.003 @ 0.010 / -0.036 @ 0.201 |
| 8 | 0.900 | precision_along_key | * | unavailable | unavailable | 0.146 @ 0.500 / 0.008 @ 0.010 |
| 8 | 0.900 | evidence_along_key | * | unavailable | 0.094 @ 0.299 / 0.010 @ 0.010 | unavailable |
| 8 | 0.900 | key_collision_score | * | 0.023 @ 0.344 / 0.005 @ 0.625 | 0.029 @ 0.156 / 0.008 @ 0.031 | 0.021 @ 0.625 / 0.003 @ 0.031 |
| 8 | 0.900 | gate_entropy | * | 0.005 @ 0.010 / -0.016 @ 0.102 | 0.005 @ 0.010 / -0.016 @ 0.500 | 0.003 @ 0.010 / -0.018 @ 0.102 |
| 8 | 0.990 | prediction_error_norm |  | -0.003 @ 0.010 / -0.081 @ 0.500 | 0.005 @ 0.010 / -0.049 @ 0.500 | 0.000 @ 0.010 / -0.047 @ 0.201 |
| 8 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.117 @ 0.500 / 0.008 @ 0.010 |
| 8 | 0.990 | evidence_along_key |  | unavailable | 0.073 @ 0.500 / 0.010 @ 0.010 | unavailable |
| 8 | 0.990 | key_collision_score |  | 0.005 @ 0.286 / -0.010 @ 0.552 | 0.005 @ 0.286 / -0.008 @ 0.552 | 0.010 @ 0.286 / -0.010 @ 0.552 |
| 8 | 0.990 | gate_entropy |  | 0.000 @ 0.010 / -0.018 @ 0.102 | 0.005 @ 0.010 / -0.018 @ 0.500 | 0.008 @ 0.299 / -0.010 @ 0.102 |
| 16 | 0.500 | prediction_error_norm |  | 0.000 @ 0.010 / -0.013 @ 0.500 | 0.000 @ 0.010 / -0.010 @ 0.051 | 0.001 @ 0.500 / -0.007 @ 0.051 |
| 16 | 0.500 | precision_along_key |  | unavailable | unavailable | 0.167 @ 0.500 / 0.005 @ 0.010 |
| 16 | 0.500 | evidence_along_key |  | unavailable | 0.085 @ 0.500 / 0.004 @ 0.010 | unavailable |
| 16 | 0.500 | key_collision_score |  | 0.001 @ 0.512 / -0.005 @ 0.155 | 0.004 @ 0.512 / -0.007 @ 0.155 | 0.007 @ 0.512 / -0.007 @ 0.155 |
| 16 | 0.500 | gate_entropy |  | 0.003 @ 0.010 / -0.010 @ 0.102 | 0.003 @ 0.010 / -0.013 @ 0.102 | 0.004 @ 0.010 / -0.010 @ 0.102 |
| 16 | 0.900 | prediction_error_norm |  | 0.003 @ 0.010 / -0.047 @ 0.500 | 0.001 @ 0.010 / -0.033 @ 0.500 | 0.003 @ 0.201 / -0.016 @ 0.500 |
| 16 | 0.900 | precision_along_key |  | unavailable | unavailable | 0.128 @ 0.500 / 0.005 @ 0.010 |
| 16 | 0.900 | evidence_along_key |  | unavailable | 0.059 @ 0.301 / 0.004 @ 0.010 | unavailable |
| 16 | 0.900 | key_collision_score |  | 0.010 @ 0.370 / -0.013 @ 0.654 | 0.009 @ 0.370 / -0.017 @ 0.654 | 0.010 @ 0.370 / -0.012 @ 0.654 |
| 16 | 0.900 | gate_entropy |  | 0.007 @ 0.201 / -0.003 @ 0.102 | 0.007 @ 0.500 / -0.007 @ 0.102 | 0.009 @ 0.500 / 0.001 @ 0.102 |
| 16 | 0.990 | prediction_error_norm |  | 0.003 @ 0.010 / -0.049 @ 0.500 | 0.000 @ 0.010 / -0.026 @ 0.500 | 0.005 @ 0.201 / -0.005 @ 0.500 |
| 16 | 0.990 | precision_along_key |  | unavailable | unavailable | 0.115 @ 0.500 / 0.004 @ 0.010 |
| 16 | 0.990 | evidence_along_key |  | unavailable | 0.070 @ 0.500 / 0.003 @ 0.010 | unavailable |
| 16 | 0.990 | key_collision_score |  | 0.008 @ 0.299 / -0.013 @ 0.557 | 0.008 @ 0.299 / -0.008 @ 0.557 | 0.020 @ 0.299 / -0.007 @ 0.557 |
| 16 | 0.990 | gate_entropy |  | 0.007 @ 0.201 / -0.005 @ 0.102 | 0.008 @ 0.301 / -0.007 @ 0.102 | 0.008 @ 0.301 / -0.004 @ 0.102 |

Reading (every claim is visible in the tables above):

- **The #28 calibration headline is regime-stable**: the native signals stay well above chance in all nine cells — `precision_along_key` AUROC ranges [0.666, 0.976] and rises with load in every cosine column; `evidence_along_key` stays in [0.688, 0.840] with no monotone trend — while the generic proxies remain flat or anti-correlated everywhere (`prediction_error_norm` peaks at 0.508; `gate_entropy` stays within [0.451, 0.581]; `state_effective_rank` never exceeds 0.437).
- **The #38 routing consequence is regime-stable**: the native signals are the only ones with no losing operating point anywhere — their worst threshold's gain over matched fixed-ratio routing is >= 0.000 in every cell (best gains reach +0.167 for precision at 16 pairs). `prediction_error_norm` has a negative-worst operating point in 9/9 cells for every variant; `key_collision_score` and `gate_entropy` in 7-9 of 9.
- **Honest negative — the exact fallback is imperfect at collision cosine 0.990**: exact accuracy there is 0.988-0.992, not 1.000 — the temperature-0.100 softmax readout cannot fully separate near-identical keys — so the fallback is not literally exact at that interference level.
- **Failure rates grow with load and collision cosine** for all three variants (the 16-pair KDA row is flat between cosines 0.900 and 0.990: 0.780 vs 0.779). Precision-gated KDA has the lowest failure rate in 8 of 9 cells — KDA-RLS-lite edges it at (4 pairs, cosine 0.990), 0.375 vs 0.385 with overlapping CIs — and its margin over fixed-beta KDA narrows at 16 pairs, consistent with the "modest, not transformative" reading above.


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
- **The exact-attention fallback is not literally exact at extreme
  collision**: at collision cosine 0.990 its own recall accuracy is
  0.988-0.992 (the temperature-0.100 softmax readout cannot fully separate
  near-identical keys), visible in the #52 endpoints table above.
- **The prototypes are sensitive to their own knobs, mostly through failure
  modes**: in the ablation tables above, forgetting `0.900` degrades
  precision-gated recall at every grid point, process noise `0.050` never
  helps materially, and the RLS-lite lower clamp essentially never binds in
  these regimes; no ablation cell was filtered, so these are visible in
  full.
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

<details><summary>Precision-gated ablation config (#51)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "ci_level": 0.95,
  "failure_threshold": 0.5,
  "forgetting_grid": [
    0.9,
    1.0
  ],
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "lambda_grid": [
    0.3,
    1.0,
    3.0
  ],
  "operating_points": [
    [
      8,
      0.9
    ],
    [
      16,
      0.9
    ]
  ],
  "p_max": 1000.0,
  "p_min": 0.001,
  "precision_init": 1.0,
  "process_noise_grid": [
    0.0,
    0.05
  ],
  "reference_variants": [
    "kda"
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "value_dim": 32
}
```

</details>

<details><summary>RLS-lite ablation config (#54)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "beta_max_grid": [
    0.5,
    1.0
  ],
  "beta_min_grid": [
    0.0,
    0.1
  ],
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "ci_level": 0.95,
  "evidence_decay_grid": [
    0.8,
    0.9,
    1.0
  ],
  "failure_threshold": 0.5,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "lambda_": 1.0,
  "operating_points": [
    [
      8,
      0.9
    ],
    [
      16,
      0.9
    ]
  ],
  "reference_variants": [
    "kda"
  ],
  "seeds": [
    0,
    1,
    2
  ],
  "value_dim": 32
}
```

</details>

<details><summary>Calibration robustness config (#50)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "bootstrap_resamples": 200,
  "bootstrap_seed": 0,
  "ci_level": 0.95,
  "collision_cosine_grid": [
    0.5,
    0.9,
    0.99
  ],
  "entropy_bins": 20,
  "failure_threshold": 0.5,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "num_pairs_grid": [
    4,
    8,
    16
  ],
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

<details><summary>Routing robustness config (#52)</summary>

```json
{
  "alpha_bias": 2.0,
  "batch_size": 8,
  "beta_bias": 1.0,
  "collision_cosine_grid": [
    0.5,
    0.9,
    0.99
  ],
  "entropy_bins": 20,
  "failure_threshold": 0.5,
  "fallback_temperature": 0.1,
  "fixed_ratio_seed": 0,
  "gate_seed_offset": 10000,
  "key_dim": 32,
  "num_pairs_grid": [
    4,
    8,
    16
  ],
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
