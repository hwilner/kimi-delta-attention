# Concept Figure

The canonical concept figure for this repository is
[`concept_figure.svg`](concept_figure.svg) — a flat-design, NeurIPS/FigForge-style
diagram (white background, pastel modules, thin strokes) showing Kimi Delta Attention — delta-rule update of a fixed-size state with per-channel gating, readout, and KV-cache vs fixed-state memory contrast.
It is embedded near the top of [../INTRODUCTION.md](../INTRODUCTION.md) and
[../EXTENDED_INTRODUCTION.md](../EXTENDED_INTRODUCTION.md).

> Note: an AI-generated PNG rendering of the same figure (1536x1024) also
> exists but is kept out of git (binary assets are not committed via the
> project tooling). The SVG above is the source of truth.

If your viewer cannot render SVG, here is a faithful Mermaid sketch of the
same structure:

```mermaid
flowchart TB
    K[key k_t] --> U[Delta rule update:<br/>erase old entry, then write new]
    V[value v_t] --> U
    B[write strength beta_t] --> U
    A[per-channel gate alpha_t<br/>one knob per row] --> U
    S0[old state S_t-1] --> U
    U --> S1[fixed-size state S_t]
    S1 -->|next step| U
    S1 --> O[readout: o_t = S_t^T q_t]
    subgraph contrast["Memory contrast"]
        KV[Full attention: KV cache grows with sequence] 
        KD[KDA: state size fixed]
    end
```
