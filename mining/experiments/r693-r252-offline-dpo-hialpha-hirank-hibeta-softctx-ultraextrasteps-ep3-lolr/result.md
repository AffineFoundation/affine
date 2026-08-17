# R693 — Soft HiRank HiBeta SoftCtx UltraExtra (r252)

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80
`margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30
vs live king (wvk=7, k=3, τ=0.03).

## n80 vs reign34 (p3753 harvest)

| field | value |
|---|---|
| margin | **+0.002353** |
| SE | 0.001626 |
| z | 1.447 |
| bar `max(2·SE, δ)` | **0.003252** |
| margin/bar | **~0.72×** |
| thought median | 159.5 ✓ |
| B pass | 0.425 ✓ |
| k / τ | 3 / 0.03 ✓ |
| n | 80 |
| challenger_wins | **false** |

**Verdict: REFUTE v4** — positive but fails 2σ. Artifact:
`r693_sim_result_reign34_wvk7.json`. Chall reaped crown 4,5 → **R717 TRAIN**.
Keep `/tmp/r693_merged`.
