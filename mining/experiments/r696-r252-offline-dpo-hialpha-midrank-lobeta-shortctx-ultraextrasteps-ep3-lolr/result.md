# R696 — Short MidRank LoBeta UltraExtra (r252)

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80
`margin > max(2·SE, δ=0.002)` AND median `|z|≥80` AND B pass ≥0.30
vs live king (wvk=7, k=3, τ=0.03).

## n80 vs reign34 (p3754 harvest)

| field | value |
|---|---|
| margin | **−0.002708** |
| SE | 0.003309 |
| z | −0.818 |
| bar `max(2·SE, δ)` | **0.006619** |
| margin/bar | **~−0.41×** |
| thought median | 140.5 ✓ |
| B pass | 0.4375 ✓ |
| k / τ | 3 / 0.03 ✓ |
| n | 80 |
| challenger_wins | **false** |

**Verdict: REFUTE v4** — Short MidRank LoBeta UltraExtra fails LME.
Artifact: `r696_sim_result_reign34_wvk7.json`. Chall reaped crown 6,7 →
**R718 TRAIN**. Keep `/tmp/r696_merged`.
