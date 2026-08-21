# R1097 — MidCtx HiRank MidLoβ HyperExtra MidLR

**Status (p4245):** **CHALL+n80 armed** on `mine-r340` GPUs3,4 :8003 (replaced stub lean; merges were idle since ~01:41Z).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **64** / **1e-6** |
| ctx / steps | MidCtx `@8192` / HyperExtra `38400` |
| parent | MidCtx HiRank MidLoβ Ultra MidLR R1033 REFUTE ~0.83× → Hyper MidLR isolate |
| p4245 | sibling of R1096 arm; outer pid**29334** chall pid**29471** |

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
