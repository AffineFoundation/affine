# R894 — v4 n80 vs reign36 (p4007/p4009)

**Axis:** marsplan×Offline-DPO×HiAlpha×MidRank×MidLoBeta×MidCtx×MegaSuperExtra×ep4×UltraLoLR  
**Base:** `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` · β=0.05 · r=32 · α=128 · lr=5e-7 · @8192 · steps=19200

## Decision (pre-registered)
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) **and** thought≥80 **and** B≥0.30 vs reign36 (k=3, τ=0.03).

## Result — **REFUTE v4**
| metric | value |
|---|---|
| margin | **−0.006518** |
| SE | 0.004164 |
| z | −1.566 |
| n | 78 |
| bar max(2·SE,δ) | 0.008327 (~**−0.78×**) |
| thought median | 200 ✓ |
| B pass | 0.573 ✓ |
| chall Reason | 0.015930 |
| king Reason | 0.021977 |

Artifact: `r894_sim_result_reign36_wvk7.json` · chall reaped R337 6,7 · slot → **R908 TRAIN** (ShortCtx MidLoβ).
