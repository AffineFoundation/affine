# R895 — v4 n80 vs reign36 (p4007/p4009)

**Axis:** marsplan×Offline-DPO×HiAlpha×MidRank×HiBeta×MidCtx×MegaSuperExtra×ep4×UltraLoLR  
**Base:** `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` · β=0.3 · r=32 · α=128 · lr=5e-7 · @8192 · steps=19200

## Decision (pre-registered)
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) **and** thought≥80 **and** B≥0.30 vs reign36 (k=3, τ=0.03).

## Result — **REFUTE v4**
| metric | value |
|---|---|
| margin | **−0.008260** |
| SE | 0.004702 |
| z | −1.757 |
| n | 79 |
| bar max(2·SE,δ) | 0.009404 (~**−0.88×**) |
| thought median | 197 ✓ |
| B pass | 0.468 ✓ |
| chall Reason | 0.013148 |
| king Reason | 0.021140 |

Artifact: `r895_sim_result_reign36_wvk7.json` · chall reaped R337 4,5 · slot → **R909 TRAIN** (ShortCtx Midβ).
