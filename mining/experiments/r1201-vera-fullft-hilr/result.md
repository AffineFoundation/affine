# R1201 — result notes

## Status
- **p4317:** After **R1191 REFUTE ~0.02×**, freed T/K/chall by PID on `mine-r1191` H200; launched **FullFT HiLR** lr=**2e-6** ep=1 @8192 on `winner_za_high_l2` (406). TRAIN pid**24806** GPUs0–7; post_train pipe pid**25390** (`SKIP_HF_PUSH=1` `SKIP_LOCAL_TKC=0`). SSH `69.63.236.163:40299`.

## Axis
vera×FullFT×HiLR isolate after MidLR REFUTE. ≠ R1191 MidLR · ≠ Offline-DPO · ≠ Online-DPO · ≠ R1158 GRPO.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
