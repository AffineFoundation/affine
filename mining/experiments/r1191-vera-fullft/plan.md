# R1191 — vera×FullFT (thought-only dense SFT)

## Claim
Dense full-FT (no LoRA) on high-Λ2 thoughts from vera6 reign36 base clears next crown under Reason v4.

## Why distinct
Fleet is Offline-DPO LoRA + Online-DPO + one GRPO. Full-FT updates all non-visual params — structural axis.

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- method: thought-only full-FT (H121/R226 recipe)
- lr=1e-6, epochs=1, max_len=8192, batch=1, grad_accum=8
- data: winner_za_high_l2.jsonl (~406 rows)

## Decision
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
