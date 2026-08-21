# R1201 — vera×FullFT×HiLR (thought-only dense SFT)

## Claim
After R1191 FullFT MidLR (lr=1e-6 ep=1) REFUTE ~0.02×, HiLR isolate lr=2e-6 ep=1 @8192 on same vera6 reign36 base + winner_za_high_l2 clears next crown under Reason v4.

## Why distinct
≠ R1191 MidLR · ≠ Offline-DPO LoRA fleet · ≠ Online-DPO marsplan · ≠ R1158 Reason-GRPO · ≠ R227 genesis FullFT-HiLR (different base/king era).

## Knobs
- base: `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- method: thought-only full-FT (H121 recipe, no LoRA)
- lr=2e-6, epochs=1, max_len=8192, batch=1, grad_accum=8
- data: winner_za_high_l2.jsonl (~406 rows)

## Decision
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
