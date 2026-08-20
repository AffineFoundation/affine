# R967 — plan / result log

## Decision rule (pre-registered)
Stage-5 iff fresh **v4** n80 margin > max(2·SE, δ=0.002) **and** thought≥80 **and** B≥0.30 vs reign36 (k=3, τ=0.03).

## Axis
vera Offline-DPO HiAlpha **LoRank(r=16)** Hiβ SoftCtx Ultra SuperExtra ep4 **HiLR(lr=2e-6)** @12288 max_steps=28800
Parent: R957 UltraLoLR REFUTE m=−0.002960 ~−0.79× → **HiLR isolate**.
≠ UltraLoLR R957 / ≠ SoftCtx MidRank Hiβ Ultra R945 / ≠ SoftCtx MidRank Hiβ Mega R938 / ≠ SoftCtx HiRank Hiβ Ultra R952 / ≠ Online / ≠ GRPO

## p4084
- Launched on `mine-crown-1` GPUs **1,3** after R957 chall reap
- train pid **110940** · wait→merge armed
- data: Soft Mid Mid Soft `dpo_duel_reason.jsonl` (604→259 kept)
- Artifacts on pod: `/root/affine_data/r967_train_launched.json` `/root/logs/r967_train.nohup`

## Verdict
*(pending TRAIN→MERGE→n80)*
