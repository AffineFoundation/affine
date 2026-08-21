# R1191 — result notes

## Status
- **p4317: REFUTE v4** m=**+0.000109** SE=0.002318 z=0.047 n=80 bar≈0.004635 (~**0.02×**) thought✓(med180) B✓(0.479) k=3/τ=0.03 vs **reign36**. Challenger_wins=false. Freed TKC → **R1201** FullFT HiLR same pod.
- **p4315:** pipe aborted (`pyarrow` missing; T/K crashed). Installed pyarrow; relaunched teacher+king **TP1 util0.90 `--enforce-eager`** + `VLLM_USE_FLASHINFER_SAMPLER=0` (chall kept on :8002). v4 n80 LIVE pid**23563** → `/root/affine_data/r1191_sim_result.json` (bh `ad0e37c3…`).
- **p4314:** train+finalize done (`/tmp/r1191_merged`, weight_identical=false). HF push aborted (public storage full). Local TKC resume.
- **p4307:** rented **8×H200** `mine-r1191-vera-fullft-1` / `swift-comet-4d` @$32/h.

## Axis
vera6 reign36 × FullFT thought-only dense SFT lr=**1e-6** ep=1 @8192 (≠ Offline-DPO LoRA; ≠ Online-DPO; ≠ R1158 GRPO).

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36. **Failed** — MidLR FullFT ≈ noise under LME.
