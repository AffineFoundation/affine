# R1191 — result notes

## Status
- **p4315:** pipe aborted (`pyarrow` missing; T/K crashed). Installed pyarrow; relaunched teacher+king **TP1 util0.90 `--enforce-eager`** + `VLLM_USE_FLASHINFER_SAMPLER=0` (chall kept on :8002). v4 n80 LIVE pid**23563** → `/root/affine_data/r1191_sim_result.json` (bh `ad0e37c3…`). Fixed local `run_sim_duel` CLI + bootstrap pyarrow. SSH `69.63.236.163:40299`.
- **p4314:** train+finalize done (`/tmp/r1191_merged`, weight_identical=false). Foreground HF push to `unconst/Affine-5czsc2fc98-r1191-fullft` **aborted** (HF public storage full). Resumed **local TKC** on same H200: `SKIP_MERGE=1 SKIP_LOCAL_TKC=0` pipe pid**6151**; teacher:8000 + king:8001 loading → chall:8002 → v4 n80. SSH `69.63.236.163:40299`.
- **p4307:** rented **8×H200** `mine-r1191-vera-fullft-1` / `swift-comet-4d` @$32/h by **node id** `4eb39f3b-3714-42f0-93eb-f81238c07834`. TTL→**2026-08-22T11:53Z**. FullFT TRAIN pid**2434**.

## Axis
vera6 reign36 × FullFT thought-only dense SFT (≠ Offline-DPO LoRA fleet; ≠ Online-DPO marsplan; ≠ R1158 Reason-GRPO). Base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
