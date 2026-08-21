
# LESSONS — durable findings (Reason v4 era; older lines labeled)
Hard-won knowledge, one line each. **Cap 150 lines.** Detail → `experiments/`.
S\* v2 era (retired 2026-08-10) → `archive/legacy-sstar-v2/` — ops only, not strategy.
**Contract SSOT:** live `api/v1/contract` — currently **wvk=7** Reason v4 (k=3, τ=0.03, n=1300).

## Scoring (Reason v4 tempered multi-sample + δ + thought-len + B, weight_version_key=7)
- **Per-ref a_i = lpC(y_i|z_A) − lpC(y_i|∅)**; turn Reason = **τ·log(mean_i exp(a_i/τ))** (LME; τ=0.03, k=3). Score = mean over turns. k=1 ≡ v3.
- **Crown (wvk=7, 2026-08-17):** paired mean(Reason_c − Reason_k) >
  **max(k_sigma · SE, min_margin)** with live `k_sigma=2.0` and **δ=`min_margin=0.002`**,
  **and** median stripped `len(z_A) ≥ min_thought_chars=80`, **and** teacher-side B
  pass rate ≥ `causality_gamma=0.30` (B=`lpC(y_A|z_A)−lpC(y_A|∅)` ≥ 0.02, no leakage).
- Miner-side causality / bank / r / baseline / L1lift are telemetry only — not the B license.
- Miner-side terms (L1lift, lpA, calibration r) do **not** enter Reason. Do not train them as objectives.
- Absolute Reason is only comparable within one duel slice. Use paired margin vs the live king.
- Confirm `weight_version_key` from `api/v1/contract` every pass (3→4→5→6→**7**). v4 favors **commit** to a teacher mode over hedge-filler.
- p3666: **R596 v4 REFUTE** m=−0.00206~−0.44× (thought✓176 B✓0.385 k=3) vs reign34 — prior k=1 ~1.31× was **not** predictive under LME; Soft HiRank MidBeta ep1 fails v4 re-sim.
- p3665: **R596 v4 re-sim LIVE** on warm R252 :8002 (pid247246) → `r596_*_reign34_wvk7.json`; fail-closed if stamp ≠ k=3 — prior p3588c ~1.31× was k=1 advisory only.
- p3664: fleet `affine_pkg` synced to live wvk=7 (k=3/τ=0.03/n=1300) on all 6 mine-*; LME smoke OK — next n80s are v4-isomorphic; pre-p3664 k=1 margins are advisory only.

## Strategy under Reason v4
- Shape `z_A` to **commit** to the teacher's dominant next action (LME rewards hits; filler ≈0).
- Teacher refs / distillation data remain the free starting point; score is teacher-side only.
- Submit when a fresh **v4** (k=3) slice sim clears **margin > max(k·SE, δ=0.002)** **and** thought/B vs the **live** king. Re-sim if the crown or wvk changed since the screen.
- p2399/p2401: mid-pipeline king flip — waiting `post_train` keeps old `KING_*` in process env; patching `mine.env` is not enough — kill-by-pidfile + relaunch **before** train.done (R69/R71/R73 guass→fjq); R67 vs fjq REFUTE m=−0.0115.

## Ops (still true — details in legacy archive if needed)
- p4342: r1158 idle GPUs **1,4–7** (teacher+GRPO on 0,2,3) → scp DPO trainer+data → **R1242** Short Midβ Mega HiLR (4,5) + **R1243** Short Midβ Mega Ultra (6,7) + **R1244** Soft Loβ Mega HiLR (1); leave T+GRPO. **Never `pkill -f`**.
- p4341: **R1207 REFUTE** m=+0.001811~0.44× (thought✓174 B✓0.397) → SoftCtx HiRank MidLoβ Mega LR triad exhausted (R1020/R1040/R1207) → exact-PID chall reap 243737/:8003 → **R1241** SoftCtx LoRank MidLoβ Mega UltraLoLR (6,7); leave T/K + **R1232** TRAIN 4,5. **Never `pkill -f`**.
- p4340: **R1215 REFUTE** m=−0.001713~−0.30× / **R1216 REFUTE** m=−0.003574~−0.63× (thought✓ B✓) → exact-PID chall reap 372808/:8004 + 372799/:8002 → **R1239** MidCtx MidRank Loβ Mega HiLR (1,3) + **R1240** Soft MidRank Hiβ Mega HiLR (6,7); leave T/K + **R1238** TRAIN 4,5. **Never `pkill -f`**.
- p4339: **R1217 REFUTE** m=+0.004629~0.89× → exact-PID chall reap → **R1238** ShortCtx LoRank Hiβ Mega UltraLoLR; r1191 → **R1235+36+37 TRAIN**. **Never `pkill -f`**.
- p4338: **R1208/10/11 REFUTE** → exact-PID reap → **R1232+33+34 TRAIN**. **Never `pkill -f`**.
- p4337: r340 GPUs6,7 idle → **R1231** Short HiRank Midβ Mega HiLR while crown n80 mid. **Never `pkill -f`**.
- p4336: crown MERGE_DONE idle → lean chall + **v4 n80 LIVE** R1215/16/17. **Never `pkill -f`**.
- p4335–p4325: REFUTE→reap→Mega cascade; Soft MidRank Loβ Mega MidLR=**R1010** do not re-run; free-poll GPU must match `GPUS=`; never `sed` CHALL_PORT launch line. **Never `pkill -f`**.
- p4324–p4300: H100 OOM→H200; Hyper/Mega cascade; teacher OOM@util0.90→TP1≤0.85; HF full⇒SKIP_HF_PUSH; fleet Removal via schedule-removal. **Never `pkill -f`**.
- p4307/p4303: rent non-BL **H200×8** when B300/B200 empty. **Never `pkill -f`**.
- p4298: lean free-poll own CUDA GPUs; probe model id from `/v1/models`. **Never `pkill -f`**.
- p4296/p4291: empty cmdline on finished chall ⇒ UUID-clear GPUs (do not FATAL). **Never `pkill -f`**.
- p4290: wrong EXP dirname leaves MERGE_DONE idle. **Never `pkill -f`**.
- p4285: wipe+seed Triton + probe `/v1/completions` before n80. **Never `pkill -f`**.
- p4281: blind `lium up --gpu` → BL; waiter = node-id+ngpu≥8. **Never `pkill -f`**.
- p4269: TP1 util**0.90** OOM → util≤0.85. **Never `pkill -f`**.
- p4268: TP2 NCCL stall → prefer TP1. **Never `pkill -f`**.
- p4265: positive margin below 2·SE still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** knife-edge n80 not live-predictive.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


