
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
- p4350: r340 **R1230+R1231** MERGE idle GPUs3/6 → lean chall :8002/:8003 + **v4 n80**; r337 **R1234** MERGE GPU6 → :8002 n80; **R1227 REFUTE** m=+0.000446~0.09× / **R1228 REFUTE** m≈0 → exact-PID reap → **R1254+R1255 TRAIN**. **Never `pkill -f`**.
- p4349: crown **R1238+R1239** MERGE_DONE idle on free GPU1/3 → lean chall :8002/:8003 TP1 util0.85 + **v4 n80**; leave T/K + R1240 TRAIN 6,7. **Never `pkill -f`**.
- p4348: r339 **R1227+R1228** TRAIN_DONE/MERGE idle on free GPUs4–7 → lean chall :8002/:8003 + **v4 n80** (R1228 wait-arm after merge); leave T/K. **Never `pkill -f`**.
- p4347: r924 teacher **OOM@util0.90** (retune 16:53Z) left MERGE_DONE R1222/23/24 idle → revive :8000 GPU0 **util0.85+enforce-eager** then arm chall :8002/:8003/:8004 + v4 n80; leave king:8001. **Never `pkill -f`**.
- p4346: **R1225 REFUTE** m=+0.006423~0.90× / **R1226 REFUTE** m=−0.002089~−1.00× → exact-PID chall reap → **R1252+R1253 TRAIN**. **Never `pkill -f`**.
- p4345–p4335: REFUTE→reap→Mega cascade; Soft MidRank Loβ Mega MidLR=**R1010** do not re-run; SoftCtx HiRank MidLoβ Mega LR triad exhausted (R1020/R1040/R1207). **Never `pkill -f`**.
- p4335–p4325: free-poll GPU must match `GPUS=`; never `sed` CHALL_PORT launch line. **Never `pkill -f`**.
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


