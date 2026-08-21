
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
- p4284: **R1143 REFUTE** ~**0.61×** → **R1163** MidLR; **R1151 REFUTE** ~**−0.02×** + **R1152 REFUTE** ~**−0.83×** → **R1164/R1165** MidLR; **R1145** n80 died on broken Triton `.so` when seed preferred stale `chall_r1145` — wipe+seed from **king** first. **Never `pkill -f`**.
- p4283: **R1147 REFUTE** m=+0.001654 SE=0.001158 ~**0.71×** (thought✓162 B✓0.425 k=3) → exact-PID reap crown :8004 → **R1162** SoftCtx LoRank Loβ MidLR TRAIN; r338 waiters died on **wrong LEAN path** (lorank-midlobeta-shortctx copy-paste ≠ hirank-lobeta-softctx / lorank-lobeta-midctx) → relaunch n80 after merge.done. **Never `pkill -f`**.
- p4282: **R1146 REFUTE** ~**−0.16×** + **R1148 REFUTE** ~**−1.03×** → **R1160/R1161** TRAIN; sole B200 **fbb1135f** BL. **Never `pkill -f`**.
- p4281: blind `lium up --gpu` bypasses executor_blacklist → re-hit **fbb1135f**; R1158 waiter = **node-id + ngpu≥8**. **Never `pkill -f`**.
- p4280: **R1139 REFUTE** ~**−0.26×** → **R1159**; r1158 1/8 GPU after DL. **Never `pkill -f`**.
- p4279: **R1130 REFUTE** ~**−0.40×** → **R1157**; softctx path typo on midctx n80. **Never `pkill -f`**.
- p4278: **R1141 REFUTE** ~**−0.43×** → **R1156**. **Never `pkill -f`**.
- p4277–p4272: UltraLoLR cascade after HiLR REFUTEs (R1155…R1146). **Never `pkill -f`**.
- p4269: TP1 util**0.90** OOM → util≤0.85 on B200. **Never `pkill -f`**.
- p4268: TP2 NCCL stall → prefer TP1. **Never `pkill -f`**.
- p4265: positive margin below 2·SE still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** knife-edge n80 not live-predictive.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


