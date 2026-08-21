
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
- p4282: **R1146 REFUTE** m=−0.001219 SE=0.003902 ~**−0.16×** (thought✓162 B✓0.438 k=3) + **R1148 REFUTE** m=−0.003190 SE=0.001552 ~**−1.03×** (thought✓161 B✓0.474 k=3) → exact-PID reap crown :8002/:8003 → **R1160** Hiβ UltraLoLR + **R1161** MidLoβ UltraLoLR TRAIN; sole `lium ls` 8×B200 still **fbb1135f** BL — do not rent. **Never `pkill -f`**.
- p4281: **r1158** teacher DL ✓ then still **1/8 GPU** → `lium rm` brave-matrix-2a; blind re-rent hit **same** exec **fbb1135f**@192.9.163.79 (eager-lion-45, API gpu_count=1) → tear again; **blind `lium up --gpu` bypasses executor_blacklist** — R1158 waiter now **node-id + ngpu≥8 gate**; fleet QUEUE HEAD=R1158. **Never `pkill -f`**.
- p4280: **R1139 REFUTE** m=−0.002210 SE=0.004226 ~**−0.26×** (thought✓211 B✓0.470 k=3) → exact-PID reap r924 :8002 pid**162026** → **R1159** UltraLoLR TRAIN pid**164539** GPUs6,7; **r1158** bootstrap LIVE (pip✓ king✓ teacher DL) but pod exposes **1/8 B200** (Device Minor 6) — tear after DL if still 1. **Never `pkill -f`**.
- p4279: **R1130 REFUTE** m=−0.003061 SE=0.003818 ~**−0.40×** (thought✓256 B✓0.377 k=3) → exact-PID reap r926 :8002 → **R1157** UltraLoLR TRAIN pid**169520** GPUs3,4; **R1139** merge ready but n80 waiter died on **softctx path typo** (dir is midctx) → re-arm lean chall :8002 GPUs6,7; **rent** `mine-r1158` 8×B200 $5.60/h (waiter `ls-empty` false-negative — `lium up --gpu B200 -c 8` succeeded). B300×8=0. **Never `pkill -f`**.
- p4278: **R1141 REFUTE** m=−0.001574 SE=0.001812 ~**−0.43×** (thought✓192 B✓0.479 k=3) → MidLoβ LR family exhausted → **R1156** Midβ UltraLoLR TRAIN. **Never `pkill -f`**.
- p4277: **R1131 REFUTE** ~**−0.62×** → **R1155** UltraLoLR. **Never `pkill -f`**.
- p4276: **R1130** TP1 util**0.85** FATAL KV −2.71 GiB on H100 → **TP1 util0.93**; **R1132 REFUTE** → **R1154**. **Never `pkill -f`**.
- p4275: **R1130** TP2 util**0.72** OOM → **TP1 util0.85**. **Never `pkill -f`**.
- p4274: **R1123 REFUTE** ~**−0.11×** → **R1153** UltraLoLR. **Never `pkill -f`**.
- p4273: r337/r338 REFUTE→**R1149–R1152** UltraLoLR. **Never `pkill -f`**.
- p4272: crown REFUTE→**R1146–R1148** UltraLoLR. **Never `pkill -f`**.
- p4271: **R1127 REFUTE** → **R1145** UltraLoLR. **Never `pkill -f`**.
- p4270: **R1126+R1128 REFUTE** → **R1143+R1144**. **Never `pkill -f`**.
- p4269: TP1 util**0.90** OOM → prefer util≤0.85 on B200. **Never `pkill -f`**.
- p4268: TP2 NCCL stall → prefer TP1. **Never `pkill -f`**.
- p4265: positive margin below 2·SE still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** knife-edge n80 not live-predictive.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


