
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
- p4272: crown **R1129+R1133+R1134 REFUTE** (~−0.24× / +0.24× / −0.32×) → exact-PID reap :8002/:8004/:8003 → **R1146+R1147+R1148** UltraLoLR TRAIN pids**305741/305738/305731**; r337 R1125+R1137 + r338 R1135+R1136 also REFUTE idle (next). B300/B200×8=0. **Never `pkill -f`**.
- p4271: **R1127 REFUTE** m=−0.001642 SE=0.003235 ~**−0.25×** (thought✓193 B✓0.473 k=3) → exact-PID reap r339 :8003 → **R1145** ShortCtx LoRank Midβ Hyper UltraLoLR TRAIN pid**62090** GPUs6,7; R1143 still on 4,5. B300/B200×8=0. **Never `pkill -f`**.
- p4270: **R1126 REFUTE** ~**−0.38×** → **R1143** UltraLoLR; **R1128 REFUTE** ~**−0.78×** → **R1144** UltraLoLR; **R1127** TP1 util0.85 n80. **Never `pkill -f`**.
- p4269: **R1128** TP1 util**0.90** OOM on prompt-logprobs → **TP1 util0.85**. Prefer util≤0.85 for TP1 chall+logprobs on B200. **Never `pkill -f`**.
- p4268: **R1128** lean_chall polled wrong GPUs → TP2 NCCL stall → **TP1**. Prefer TP1 after first TP2 stall on r340. **Never `pkill -f`**.
- p4267: **R1120 REFUTE** ~**−1.52×** → **R1142** UltraLoLR. **Never `pkill -f`**.
- p4265: **R1121 REFUTE** ~**0.41×** (below 2·SE) → **R1141** UltraLoLR. Positive margin below bar is still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4262: **R1110 REFUTE** ~−0.62× → **R1138** UltraLoLR. **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** from stale merges → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** chal-00974 m=**−0.000659** (n80 was +0.006632 ~1.021×) — knife-edge n80 not live-predictive.
- p4254–p4261: REFUTE→next-axis TRAIN same pass; fill idle GPU pairs. **Never `pkill -f`**.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


