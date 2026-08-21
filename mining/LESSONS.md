
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
- p4270: **R1126 REFUTE** m=−0.002916 ~**−0.38×** (thought✓194 B✓0.40) → **R1143** UltraLoLR TRAIN r339 GPUs4,5; **R1128 REFUTE** m=−0.004148 ~**−0.78×** (thought✓188 B✓0.475) → **R1144** UltraLoLR TRAIN r340 GPUs6,7; **R1127** prior TP2 chall KeyboardInterrupt mid-tokenizer → reuse merge + **TP1 util0.85** :8003 GPU6 n80 pid**61316**. Fill idle pairs same pass. **Never `pkill -f`**.
- p4269: **R1128** p4268 TP1 chall util**0.90** OOM on prompt-logprobs (~+7.6GiB need, ~2.7 free) → n80 ConnectError; exact-PID reap → **TP1 util0.85** :8004 CHALL_READY ~85s + n80 pid**67441** (alive past probe). Reap scripts must **not** match own basename (self-kill). Prefer util≤0.85 for TP1 chall+logprobs on B200. **Never `pkill -f`**.
- p4268: **R1128** MERGE done but lean_chall polled **GPUs1,2** (busy R1142) not **6,7** → idle forever; then TP2 NCCL stall VRAM≈1GiB :8004 dead → exact-PID reap → **TP1** chall GPU6 util0.90 pid**63587** CHALL_READY ~85s + n80 pid**65193**. Fix lean poll `-i 6,7` + awk `index()`. Prefer TP1 after first TP2 stall on r340. **Never `pkill -f`**.
- p4267: **R1120 REFUTE** m=−0.013799 SE=0.004529 ~**−1.52×** (thought✓209 B✓0.521 k=3) → exact-PID reap r340 :8002 → **R1142** SoftCtx MidRank MidLoβ Hyper UltraLoLR TRAIN pid**60510** GPUs1,2; touch `r1128_train.done` unblocked MERGE. B300/B200×8=0. **Never `pkill -f`**.
- p4266: **R1120** TP2 NCCL stall (VRAM≈1GiB, :8002 never listens) → exact-PID reap → **TP1** chall GPU1 util0.90 + n80; Prefer TP1 after first TP2 stall on r340. **Never `pkill -f`**.
- p4265: **R1121 REFUTE** m=+0.002685 SE=0.003287 ~**0.41×** → **R1141** MidCtx UltraLoLR TRAIN r340 GPUs3,4. Positive margin below 2·SE is still REFUTE. **Never `pkill -f`**.
- p4264: **R1121** TP2 stall→TP1; **R1114→R1139** + **R1118→R1140** TRAIN. Lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4263: **R1121** stuck chall after NCCL → re-arm; purge r926+r252 stale merges. **Never `pkill -f`**.
- p4262: **R1110 REFUTE** ~−0.62× → **R1138** UltraLoLR TRAIN r252. **Never `pkill -f`**.
- p4261: **R1122+R1124 REFUTE** → R1135+R1136; **R1119 REFUTE** → R1137; disk purge r338+r924. **Never `pkill -f`**.
- p4260: **R1112+13 REFUTE** → R1131+R1132; **R1116+17 REFUTE** → R1133+R1134. **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** from stale merges → purge + re-arm n80 + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** chal-00974 m=**−0.000659** SE=0.000858 z=−0.77 (n80 was +0.006632 ~1.021×) — knife-edge n80 not live-predictive.
- p4257: **R1101 REFUTE** ~−0.48× → **R1129** TRAIN crown. **Never `pkill -f`**.
- p4256: R1101 MERGE n80 waiter path mismatch → re-arm. Audit wait→lean paths. **Never `pkill -f`**.
- p4255: R340 idle GPUs6,7 → **R1128** TRAIN. Fill idle pairs same pass. **Never `pkill -f`**.
- p4254: **R1108+R1109 REFUTE** → **R1126+R1127** TRAIN r339. **Never `pkill -f`**.
- p4253: **R1106 REFUTE** → **R1125** TRAIN r337. **Never `pkill -f`**.
- p4252: **R1105+R1102 REFUTE** → **R1123+R1124** TRAIN. **Never `pkill -f`**.
- p4251: **R1096+R1097+R1111 REFUTE** → **R1120+R1121+R1122** TRAIN. **Never `pkill -f`**.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


