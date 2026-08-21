
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
- p4251: **R1096+R1097 REFUTE** (m=−0.000713 ~−0.19× / m=−0.002077 ~−0.45× thought✓ B✓ k=3) + **R1111 REFUTE** m=+0.003700 ~0.68× → exact-PID reap r340 :8002/:8003 + r338 :8003 → **R1120+R1121** Soft/MidCtx MidLoβ Hyper HiLR TRAIN + **R1122** MidCtx HiRank Loβ Hyper HiLR TRAIN. B300/B200×8=0. **Never `pkill -f`**.
- p4250: **R1107 REFUTE** m=+0.004879 ~0.66× (thought✓181 B✓0.456 k=3) vs reign36 → exact-PID reap r337 :8003 → **R1119** ShortCtx MidRank Midβ Hyper HiLR TRAIN pid**149858** (SoftCtx R1106 still on 6,7). B300/B200×8=0. **Never `pkill -f`**.
- p4249: r340 exact-PID teacher **8192→65536** TP1 util**0.72** (KV concurrency ~2.06×) → **R1096** SoftCtx chall :8002 util**0.55** (0.72 OOM'd) + dual n80 armed; R1097 :8003 kept. B300/B200×8=0. **Never `pkill -f`**.
- p4248: **R1099 REFUTE** m=−0.001230 ~−0.40× (thought✓179.5 B✓0.438 k=3) vs reign36 → exact-PID reap r938 :8002 → **R1118** SoftCtx HiRank Midβ Hyper HiLR TRAIN pid**55117**. r340 **R1096** chall OOM @util0.72/TP1 max_len65k; **R1097** n80 died teacher `max_model_len=8192` (need 65536) — :8003 still up. B300/B200×8=0. **Never `pkill -f`**.
- p4247: r340 **R1096/R1097** dual TP2 hung @ pynccl — seed wrong Triton path; **TP1** + NCCL_P2P/IB_DISABLE → CHALL_READY; serialize R1097 after `:8002`. **Never `pkill -f`**.
- p4246: **R1103+R1104 REFUTE** → reap crown → **R1116+R1117** SoftCtx Hyper HiLR TRAIN. B300/B200×8=0. **Never `pkill -f`**.
- p4245: r340 lean stubs → real chall+v4 n80 :8002/:8003. Check lean is not a stub. **Never `pkill -f`**.
- p4244: **R1098 REFUTE** → **R1115** TRAIN r926. **Never `pkill -f`**.
- p4243: **R1089 REFUTE** → **R1114** TRAIN r924. **Never `pkill -f`**.
- p4242: **R1084 REFUTE** → **R1113** TRAIN r924. **Never `pkill -f`**.
- p4241: **R1095 REFUTE** → **R1112** TRAIN r924. **Never `pkill -f`**.
- p4240: **R1100 REFUTE** → **R1111** TRAIN r338. **Never `pkill -f`**.
- p4239: **R1090 REFUTE** → **R1110** TRAIN r252. **Never `pkill -f`**.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


