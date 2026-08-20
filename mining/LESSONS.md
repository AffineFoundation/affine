
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
- p4165: **R1015 REFUTE** m=−0.002353~−0.94× thought✓167 B✓0.403 (finished 14:35, delayed harvest) → exact-PID reap r924 :8003 → **R1032 ShortCtx MidRank Hiβ Ultra MidLR TRAIN** GPUs1,3 pid**85988** + MERGE→n80; R1008 scoring 918/1300; B300×8=0 BL-only. **Never `pkill -f`**.
- p4164: **R1014 REFUTE** m=−0.005437~−1.08× + **R1019 REFUTE** m=+0.003346~0.95× → **R1030+R1031 Ultra MidLR**; R1021/R1022 mega≠ultra path fix+rearm. **Never `pkill -f`**.
- p4162: **R1012+R1018 REFUTE** → **R1029+R1028 TRAIN**; B300×8=0 BL-only. **Never `pkill -f`**.
- p4161: **R1017 REFUTE** → **R1027 TRAIN** r338 6,7. **Never `pkill -f`**.
- p4160: **R1005 REFUTE** → **R1026 TRAIN** r924 6,7. **Never `pkill -f`**.
- p4159: r926 teacher **TP4→TP2** freed 5,6 → **R1025 TRAIN**. **Never `pkill -f`**.
- p4158: TTL+Soft/Dead +24h all 7 mine-* → **2026-08-21T13:26Z**; sole B200=`8f34559f` BL. **Never `pkill -f`**.
- p4157: α→TAO→Lium r252 147.6α→τ8.2526 → Lium +~$1811; `lium fund` fail→`btcli`→`5FqACMt…zsThe`. **Never `pkill -f`**.
- p4156: **R1004+R1011 REFUTE** → **R1023+R1024 TRAIN**. **Never `pkill -f`**.
- p4155: **R1009+R1010 REFUTE** → **R1021+R1022 TRAIN**. **Never `pkill -f`**.
- p4154: **R998 REFUTE** → **R1020 TRAIN**. **Never `pkill -f`**.
- p4153: **R1006** B✗0.2875 → **R1019 TRAIN**. **Never `pkill -f`**.
- p4152: always arm MERGE→n80 with TRAIN. **Never `pkill -f`**.
- p4150: **R1008 CROWN_OK** → **SUBMITTED** chal-00961. **Never `pkill -f`**.
- p4149: **R1001 REFUTE** → **R1015 MidLR TRAIN**. **Never `pkill -f`**.
- p4147: check executor_blacklist before `lium up` (BL `fbb1135f`/`8f34559f`). **Never `pkill -f`**.
- p4109: `lium fund` Subtensor.transfer fail → `btcli`→`5FqACMt…zsThe`. **Never `pkill -f`**.
- p4101/p4141: R959 **CROWN_OK**→**LOST** chal-00957. **Never `pkill -f`**.

