
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
- p4221: **R1077 REFUTE** m=+0.001670 ~0.32× (thought✓168 B✓0.361) → reap r338 :8002 → **R1088** SoftCtx MidRank Hiβ Hyper MidLR; **R1073 REFUTE** m=−0.009593 ~−0.94× (thought✓188 B✓0.3625) → reap r924 :8002 → **R1089** SoftCtx HiRank Hiβ Hyper MidLR. B300/B200×8=0. **Never `pkill -f`**.
- p4220: **R1077** merge.done but n80 stuck — waiter `EXP=` pointed at wrong Ultra path; lean from Hyper path → n80 LIVE. **R1072 REFUTE** ~−0.05× → **R1087**. B300/B200×8=0. **Never `pkill -f`**.
- p4219: **R1070+R1071 REFUTE** → **R1085+R1086 TRAIN**; R1073 MERGE→n80. B300/B200×8=0. **Never `pkill -f`**.
- p4218: **R1074 REFUTE** ~0.02× → **R1084** Hyper MidLR. B300/B200×8=0. **Never `pkill -f`**.
- p4217: **R1078 REFUTE** ~−0.71× → **R1083** SoftCtx MidRank Midβ Hyper MidLR. **Never `pkill -f`**.
- p4216: **R1079 REFUTE** ~−0.75× → **R1082** ShortCtx HiRank Midβ Hyper MidLR. **Never `pkill -f`**.
- p4215: **R1075 REFUTE** ~0.53× → **R1081** ShortCtx HiRank Midβ Ultra HiLR. **Never `pkill -f`**.
- p4214: **R1051 REFUTE** ~0.27× → **R1080** cryptoDev ShortCtx Hiβ Ultra MidLR. **Never `pkill -f`**.
- p4213: **R1060 REFUTE** ~−1.66×; **R340** TP2 hung → **TP1** teacher GPU0. **Never `pkill -f`**.
- p4209: **CLI rent B200** `mine-r340`@$37.60/h; R1064 **chal-00974**. Host-key churn — `ssh-keygen -R` + accept-new. **Never `pkill -f`**.
- p4208: **R1064 CROWN_OK** ~1.021× → SUBMIT reveal **31485871**. B300×8=0. **Never `pkill -f`**.
- p4207–p4200: α→TAO→Lium + LOST/REFUTE→TRAIN cascade; TTL→**2026-08-21T13:26Z**. **Never `pkill -f`**.


