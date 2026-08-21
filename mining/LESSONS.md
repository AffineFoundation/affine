
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
- p4243: **R1089 REFUTE** m=−0.001673 ~−0.32× (thought✓181 B✓0.519 k=3) vs reign36 → exact-PID reap r924 :8002 → **R1114** SoftCtx HiRank Hiβ Hyper HiLR TRAIN pid**138299** (R1113 TRAIN GPUs1,3 + R1112 TRAIN GPUs4,5 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4242: **R1084 REFUTE** m=−0.003045 ~−0.99× (thought✓193 B✓0.400 k=3) vs reign36 → exact-PID reap r924 :8003 → **R1113** ShortCtx LoRank Hiβ Hyper HiLR TRAIN pid**137213** (R1089 n80 :8002 + R1112 TRAIN GPUs4,5 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4241: **R1095 REFUTE** m=−0.001309 ~−0.53× (thought✓165 B✓0.526 k=3) vs reign36 → exact-PID reap r924 :8004 → **R1112** MidCtx MidRank MidLoβ Hyper HiLR TRAIN pid**136444** (R1089 n80 :8002 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4240: **R1100 REFUTE** m=−0.002847 ~−0.26× (thought✓196 B✓0.462 k=3) vs reign36 → exact-PID reap r338 :8003 → **R1111** MidCtx HiRank Hiβ Hyper HiLR TRAIN pid**182821** (R1102 TRAIN GPUs6,7 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4239: **R1090 REFUTE** m=−0.001520 ~−0.56× (thought✓187 B✓0.466 k=3) vs reign36 → exact-PID reap r252 :8002 → **R1110** SoftCtx HiRank MidLoβ Hyper HiLR TRAIN pid**180255** (R1105 TRAIN GPUs6,7 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4238: **R1085 REFUTE** m=−0.002495 ~−0.82× (thought✓171 B✓0.496) + **R1086 REFUTE** m=+0.000154 ~0.05× (thought✓171 B✓0.353) vs reign36 → exact-PID reap r339 :8002/:8003 → **R1108+R1109** Hyper HiLR TRAIN pid**43401**/**43394**. B300/B200×8=0. **Never `pkill -f`**.
- p4237: **R1094 REFUTE** m=−0.001575 ~−0.55× (thought✓165 B✓0.423 k=3) vs reign36 → exact-PID reap r337 :8003 → **R1107** MidCtx MidRank Midβ Hyper HiLR TRAIN pid**144238** (R1106 TRAIN GPUs6,7 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4236: **R1083 REFUTE** m=+0.001741 ~0.53× (thought✓168 B✓0.369 k=3) vs reign36 → exact-PID reap r337 :8002 → **R1106** SoftCtx MidRank Midβ Hyper HiLR TRAIN pid**143414** (R1094 n80 :8003 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4235: **R1081 REFUTE** m=+0.001560 ~0.40× (thought✓193 B✓0.597 k=3) vs reign36 → exact-PID reap r252 :8003 → **R1105** ShortCtx HiRank Midβ Hyper HiLR TRAIN pid**174547** (R1090 intact). B300/B200×8=0. **Never `pkill -f`**.
- p4228: paygo **r252 88.56α≈τ5.06** ≥τ5 bar → `btcli stake remove --amount-alpha all` OK; `lium fund` fails → **`btcli wallet transfer`** to Lium coldkey → Lium **~$77760**. **Never `pkill -f`**.


