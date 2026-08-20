
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
- p4201: **R1052 REFUTE** m=−0.001866 ~−0.40× thought✓152 B✓0.397 → exact-PID reap :8003 → **R1071 TRAIN** (HiLR isolate ShortCtx HiRank Hiβ Ultra); R1070 untouched; B300×8=0 B200×8=0. **Never `pkill -f`**.
- p4200: **R1052** chall hung — orphan `VLLM::EngineCore` (parent killed, workers still on GPUs6,7) + missing `__triton_launcher.so`; exact-PID kill EngineCore/workers → FORCE seed from `chall_r1053` (n_so=26) → CHALL_READY + **v4 n80 LIVE** pid**27596**; R1070 untouched. **Never `pkill -f`**.
- p4199: **R1059 REFUTE** m=−0.000889 ~−0.12× → **R1069 TRAIN**; **R1053 REFUTE** → **R1070 TRAIN**; B300×8=0. **Never `pkill -f`**.
- p4198–p4150: REFUTE→TRAIN cascade; TTL→**2026-08-21T13:26Z**; B300 waiters armed. **Never `pkill -f`**.


