
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
- p4015: train→merge waiters alone leave **MERGE SIZE_OK idle** (R901/R902 crown; also lunar R896/R897 + R898) — always ship **merge→n80** waiter or launch lean_chall same pass. Crown R901+R902 n80 LOADING :8003/:8002. **Never `pkill -f`**.
- p4014: **R899 REFUTE** causality_fail B=0.2375 m=−0.000597; **R900 REFUTE** m=−0.004297~−0.49× → exact-PID reap → **R910 ShortCtx + R911 HiRank SoftCtx TRAIN**. Tore SSH-dead R337+R338. Burn **~$306.66/h**. **Never `pkill -f`**.
- p4013: R900 same bad-LAUNCHED pattern — waiter path `…midrank-hibeta-midctx…` (real `…midrank-midlobeta-softctx…`); merge SIZE_OK idle on GPUs6,7. Cleared stamp → launched real lean_chall :8003. R337+R338 both TCP-dead (86.38.182.x). **Never `pkill -f`**.
- p4012: R899 MERGE sat idle — waiter stamped `LAUNCHED` then `bash` failed on typo path `…midrank-lobeta…` (real dir `…midrank-midlobeta…`). Clear bad LAUNCHED, launch real `lean_chall`; stamp LAUNCHED only after `test -x "$CHALL"`. **R882 REFUTE** m=−0.007812~−1.03×. Never kill `VLLM::EngineCore` by broad argv match (swept teacher/king). R338 SSH flapped after cleanup. **Never `pkill -f`**.
- p4010: R338 HF king DL stalled on last ~50G shard — **stop by pid**, rsync blob from crown → R338 @~287MB/s → KING_READY. Prefer peer rsync over public HF. **Never `pkill -f`**.
- p4009–p3900: REFUTE→TRAIN; MERGE idle→n80; GRPO; R861 LOST; Alpha→TAO→Lium; never `pkill -f`.
