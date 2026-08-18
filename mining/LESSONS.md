
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
- p3830: **R761+R762 MERGE_DONE** sat idle on brave (no n80 armed) → **host-relay R761→R252** queue after **R769 N80**; R762 next. **Never `pkill -f`**.
- p3829: **R765 REFUTE** m=−0.005388~**−0.72×** → **R779** marsplan Soft HiRank LoBeta SoftCtx Mega (≠ R778 Soft MidRank); R778 TRAIN kept lunar 6,7. **Never `pkill -f`**.
- p3828: **R764 REFUTE** m=+0.000788~**0.12×** → **R778** marsplan Soft MidRank LoBeta SoftCtx Mega; R765 N80 kept lunar 4,5. **Never `pkill -f`**.
- p3827: **R763 REFUTE** m=+0.000158~**0.038×** → **R776**; **R766 REFUTE** m=−0.00211~**−0.30×** → **R777**. **Never `pkill -f`**.
- p3823: Freed **R771** (≈R755) → **R775** tammy Soft MidRank MidBeta SoftCtx Mega; hard-pin BASE after `mine.env`. **Never `pkill -f`**.
- p3822: **R755–R758 REFUTE** → **R771–R774 TRAIN**. **Never `pkill -f`**.
- p3819: **R760 REFUTE** m=−0.004627~**−0.68×** → **R770**; CLI 8×B200 = bl ghost (`bl_skip=1`). **Never `pkill -f`**.
- p3817: **R759 REFUTE** m=−0.001882~**−0.47×** → **R769**. **Never `pkill -f`**.
- p3815: API blacklist must strip `# comment`; restart API by pid only. **Never `pkill -f`**.
- p3814: **brave idle GPUs 4–7** → **R767+R768**. **Never `pkill -f`**.
- p3813: **R337 REAPED** SSH-DEAD → bl `fbb1135f…`. **Never `pkill -f`**.
- p3811: **R752 REFUTE** → **R765**. **Never `pkill -f`**.
- p3809: **R753+R749 REFUTE** → **R763+R764**. **Never `pkill -f`**.
- p3795: **α→TAO→Lium** via `btcli wallet transfer` if `lium fund` fails. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — **host-relay**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.

