
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
- p3814: **brave idle GPUs 4–7** → same-pass **R767 Soft MidRank Loβ SoftCtx Mega** 4,5 + **R768 Soft HiRank Loβ SoftCtx Mega** 6,7 (wait→merge only; host-relay n80); B300×8 stock still 0. **Never `pkill -f`**.
- p3813: **R337 REAPED** `golden-lion-72` SSH-DEAD (mapped :20296 refused; host :22 pubkey-denied; `lium reboot` fail) → `lium rm` mine-* only; bl executor `fbb1135f…`; clear rented/bootstrapped stamps; waiters **6/25** HEAD=R337 requeue. **Never `pkill -f`**.
- p3812: **R754 REFUTE v4** m=−0.008616~**−0.91×** (thought✓238 B✓0.50 k=3) → reap zesty 6,7 → **R766** MidRank HiBeta MidCtx Mega; R337 `golden-lion-72` SSH :20296 **refused** (pod RUNNING). **Never `pkill -f`**.

- p3811: **R752 REFUTE v4** m=+0.000418~**0.07×** (thought✓227 B✓0.453 k=3) → reap lunar 4,5 → **R765** MidRank LoBeta MidCtx Mega (amplify R721 ~−0.05×); R337 cache copy still ~4.6/66G. **Never `pkill -f`**.
- p3810: **R337** `golden-lion-72` 8×B200 claimed by fleet waiter; marsplan HF **gated on pod IP** (host token OK) — copy cache from live pod then skip `snapshot_download`; retarget KING→reign35. **Never `pkill -f`**.
- p3809: **R753 REFUTE v4 near-miss** m=+0.000428~**0.14×** (thought✓156.5 B✓0.538 k=3) + **R749 REFUTE v4** m=−0.00405~**−0.54×** (thought✓207 B✓0.392 k=3) → dual reap → **R763** Soft HiRank MidBeta SoftCtx Mega crown 6,7 + **R764** marsplan MidRank MidBeta MidCtx Mega lunar 6,7 (both wait→merge+n80). **Never `pkill -f`**.
- p3808: **brave idle 8×GPU** → same-pass **R761 Soft Midβ SoftCtx Mega** 0,1 + **R762 Soft Hiβ SoftCtx Mega** 2,3 (wait→merge only; host-relay n80). **Never `pkill -f`**.
- p3807: **R751 REFUTE v4** m=−0.00119~**−0.25×** → **R760** Short MidRank MidBeta Mega R252 6,7. **Never `pkill -f`**.
- p3806: **R749+R752+R753** wait→n80 ARMED same-pass. **Never `pkill -f`**.
- p3805: **R750 REFUTE** near-parity → **R759** Short HiRank HiBeta Mega; R751 MERGE idle→n80. **Never `pkill -f`**.
- p3804: **R748+R716 REFUTE** → **R757+R758 TRAIN**. **Never `pkill -f`**.
- p3798: merge script stops at MERGE_DONE — **always arm wait→n80**. **Never `pkill -f`**.
- p3795: **α→TAO→Lium** r252 **89.96α → τ5.1067**; `lium fund` fail → **`btcli wallet transfer`** to Lium ck `5FqAC…zsThe`. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — do not cold-TK brave; **host-relay**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.

