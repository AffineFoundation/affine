
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
- p3806: **R749+R752+R753** launched with wait→merge only — **armed wait→n80 same-pass** (lunar 6,7/4,5 + crown 6,7) before MERGE_DONE; R751 n80 ~59/80 kept. **Never `pkill -f`**.
- p3805: **R750 REFUTE v4 near-parity** m=−0.000909~**−0.16×** (thought✓153 B✓0.377 k=3) → reap → **R759** Short HiRank HiBeta MegaSuperExtra ep4 R252 4,5 (wait→merge+n80); **R751 MERGE sat idle** (no wait→n80 at launch) → same-pass lean chall/:8002 GPUs 6,7. **Never `pkill -f`**.
- p3804: **R748 REFUTE v4** m=−0.01885~**−1.20×** (thought✓226 B✓0.412 k=3) SoftCtx MidBeta SuperExtra flop + **R716 REFUTE v4 near-parity** m=−0.000989~**−0.13×** (thought✓142 B✓0.456 k=3) → dual reap → **R757** SoftCtx MidBeta MegaSuperExtra ep4 zesty 4,5 + **R758** SoftCtx HiBeta MegaSuperExtra ep4 crown 4,5 (both wait→merge+n80); R750 N80 ~75/80 kept. **Never `pkill -f`**.
- p3803: **R750 MERGE sat idle ~2m** (no wait→n80) → same-pass lean chall/:8002 + v4 n80 R252 4,5. **Never `pkill -f`**.
- p3802: **R748 MERGE sat idle ~9m** (no wait→n80) → same-pass lean chall/:8002 + v4 n80 zesty 4,5. **Never `pkill -f`**.
- p3801: **R746 REFUTE v4 near-parity** m=−0.00049~**−0.14×** → **R756 TRAIN** MegaSuperExtra ep4 golden 6,7. **Never `pkill -f`**.
- p3800: **R745 REFUTE v4** m=−0.00368~**−0.76×** → **R755 TRAIN** MegaSuperExtra ep4 golden 4,5. **Never `pkill -f`**.
- p3799: **R747 REFUTE v4** m=−0.00346~**−0.37×** → **R754 TRAIN** MegaSuperExtra ep4 zesty 6,7 (hard-pin BASE after mine.env). **Never `pkill -f`**.
- p3798: merge script stops at MERGE_DONE — **always arm wait→n80**. **Never `pkill -f`**.
- p3796: **R744 REFUTE v4** m=−0.001936~**−0.43×** → **R753 TRAIN** MegaSuperExtra ep4 crown 6,7. **Never `pkill -f`**.
- p3795: **α→TAO→Lium** r252 **89.96α → τ5.1067**; `lium fund` fail → **`btcli wallet transfer`** to Lium ck `5FqAC…zsThe`. **Never `pkill -f`**.
- p3794: **R741 REFUTE** → **R752 TRAIN**; MERGE idle → same-pass lean n80. **Never `pkill -f`**.
- p3793: **R742+R743 REFUTE** → **R750+R751 TRAIN** on R252. **Never `pkill -f`**.
- p3792: **R738 REFUTE** near-parity → **R749 TRAIN** MegaSuperExtra ep4. **Never `pkill -f`**.
- p3789: **R740 REFUTE** → **R748 TRAIN** SoftCtx SuperExtra. **Never `pkill -f`**.
- p3787: **R715 REFUTE** → **R716 host-relay** brave→crown. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — do not cold-TK brave; **host-relay**. **Never `pkill -f`**.
- p3763: **α→TAO→Lium** r252 **295.20α → τ16.73**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.

