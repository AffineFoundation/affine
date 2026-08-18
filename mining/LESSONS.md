
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
- p3800: **R745 REFUTE v4** m=−0.00368~**−0.76×** (thought✓157 B✓0.449 k=3) → reap → **R755 TRAIN** MegaSuperExtra ep4 golden 4,5 (wait→merge+n80 armed); R746 N80 ~77/80 kept 6,7. **Never `pkill -f`**.
- p3799: **R747 REFUTE v4** m=−0.00346~**−0.37×** (thought✓243 B✓0.385 k=3) → reap → **R754 TRAIN** MegaSuperExtra ep4 zesty 6,7 (hard-pin BASE after mine.env — mine.env BASE=r252 breaks marsplan axes); **R746 MERGE→n80 ARMED→LIVE** golden 6,7. **Never `pkill -f`**.
- p3798: **R745 TRAIN→MERGE** golden 4,5 + **wait→n80 ARMED** (lean_outer dead; merge script stops at MERGE_DONE — always arm chall n80); R747 N80 sim**895303**; B300×8=0. **Never `pkill -f`**.
- p3797: **R747 MERGE→N80 LIVE** zesty 6,7 :8003 (train.done→merge 16sh→chall**892881**); leave R748 on 4,5; R716 ~11/16; B300×8=0. **Never `pkill -f`**.
- p3796: **R744 REFUTE v4** m=−0.001936~**−0.43×** (thought✓135 B✓0.363 k=3) MidCtx HiRank MidBeta SuperExtra → reap chall → **R753 TRAIN** MegaSuperExtra ep4 crown 6,7; R716 relay ~11/16. **Never `pkill -f`**.
- p3795: **α→TAO→Lium** r252 **89.96α → τ5.1067** (stake crossed ~τ5); `lium fund` `Subtensor.transfer` miss → **`btcli wallet transfer`** to Lium ck `5FqAC…zsThe`; bal **$86668→$87698**; free τ**1260.38**. **Never `pkill -f`**.
- p3794: **R741 REFUTE v4** m=−0.00452~**−0.53×** (thought✓205 B✓0.534 k=3) MidCtx HiRank LoBeta HyperExtra → reap chall → **R752 TRAIN** MegaSuperExtra ep4 lunar 4,5; **R744 MERGE sat idle ~20m** → same-pass lean chall/:8003 + v4 n80 crown 6,7. **Never `pkill -f`**.
- p3793: **R742+R743 REFUTE v4** (−0.47/−0.27×; thought+B✓ k=3) Short HiRank HyperExtra flop → reap both challs → **R750** HiBeta Short SuperExtra + **R751** MidBeta Short MegaSuperExtra ep4 on R252. **Never `pkill -f`**.
- p3792: **R738 REFUTE v4** m=−0.000813~**−0.11×** (thought✓215 B✓0.464 k=3) MidCtx HiRank MidBeta SuperExtra near-parity → **R749 TRAIN** MegaSuperExtra ep4 lunar 6,7. **Never `pkill -f`**.
- p3789: **R740 REFUTE v4** ~−0.48× → **R748 TRAIN** SoftCtx SuperExtra zesty 4,5. **Never `pkill -f`**.
- p3788: **R739 REFUTE v4** ~−1.01× → **R747 TRAIN** SuperExtra zesty 6,7. **Never `pkill -f`**.
- p3787: **R715 REFUTE v4** ~−0.42× → **R716 host-relay** brave→crown. **Never `pkill -f`**.
- p3784: **R734/R735/R737/R729/R730 REFUTE v4** → **R742–R746 TRAIN**; B300×8=0. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — do not cold-TK brave; **host-relay**. **Never `pkill -f`**.
- p3763: **α→TAO→Lium** r252 **295.20α → τ16.73**; bal **$85135→$88582**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.

