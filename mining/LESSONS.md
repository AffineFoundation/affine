
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
- p3779: **R733 REFUTE v4** m=−0.000192~**−0.025×** (thought✓213 B✓0.50 k=3) MidCtx MidRank HiBeta HyperExtra near-parity; reap zesty 6,7 by pidfile → keep `/tmp/r733_merged`; **R739 TRAIN** MidCtx HiRank HiBeta HyperExtra (β=0.3 r=64 @8192 steps=**10800**) zesty 6,7 pid**877334** + wait→merge; **R732 MERGE→CHALL→N80 LIVE** zesty **4,5**/:8002 MidCtx MidRank MidBeta HyperExtra vllm**874651** sim**876822** (`*_reign35_wvk7` fail-closed k=3; Triton seed chall_r722 n_star=26); R715 relay ~18G/66G; B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3778: **R728 REFUTE v4** m=−0.01005~**−1.07×** (thought✓204 B✓0.377 k=3) MidCtx HiRank LoBeta SuperExtra; reap lunar 6,7 by pidfile → keep `/tmp/r728_merged`; **R738 TRAIN** MidCtx HiRank MidBeta SuperExtra (β=0.1 r=64 @8192 steps=**14400**) lunar 6,7 pid**699498** + wait→merge; **R733 MERGE→CHALL→N80 LIVE** zesty **6,7**/:8003 MidCtx MidRank HiBeta HyperExtra vllm**871100** sim**873760** (`*_reign35_wvk7` fail-closed k=3; Triton seed chall_r723 n_star=26); R715 relay ~14G/66G; B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3777: **R728 MERGE→CHALL→N80 LIVE** lunar **6,7**/:8003 marsplan MidCtx HiRank LoBeta SuperExtra vs reign35 (16sh/66G local) vllm**696415** sim**698423** (`*_reign35_wvk7` fail-closed k=3; Triton seed chall_r721 n_star=30); reaped orphan **R736** wait pid**123943** by pidfile (train freed p3776, adapter_ok=0 loop); R715 relay ~4G/66G; B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** @pynccl ~1GB (eager / GPUs6,7 / local snapshot / `NCCL_P2P_DISABLE` / `--disable-custom-all-reduce` all fail — same as p3244/p3431); abort local TK by pidfile; **do not cold-TK brave**; freed early **R736** crown 4,5 → **R715 host-relay** brave→crown tar~66G (wait**126681** → lean n80 vs reign35); R716 MERGE stays on brave; R737 TRAIN kept 6,7; B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3775: **R727 REFUTE v4** m=−0.007815~**−0.90×** (thought✓136 B✓0.454 k=3) MidCtx HiRank LoBeta SuperExtra; reap crown 6,7 by pidfile → keep `/tmp/r727_merged`; **R737 TRAIN** MidCtx HiRank MidBeta HyperExtra (β=0.1 r=64 @8192 steps=**10800**) crown 6,7 pid**125075** + wait→merge; R736 still TRAIN 4,5; brave teacher arming (prefetch done); B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3774: **R726 REFUTE v4** m=−0.002183~**−0.48×** (thought✓149.5 B✓0.412 k=3) MidCtx MidRank LoBeta SuperExtra; reap crown 4,5 by pidfile → keep `/tmp/r726_merged`; **R736 TRAIN** MidCtx MidRank LoBeta HyperExtra (β=0.02 r=32 @8192 steps=**10800**) crown 4,5 pid**123861** + wait→merge; R727 N80 ~80/80 on 6,7; brave prefetch~96%; B300×8=0; burn ~$331/h. **Never `pkill -f`**.
- p3771: **R722–R725 REFUTE** + **R732–R735 TRAIN** zesty+R252 (HyperExtra amplify); B300×8=0; burn ~$331/h. **Never `pkill -f`**. Detail → `archive/lessons_ops_pre_p3770.md` / prior lines.
- p3763: **α→TAO→Lium** r252 **295.20α → τ16.73** then transfer→Lium ck; bal **$85135→$88582**; free τ**1260.38**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8…`; next n80 **must** pin tammy. **Never `pkill -f`**.

