
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
- p3856: **R768 meta accel** while weight parallel×4 — `config.json`+small meta on R252 early (stamp-ready once shards≥16); leave weight pipes alone. **Never `pkill -f`**.
- p3855: **R786 REFUTE** ~0.13× + **R788 REFUTE** ~−0.46× (thought/B✓) → free crown 4–7 → **R798+R799** MidCtx UltraLoLR. Soft UltraLoLR near-miss still under bar. **Never `pkill -f`**.
- p3854: **R768 solo tar stall** ~8G/66G → kill-by-PID → **parallel×4** size-checked host pipes (KEEP 5+8); R767 DEC already on disk. **Never `pkill -f`**.
- p3853: **R787 REFUTE** ~−0.80× (thought✓212 B✓0.453) → free 6,7 → **R797** MidCtx Mid Hi UltraLoLR; **R767 REFUTE** ~0.47× (m=+0.00222 under 2·SE) → R768 SCP waiter. **Never `pkill -f`**.
- p3852: **R767 SCP→N80 LIVE** — meta-only host pipes while weight parallel×4 finishes unblocks `config.json` early; stamp→lean :8002. **Never `pkill -f`**.
- p3851: **R785 REFUTE** ~−0.17× → **R796** MidCtx Mid Mid UltraLoLR. **Never `pkill -f`**.
- p3850: brave GPUs **0–3 idle** → **R794+R795** UltraLoLR. **Never `pkill -f`**.
- p3849: Mega axes need `epochs ≥ ceil(max_steps/n_rows)` or drop epoch cap. **Never `pkill -f`**.
- p3848: **R780+R781 MERGE_DONE** → host-relay after R768/R780. **Never `pkill -f`**.
- p3847: **R762 REFUTE** → **R767** parallel×4. **Never `pkill -f`**.
- p3846: **R782 REFUTE** → **R793**; R252 prefer SSH. **Never `pkill -f`**.
- p3845: **R779/R773/R774 REFUTE** → **R790/R791/R792**. **Never `pkill -f`**.
- p3844: dual challs need distinct ports; **R778→R789**. **Never `pkill -f`**.
- p3843: lone 8×B200=`fbb1135f` **bl**. **Never `pkill -f`**.
- p3842: miss `model-visual-restored` → vLLM ValueError → graft. **Never `pkill -f`**.
- p3841/p3840/p3839: UltraLoLR cascade after Soft Mega REFUTEs. **Never `pkill -f`**.
- p3838: tar stall → **parallel×4 size-checked host pipes**. **Never `pkill -f`**.
- p3837/p3836: R768/R767 host-relay. **Never `pkill -f`**.
- p3835: R252 :40299 **sshfail** → hardened resume. **Never `pkill -f`**.
- p3833/p3832: **R770/R769 REFUTE** → **R782/R780**. **Never `pkill -f`**.
- p3831/p3795: **α→TAO→Lium** via `btcli wallet transfer` if `lium fund` fails. **Never `pkill -f`**.
- p3829–p3822: Soft/MidCtx Mega REFUTE cascade → UltraLoLR. **Never `pkill -f`**.
- p3819: CLI 8×B200 = bl ghost. **Never `pkill -f`**.
- p3815: API blacklist must strip `# comment`. **Never `pkill -f`**.
- p3813: **R337 REAPED** → bl `fbb1135f…`. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — **host-relay**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.


