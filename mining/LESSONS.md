
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
- p3878: **R800 MERGE_DONE** (16+vis) while R783 mid-relay → **defer** host-relay until `r783_scp_ready.done` + no `.tmp` (no dual-pipe), then SIZE_OK stamp → n80 crown **4,5/:8002**; leave R783 **6,7**. **Never `pkill -f`**.
- p3877: **R795 REFUTE v4** m=+0.000224~**0.045×** (thought✓153.5 B✓0.4125 k=3) Soft Hi Mid Soft UltraLoLR near-parity → free chall by exact PID → **R808** Soft Hi Hi Soft UltraLoLR marsplan lunar 6,7 (R794 Midβ Soft near-miss → HiBeta Soft); leave R807 TRAIN 4,5 / R783 relay. **Never `pkill -f`**.
- p3876: **R802 REFUTE** ~−0.47× + **R794** ~0.61× near-miss + **R784** ~0.45× same pass → free challs by exact PID → **R806** Soft Mid Hi Soft UltraLoLR R252 6,7 + **R807** MidCtx Hi Mid Soft UltraLoLR lunar 4,5 (R794 MidCtx transfer); leave R795 n80 / R805 TRAIN / R783 relay. **Never `pkill -f`**.
- p3875: R784 finish = wait finals (no `.tmp`) then **byte-match all 16+vis vs brave** before `r784_scp_ready.done`; waiter arms lean :8002; R783×4 relay only after R784 stamped. **Never `pkill -f`**.
- p3874: R784 mid-pipe accel = **SIGSTOP** p3871 parent (exact PID) → tail+meta free shards (busy-skip `.tmp`) → **kill −9** STOP'd parent after SIZE_OK (never CONT) → arm **R783-only** parallel (parent dies before `relay_hypo r783`). **Never `pkill -f`**.
- p3873: p3871 stamp wrote `/root/logs/r${hypo}_…` with hypo=`r784` → **`rr784_scp_ready.done`** while waiters poll **`r784_…`** — size-verify fixer must write `${hypo}_scp_ready.done` (no extra `r`). **Never `pkill -f`**.
- p3872: lunar idle REFUTE challs (R790/R789) still hold GPUs — kill by exact PID then **host-relay MERGE_DONE** brave→lunar **×2** (leave R784×4 bandwidth); arm wait→n80 before pipes. **Never `pkill -f`**.
- p3871: sequential `tar cf|xf` brave→crown for dual 66G merges is too slow (~1 shard/5min) — kill tar by exact PID, keep size-matched shards, **parallel×4** size-checked SSH pipes; stamp only after all 16+vis match. **Never `pkill -f`**.
- p3870: brave `/tmp` **ENOSPC** killed R794/R795 merge mid-shard — keep only needed merges (r783/r784) then rematch; **R798 REFUTE** ~−0.33× + **R799 REFUTE** ~−0.54× → dual host-relay **R784+R783** crown; also **R790/R789 REFUTE** lunar idle. **Never `pkill -f`**.
- p3869: **R792 REFUTE** ~−0.38× + **R781 REFUTE** ~−0.14× same pass → free challs by exact PID → **R804** MidCtx Hi Mid UltraLoLR golden 4,5 + **R805** Soft Hi Hi Soft UltraLoLR R252 4,5; fix chall `GPUS=` pin after rename (6,7→4,5). **Never `pkill -f`**.
- p3868: **R781** SIZE_OK all 17 vs brave → stamp; STOP parent ignores SIGTERM while state=T — **kill −9 by exact PID** (never CONT); lean chall :8002 loading after stamp. **Never `pkill -f`**.
- p3867: **R781** finish = size-verify all 16+vis vs brave **before** stamp; mid/tail count-only stamps unsafe (p3863); kill STOP parent then stamp. **Never `pkill -f`**.
- p3866: **R781** mid accel (11–12) after **SIGSTOP** p3848 parent — busy-skip live `.tmp`; kill STOP parent after size-ok (**never CONT**). **Never `pkill -f`**.
- p3865: **R781** SCP accel = main×4 (05–12) + **tail×3** (13–16+vis) + **meta** (config/tok); partition shards to avoid dual-write; `lium ls --format json` B200×8=0 while table may show bl ghost. **Never `pkill -f`**.
- p3864: **R780 REFUTE v4** m=+0.001591~**0.17×** (thought✓177 B✓0.407) Soft MidRank HiBeta SoftCtx Mega LoLR — positive but far below max(2·SE,δ); free :8002 by exact chall PID; lone API 8×B200 still `fbb1135f` **bl**. **Never `pkill -f`**.
- p3863: **R780** `stamp_check` count-only SCP_READY while shard **08** truncated → vLLM load dies @~41%; size-verify all shards before lean; kill STOP'd parent by PID (not CONT) to unblock R781 `pgrep -f` wait. **Never `pkill -f`**.
- p3862: **R780** dual-write on shard **13** (p3848+tail) corrupts `.tmp`; EOF-kill → partial over good final — size-checked `.fixing`; SIGSTOP before freeing slots. **Never `pkill -f`**.
- p3861: **R780** pipe stall on shard **11** → kill ssh PIDs only; **tail accel** 13–16+vis. **Never `pkill -f`**.
- p3860: **R791/R793 REFUTE** → **R803+R802** UltraLoLR; leave :8002 for R780. **Never `pkill -f`**.
- p3859: brave idle **4–7** → **R800+R801**. **Never `pkill -f`**.
- p3858: **R768 REFUTE** → **R780 meta accel**. **Never `pkill -f`**.
- p3857–p3850: SCP→n80 / UltraLoLR cascade / brave fill. **Never `pkill -f`**.
- p3849: Mega axes need `epochs ≥ ceil(max_steps/n_rows)`. **Never `pkill -f`**.
- p3848: **R780+R781 MERGE_DONE** → host-relay. **Never `pkill -f`**.
- p3844: dual challs need distinct ports. **Never `pkill -f`**.
- p3843/p3813: lone 8×B200=`fbb1135f` **bl**. **Never `pkill -f`**.
- p3842: miss `model-visual-restored` → graft. **Never `pkill -f`**.
- p3835: R252 :40299 **sshfail** → `lium exec`. **Never `pkill -f`**.
- p3831/p3795: **α→TAO→Lium** via `btcli wallet transfer` if `lium fund` fails. **Never `pkill -f`**.
- p3815: API blacklist must strip `# comment`. **Never `pkill -f`**.
- p3776: **brave TP=2 teacher NCCL hang** — **host-relay**. **Never `pkill -f`**.
- p3762: **king flip reign34→reign35** `tammyfritz/…tammy2`@`7e5fd5f8…`. **Never `pkill -f`**.


