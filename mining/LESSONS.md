
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
- p4004: **R886 REFUTE v4** ~0.27× + **R885 REFUTE v4** ~−0.05× (vera SoftCtx β-isolates) → reap exact chall PIDs → **R901** MidCtx MidLoβ TRAIN crown 6,7 + **R902** MidCtx Loβ TRAIN crown 4,5; **R887 MERGE idle** brave GPU2 → lean_chall+v4 n80 :8004 TP1 (`p4004_*_armed.done`); B300/8×B200 stock=0; burn **~$405.70/h**. **Never `pkill -f`**.
- p4003: **R877+R878 MERGE_DONE** golden + **R889+R890 MERGE_DONE** brave sat idle (no lean_chall scripts) → ship p4003 lean_chall+v4 n80 (`p4003_r877_r878_n80_armed.done`, `p4003_r889_r890_n80_armed.done`); brave TP1 GPU4/6; R887 still MERGE idle GPUs 2,3; R885/R886 n80 mid (~22–23/80); B300/8×B200 stock=0; burn **~$405.70/h**. **Never `pkill -f`**.
- p4002: **R885+R886 MERGE_DONE** sat idle on crown — wait→n80 CHALL paths wrong → fix + dual lean chall+v4 n80 (`p4002_r885_r886_n80_armed.done`). **Never `pkill -f`**.
- p4001: **R892 REFUTE v4** ~−1.46× + **R893 REFUTE v4** ~−1.07× → **R899+R900 TRAIN** R252 (`p4001_r899_r900_armed.done`). **Never `pkill -f`**.
- p4000: R888 idle GPUs **5,6** → **R898** vera ShortCtx TRAIN (`p4000_r898_armed.done`). **Never `pkill -f`**.
- p3999–p3993: TRAIN_DONE→MERGE+n80; REFUTE→reap→TRAIN swarm; Triton seed; `*_armed.done`. **Never `pkill -f`**.
- p3992–p3900: SIZE_OK→n80; GRPO; DeepGEMM offs; R861 LOST; merge→n80 waiters. **Never `pkill -f`**.
- p3899–p3880: merge idle traps; host-relay; Soft-dead SSH→`lium rm` mine-* only; Marsplan gated; Alpha→TAO→Lium. **Never `pkill -f`**.
- p3879–p3875: SIZE_OK stamps; SIGSTOP/tail relay; byte-match before `*_scp_ready.done`. **Never `pkill -f`**.
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
