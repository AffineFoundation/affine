
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
- p4041: R338 n80 404 `model vera6/… does not exist` when king served from **local path** — pass `--king-repo`=`/v1/models` id (not hub string); fill crown idle GPUs **1,3** → **R936** SoftCtx HiRank Loβ; stock still BL `8f34559f`/`fbb1135f`. **Never `pkill -f`**.
- p4040: B300×8=0 + sole 8×B200 BL `8f34559f` → fill R338 idle GPUs **6,7** with **R935** HiRank Loβ MidCtx while chall:8002 loads; `lium scp` flaky → plain `scp -P`. **Never `pkill -f`** (matches SSH cmdline).
- p4039: R338 online-DPO TRAIN_DONE (189 steps) but post_train merge aborted — `mine.env`/pipe still pinned **marsplan queen** path after p4027 vera pivot; fix=merge `--base` vera `8e3f1695` + pin mine.env + king:8001 + chall:8002 + v4 n80. **Never `pkill -f`**.
- p4038: B300×8=0 + sole 8×B200 still BL `8f34559f` → fill R926 idle H100 GPUs **4–7** with **R933** ShortCtx MidLoβ + **R934** MidCtx Loβ; `lium scp TARGET local remote`. **Never `pkill -f`**.
- p4037: R923 **REFUTE** ~0.56× (m=+0.001172 SE=0.001042 thought✓175 B✓0.315) → exact-PID reap chall :8002 → **R932** HiRank Hiβ ShortCtx; stock BL-only → fill R924 idle GPUs **4–7** with **R930/R931**. **Never `pkill -f`**.
- p4036: `lium fund` fails (`Subtensor` has no `transfer`) — fallback `btcli wallet transfer` τ→ Lium ck `5FqACMt…zsThe`; r252 236α/~τ13→τ12.5→Lium (+~$2.6k). R923 chall READY→n80 RUNNING. Stock BL-only. **Never `pkill -f`**.
- p4035: R913 **REFUTE** ~−0.17× → R929 HiRank ShortCtx MidLoβ; R923 TRAIN_DONE but merge failed `--adapter …/train` (peft wants `…/train/adapter`) — relaunch merge+n80 arm. Stock only BL `fbb1135f`. **Never `pkill -f`**.
- p4034: cold crown n80 died — missing `s4-h2-merge/run_sim_duel.py` + no pyarrow; fix=upload sim + `ensurepip`+pyarrow + corpus sync. R912 **REFUTE** ~0.53× → R928 HiRank Midβ MidCtx. Chall relaunch needs CUDA_HOME=cu13 (lean_chall has it). **Never `pkill -f`**.
- p4033: crown cold TK finished → teacher :8000 + king :8001 READY; R912/R913 MERGE_DONE → dual n80 chall launched (:8003/:8002). Waiter `king id=?` is expected until king READY — next 30s poll fires. Stock still BL-only 8×B200. **Never `pkill -f`**.
- p4032: R927 premature TRAIN failed `ModuleNotFoundError: evalsrv` + partial shards; harden arm to require **16** `model-*-of-*.safetensors` **and** live `from evalsrv.chat import THINK_OPEN` before lean; crown cold TK (teacher DL GPU0 + king GPU2) + arm MERGE→n80 while R912/R913 merge. **Never `pkill -f`**.
- p4031: sole 8×B200 is **BL** `fbb1135f` → skip; fill idle R926 H100 GPUs **2,3** with R927 MidCtx MidLoβ; stub empty `affine_pkg` → upload fleet-v4-sync; arm waits `cryptodev_dl.done`+weight shards (not bare `config.json`). **Never `pkill -f`**.
- p4030: B300/B200×8 empty → rent **8×H100** `$13.76` for R926 SoftCtx MidLoβ restart; fill idle H200 GPUs **2,3** with R925 HiRank MidCtx MidLoβ (R901 0.64×). `lium scp` rejects dirs — use `lium rsync` or tar+scp. **Never `pkill -f`**.
- p4028: B300×8=0 + only bl `8f34559f` B200 → rented **8×H200** `mine-r924` MidCtx Hiβ ($33.81) to cut burn gap; replace when B300 appears. **Never `pkill -f`**.
- p4027: **marsplan0624/…queen gated** (403 even with `canReadGatedRepos`) — old rev `556d02a2` 404; pivot online-DPO to **vera** `8e3f1695`. **R914 REFUTE** ~0.14× → **R923** ShortCtx Hiβ. **Never `pkill -f`**.
- p4026: **`mine-crown-1` has no wait_bootstrap case** → manual cold pack (`p4026_crown_cold_r912_r913.sh`) + Soft Mid Mid Soft from local r886; API stock `bl_skip=1` only. **Never `pkill -f`**.
- p4025: **stale `.bootstrapped` after re-rent** skipped R252/R337/R338; `mine-crown-1` falls to `needs_axis_uploader` (no case) → clear markers + manual upload; **R914 n80 relaunched** on R888. **Never `pkill -f`**.
- p4024: **TTL collapse** (~17:23Z) killed 5 mine-* → rented **8×B300 crown** + **3×8×B200**; tore bl **`8f34559f` R339**; R914 MERGE OK / n80 ConnectError. **Never `pkill -f`**.
- p4022–p3900: REFUTE→TRAIN; α→τ→Lium; stamp after `test -x`; never `pkill -f`.

