
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
- p4117: R978 **REFUTE** m=−0.003060 ~−0.55× (thought✓187 B✗0.297 causality_fail k=3) → exact-PID reap :8002 → **R990 MidCtx MidRank Hiβ Mega HiLR** TRAIN R338 6,7 pid73874; R979 **REFUTE** m=−0.013218 ~−1.09× (thought✓197 B✓0.502 k=3) → exact-PID reap :8003 → **R991 MidCtx HiRank Midβ Mega HiLR** TRAIN R338 4,5 pid74782 (R990 kept); B300×8=0. **Never `pkill -f`**.
- p4116: R977 **REFUTE** m=−0.004384 ~−0.71× (thought✓199 B✓0.5625 k=3) → exact-PID reap :8002 → **R989 SoftCtx MidRank Midβ Ultra HiLR** TRAIN R337 6,7 pid70560 (R987 TRAIN 4,5 kept); R979 MERGE idle→chall:8003+**v4 n80 LIVE** R338 4,5 (Triton seed chall_r978); B300×8=0. **Never `pkill -f`**.
- p4115: R978 **MERGE_DONE sat idle** on warm R338 GPUs6,7 → chall:8002+**v4 n80 LIVE** (Triton seed chall_r964); R979 MERGE idle on 4,5 (next); R977 n80 ~76/80; B300×8=0. **Never `pkill -f`**.
- p4114: R977 **MERGE_DONE sat idle** on warm R337 GPUs6,7 → chall:8002+**v4 n80 LIVE** (Triton seed chall_r954); R978 also MERGE idle on R338 6,7 (next); B300×8=0. **Never `pkill -f`**.
- p4113: R974 **REFUTE** m=−0.004988 ~−0.51× (thought✓199 B✓0.4625 k=3) → exact-PID reap :8002 → **R988 MidCtx MidRank MidLoβ Mega HiLR** TRAIN R252 4,5 pid94775 (R975 TRAIN 6,7 kept); B300×8=0. **Never `pkill -f`**.
- p4112: R976 **REFUTE** m=−0.005110 ~−0.91× (thought✓186 B✓0.466 k=3) → exact-PID reap :8002 → **R987 MidCtx MidRank Loβ Mega HiLR** TRAIN R337 4,5 pid66448 (R977 TRAIN 6,7 kept); R974 n80 ~78/80; B300×8=0. **Never `pkill -f`**.
- p4111: R968 **REFUTE** m=−0.003081 ~−0.64× (thought✓192.5 B✓0.488 k=3) → exact-PID reap :8002 → **R986 SoftCtx HiRank Hiβ Mega HiLR** TRAIN R924 6,7 pid49928; R976+R974 MERGE idle→dual chall+**v4 n80 LIVE** R337/R252 :8002 (Triton seed chall_r949/r925) pid65564/93819; B300×8=0. **Never `pkill -f`**.
- p4110: R969 **REFUTE** m=−0.003849 ~−0.45× (thought✓171 B✓0.494 k=3) → exact-PID reap :8002 → **R985 MidCtx HiRank Midβ Ultra HiLR** TRAIN R924 4,5 pid46569; R968 TRAIN_DONE early@1036→MERGE→chall:8002+**v4 n80 LIVE** GPUs6,7 (Triton seed chall_r969 n_so=26) pid49211; B300×8=0 BL. **Never `pkill -f`**.
- p4109: r252 α29.5/~τ1.66→τ1.65→Lium (+~$354; `lium fund` still no `Subtensor.transfer` → `btcli`→`5FqACMt…zsThe`); R924 idle GPUs1,3 → **R984 MidRank Hiβ ShortCtx Ultra HiLR** TRAIN pid45343; B300×8=0 BL `8f34559f`. **Never `pkill -f`**.
- p4108: R969 **MERGE_DONE sat idle** on warm R924 → chall:8002+**v4 n80 LIVE** (Triton seed chall_r953 n_so=26); B300×8=0 BL. **Never `pkill -f`**.
- p4107: R970+R971+R972 **REFUTE** → **R981+R982+R983 TRAIN** crown 6,7/4,5/1,3; B300×8=0. **Never `pkill -f`**.
- p4106: R972 MERGE idle→**v4 n80 LIVE** crown :8004; B300×8=0. **Never `pkill -f`**.
- p4105: R970+R971 MERGE idle→**dual v4 n80 LIVE** crown :8002/:8003; B300×8=0. **Never `pkill -f`**.
- p4104: R962 **REFUTE** → **R980 SoftCtx HiRank Midβ Ultra HiLR** TRAIN R938 2,3; B300×8=0. **Never `pkill -f`**.
- p4103: α147.6/~τ8→Lium (+~$1.7k) + **R979** TRAIN R338 4,5; B300×8=0. **Never `pkill -f`**.
- p4102: R962 MERGE idle→**v4 n80 LIVE** R938 :8002; R959 **QUEUED chal-00957**. **Never `pkill -f`**.
- p4101: R959 **CROWN_OK**→**SUBMITTED** chal-00957; R964→**R978 TRAIN**; B300×8=0. **Never `pkill -f`**.
- p4100: R963+R954 **REFUTE** → **R976+R977 TRAIN** R337 4,5/6,7; B300×8=0. **Never `pkill -f`**.
- p4099: R960+R951 **REFUTE** → **R974+R975 TRAIN** R252; dual n80 R338. **Never `pkill -f`**.
- p4098: R337 dual n80 + R944 **REFUTE**→**R973 TRAIN** R926. **Never `pkill -f`**.
- p4096/95/90/87/86: R944 n80 deaths — teacher OOM/Triton/affine_pkg; repair=exact-PID + FORCE Triton seed + TP/gpu_mem. **Never `pkill -f`**.
- p4085: R938 **LOST** chal-00949 m=−0.000615 ~−0.31× vs reign36. **Never `pkill -f`**.
- p4076: catalog 8×B200 BL; UUID→1×B200 mislabel → **rm**. **Never `pkill -f`**.
- p4066: chall Triton miss on king-seed → FORCE seed from live sibling chall. **Never `pkill -f`**.
- p4061: peft writes `…/train/adapter/` not flat adapter path; `lium fund` Subtensor.transfer fail → btcli→`5FqACMt…zsThe`. **Never `pkill -f`**.
- p4058: R934 **REFUTE** m=−0.002638 ~−0.34× (thought✓188 B✓0.434 k=3) vs reign36 → exact-PID reap :8002 → **R946** MidCtx Midβ UltraExtra on crown 6,7 + **R937** MERGE_DONE idle→n80 on R338 4,5; R924 wait FATAL looked for flat adapter while `…/train/adapter` exists — next pass remerge. Stock B300=0 BL `8f34559f`. **Never `pkill -f`**.
- p4057: R926 **REFUTE** m=−0.004111 ~−0.80× (thought✓204 B✓0.4255 k=3) vs reign36 → exact-PID reap :8003 → **R944** SoftCtx Midβ UltraExtra on idle H100 + **R945** SoftCtx Hiβ UltraExtra on crown 1,3; R934 SIZE_OK→lean:8002 **n80 LIVE**; p4052 wait SSH-fragile → re-arm SIZE_OK self-stamp; crown SSH flaky (retry). Stock B300=0 BL `8f34559f`. **Never `pkill -f`**.
- p4056: R940 **REFUTE** m=+0.002212 ~0.77× (thought✓173 B✓0.3875 k=3) vs reign36 → exact-PID reap :8004 → **R943** HiRank Midβ ShortCtx UltraExtra(28800); p4053 wait_r926 died on SSH timeout (`set -e`) → re-arm ssh-tolerant + SIZE_OK→lean:8003; stock B300=0 BL `8f34559f`. **Never `pkill -f`**.
- p4055: R252 R3 **REFUTE** m=−0.000104 ~−0.03× (thought✓169 B✓0.55 k=3) vs reign36 → exact-PID reap :8002 → **R942** SoftCtx MidLoβ UltraExtra(28800) on free GPUs **4,5**; stock B300/H200/H100=0 BL `8f34559f`. **Never `pkill -f`**.
- p4054: R940 TRAIN_DONE+MERGE_DONE sat idle (READY_FOR_N80) while relays occupied 1,3/6,7 — launch chall on free GPUs **4,5 :8004** (not 8003/8002) + seed `chall_r928` (n_so=26). Stock B300=0 BL `8f34559f`. **Never `pkill -f`**.
- p4053: R933 **REFUTE** m=−0.005798 ~−0.68× (thought✓200 B✓0.45) → exact-PID reap → **R941** SoftCtx MidLoβ MegaSuperExtra; also kill late host-relay overwrite during n80 + arm **wait_r926→lean**. Stock B300=0. **Never `pkill -f`**.
- p4052: R936 **REFUTE** m=+0.000310 ~0.07× (thought✓156 B✓0.42) → exact-PID reap → **R934** host-relay→crown GPUs**6,7**; R252 king hung missing `__triton_launcher*.so` → exact-PID reap + FORCE seed from crown king (n_so=26) → king reload→R3 n80. Stock BL-only. **Never `pkill -f`**.
- p4051: R936 MERGE_DONE but lean path typo’d MidCtx Midβ → Triton wipe+seed from **king** missing `.cubin` → engine fail; fix=reseed from **chall_r928** (n_so=26 n_cubin=238) + REUSE skip-wipe → chall:8002 **n80 LIVE**; R926 rematch→host-relay crown GPUs**1,3** :8003. Stock BL-only. **Never `pkill -f`**.
- p4050: R924 **REFUTE** m=−0.00249 ~−1.00× + R927 **REFUTE** m=−0.00986 ~−1.18× → exact-PID reap → **R928** TRAIN_DONE idle merge→chall:8002 + **R933** MERGE_DONE host-relay→R337; R926 rematch merge (wait FATAL before adapter). Stock BL-only `8f34559f`. **Never `pkill -f`**.
- p4049: host-relay `wait -n` can set `fail=1` after all PIPE ok — **verify SIZE_OK on dst** then stamp+lean (R924 16/16); R252 chall shm-hang ~70m → exact-PID reap + king swap justice→vera + chall GPUs6,7 `--enforce-eager`; R927 relay→:8003 n80. Stock BL-only `8f34559f`. **Never `pkill -f`**.
- p4048: R929 **REFUTE** m=+0.004247 SE=0.003106 bar≈0.006212 ~**0.68×** (thought✓189 B✓0.371 k=3) vs reign36 → exact-PID reap chall:8003 → **R940** HiRank Midβ ShortCtx; stock BL-only `8f34559f`. **Never `pkill -f`**.
- p4047: R929 MERGE_DONE sat idle on crown → chall:8003 GPUs4,5 + v4 n80 (port≠R924 :8002); R337 **REFUTE** m=−0.009386 ~−0.95× → exact-PID reap chall → **R939** SoftCtx Midβ UltraExtra(28800); stock BL-only `8f34559f`. **Never `pkill -f`**.
- p4046: R337 chall hung (Triton `ImportError` missing `__triton_launcher*.so` in `chall_r337`) — exact-PID reap + **FORCE wipe+seed from king cache** (n_so=16) before relaunch :8002; R927 host-relay→R337:8003 on idle 6,7. **Never `pkill -f`**.
- p4045: B300×8=0 + BL B200 only → rent **8×H200** `mine-r938` SoftCtx MidRank Hiβ `$15.96` (`f092246d`→noble-wolf-22); R927 wait FATAL→fix `--adapter …/train/adapter` MERGE_DONE 16shards. **Never `pkill -f`**.
- p4044: R924 MERGE_DONE → host-relay parallel×4 → crown GPUs **6,7** :8002 chall+v4 n80; B300×8=0 + both 8×B200 BL (`8f34559f`/`fbb1135f`); non-BL 8×H200 `f092246d` @$31.92 available. **Never `pkill -f`**.
- p4043: R337 TRAIN_DONE but merge aborted on **marsplan** path (404); wait scripts that check flat `…/train/adapter_model.safetensors` FATAL before peft writes `…/train/adapter/` — fix=`--adapter …/adapter` + vera `@8e3f1695` (same as p4039). **Never `pkill -f`**.
- p4042: R338 **REFUTE** m=−0.003241 ~−0.35× (thought✓218 B✓0.399 k=3) vs reign36 → exact-PID reap chall:8002 → **R937** SoftCtx HiRank Midβ; stock only BL `8f34559f`. **Never `pkill -f`**.
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

