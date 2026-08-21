
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
- p4330: r924 idle challs after R1198/99/1200 REFUTE → exact-PID reap :8002/:8003/:8004 (leave T:8000 K:8001) → Mega **R1222** ShortCtx HiRank Loβ MidLR / **R1223** ShortCtx MidRank Loβ HiLR / **R1224** MidCtx HiRank Loβ HiLR TRAIN; no non-BL 8× stock. **Never `pkill -f`**.
- p4329: **R1209 REFUTE** m=+4e-5~0.007× (thought✓155 B✓0.444) → **R1221** ShortCtx MidRank Loβ Mega MidLR on r1191; **R1198/99/1200 REFUTE** (~0.34×/0.62×/−0.31×); r1214 multi-GPU `device_map=auto` → CUDA `operation not supported` — **single-GPU** TRAIN works; FullFT pod lacked **peft** → `pip install peft` + `/root/venv/bin/python3` before Offline-DPO. **Never `pkill -f`**.
- p4328: tarball nested `affine_pkg/affine_pkg/evalsrv` → flatten before train (`ModuleNotFoundError: evalsrv`); **R1187 REFUTE** m=−0.00715~−0.65× → **R1220** SoftCtx MidRank MidLoβ Mega MidLR; **R1218** relaunch pid**3091**. **Never `pkill -f`**.
- p4327: reap **R1197** orphan chall exact-PID → **R1219** SoftCtx LoRank Hiβ Mega HiLR (β=0.3 r=16 lr=2e-6 @12288) on r938 GPUs2,3; r938 is **4×H200** not 8×; R1187 n80 **58/80**. **Never `pkill -f`**.
- p4326: blank **r1214** BOOT → **R1218** MidCtx MidRank Midβ Mega HiLR (not MidLR SoftCtx=R1010); R1187 n80 healthy **28/80** at mlen65536. **Never `pkill -f`**.
- p4325: SoftCtx MidRank Loβ Mega MidLR = **R1010 REFUTE ~0.39×** — do not re-run (R1214 MidLR was a dup); SoftCtx chall **mlen32768** ContextLengthError on ~31k prompts → **TP2 util0.55 mlen65536**; crown orphan R1188/89/96 → Mega **R1215 SoftCtx MidRank Loβ HiLR** / **R1216 SoftCtx LoRank Hiβ Ultra** / **R1217 ShortCtx LoRank Hiβ MidLR**. **Never `pkill -f`**.
- p4324: **R1187** H100: TP2 util0.72 OOM → TP1 util0.80 no KV → **TP2 util0.65 mlen32768** n80 (later ctx-fail); rented H200×8 **r1214**. **Never `pkill -f`**.
- p4323: reaped R1192/93+R1194/95 → Mega **R1210–13**; R1187 FATAL missing sim. **Never `pkill -f`**.
- p4322: **R1187** wait midctx typo → lean chall; stamped R1192–97 REFUTE orphans. **Never `pkill -f`**.
- p4321: **R1201 REFUTE** ~−0.46× → **R1209** UltraLoLR; orphans → Mega **R1205–08**. **Never `pkill -f`**.
- p4320: teacher OOM@util0.90 → serve_three TP1 util≤0.85. **Never `pkill -f`**.
- p4319: **R1181/82/83 REFUTE** → Mega **R1202–04**; Hyper Soft/Mid/Short×rank×β×lr **108/108 full**. **Never `pkill -f`**.
- p4317: **R1191 ~0.02× REFUTE** → **R1201** HiLR; HF full ⇒ `SKIP_HF_PUSH=1` + `SKIP_LOCAL_TKC=0`. **Never `pkill -f`**.
- p4316: r340 merge_ready stamp + lean chall LOAD. **Never `pkill -f`**.
- p4315: R1191 pyarrow + T/K TP1 flashinfer-off; `run_sim_duel --n-turns`. **Never `pkill -f`**.
- p4314: HF push abort → `SKIP_LOCAL_TKC=0` + `SKIP_MERGE=1` local n80. **Never `pkill -f`**.
- p4313–p4300: Hyper/Mega cascade after REFUTEs; TP1 prefer; sole B200 BL `fbb1135f`. **Never `pkill -f`**.
- p4308: fleet Removal via `POST /pods/{id}/schedule-removal`. **Never `pkill -f`**.
- p4307/p4303: rent non-BL **H200×8** when B300/B200 empty. **Never `pkill -f`**.
- p4298: lean free-poll own CUDA GPUs; probe model id from `/v1/models`. **Never `pkill -f`**.
- p4296/p4291: empty cmdline on finished chall ⇒ UUID-clear GPUs (do not FATAL). **Never `pkill -f`**.
- p4290: wrong EXP dirname leaves MERGE_DONE idle. **Never `pkill -f`**.
- p4285: wipe+seed Triton + probe `/v1/completions` before n80. **Never `pkill -f`**.
- p4281: blind `lium up --gpu` → BL; waiter = node-id+ngpu≥8. **Never `pkill -f`**.
- p4269: TP1 util**0.90** OOM → util≤0.85. **Never `pkill -f`**.
- p4268: TP2 NCCL stall → prefer TP1. **Never `pkill -f`**.
- p4265: positive margin below 2·SE still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** knife-edge n80 not live-predictive.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


