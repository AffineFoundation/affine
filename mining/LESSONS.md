
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
- p4313: **R1178 ~−0.58× REFUTE** (m=−0.003298 SE=0.002859 thought✓159 B✓0.452) MidCtx HiRank Hiβ MidLR → MidCtx HiRank Hiβ LR exhausted R1139/R1159/R1178 → last free Hyper cell **R1200** ShortCtx HiRank Loβ HiLR same pass. **Never `pkill -f`**.
- p4312: **R1169 ~0.32× / R1173 ~0.09× REFUTE** reaped r924 :8003/:8004 → ShortCtx MidRank Loβ MidLR (**R1198**) + ShortCtx MidRank MidLoβ MidLR (**R1199**) same pass; R1178 n80 left on GPUs6,7; sole B200=`fbb1135f` BL. **Never `pkill -f`**.
- p4311: **R1174 ~0.03× / R1179 ~−0.31× REFUTE** → ShortCtx HiRank Loβ UltraLoLR (**R1197**) r938 + SoftCtx MidRank Loβ UltraLoLR (**R1196**) crown; also **R1173 ~0.09× REFUTE** orphan r924 (+R1169); stock empty. **Never `pkill -f`**.
- p4310: **R1177 ~−0.57× / R1180 ~0.24× REFUTE** → MidCtx MidRank Loβ UltraLoLR (**R1194**) + MidCtx HiRank Loβ UltraLoLR (**R1195**) same pass on r338; **R1174/R1179** orphans still pending; sole 8×B200=`fbb1135f` BL. **Never `pkill -f`**.
- p4308: fleet Removal **13:26Z→22T13:30Z** via `POST /pods/{id}/schedule-removal` (8 pods) + Soft/Dead in `mine.env`; recent wait/lean scripts do **not** Soft-abort (env Soft unused) — still retarget Soft for any post that sources `mine.env`. **Never `pkill -f`**.
- p4307: non-BL **H200×8** `eager-fox-11`/`4eb39f3b…` rented while fleet waiter only polls B300/B200 — **R1191** vera FullFT TRAIN pid**2434**; burn **~$456/h**. **Never `pkill -f`**.
- p4306: **R1168 ~−0.42× REFUTE** (m=−0.001512 SE=0.001782 thought✓163 B✓0.418) SoftCtx HiRank Hiβ UltraLoLR → SoftCtx HiRank Hiβ LR exhausted → **R1190** ShortCtx UltraLoLR same pass; empty cmdline chall ⇒ UUID-clear GPUs4,5; sole B200=`fbb1135f` BL. **Never `pkill -f`**.
- p4305: **R1175 ~−0.24× / R1176 ~−0.29× REFUTE** (thought✓ B✓) UltraLoLR → MidLR SoftCtx LoRank Hiβ (**R1188**) + Hiβ MidCtx MidRank UltraLoLR (**R1189**; MidLoβ LR exhausted) same pass; stock B300/B200/H200 empty. **Never `pkill -f`**.
- p4304: **R1158** BOOT_HF_DONE → teacher TP1 :8000 + GRPO TRAIN GPUs2,3 same pass; **R1170 ~0.57× REFUTE** (m=+0.004094 SE=0.003599 thought✓200 B✓0.399) MidCtx MidLR → **R1187** SoftCtx MidLR same pass; sole B200=`fbb1135f` BL. **Never `pkill -f`**.
- p4303: B300/B200 empty post-BL → rented non-BL **H200×8** `golden-orbit-7b`/`e350ebc9…` for **R1158** by node id (ssh_gpus=8); waiters that only poll B300/B200 miss H200 — rent H200 same pass when under burn floor. **Never `pkill -f`**.
- p4302: **R1167 ~−0.15× REFUTE** (m=−0.000534 SE=0.001785 thought✓171 B✓0.394) ShortCtx HiRank MidLoβ LR exhausted → **R1186** MidCtx MidLR same pass; H200×8 non-BL available while B200 BL-only. **Never `pkill -f`**.
- p4301: **R1166 ~0.45× REFUTE** (m=+0.002389 SE=0.002645 thought✓181 B✓0.423) ShortCtx LoRank Midβ LR exhausted → **R1185** MidLoβ MidLR same pass; sole ls B200=`fbb1135f` BL. **Never `pkill -f`**.
- p4300: **R1166** p4286 lean **TP2** util0.72 stalled ~38 GiB cutlass → kill-by-pid; relaunch **TP1 util0.85 GPU6** :8003 → CHALL_READY+probe+n80. Prefer TP1 on B200. **Never `pkill -f`**.
- p4299: **R1163 ~0.18× REFUTE** MidCtx LoRank Midβ MidLR → MidRank UltraLoLR same pass on r339 GPUs4,5; reap :8002 by token. **Never `pkill -f`**.
- p4298: lean free-poll must use **own** CUDA GPUs (R1163 polled 6,7 while chall on 4,5 → 120s stall); probe `/v1/completions` needs **model id from `/v1/models`** (not `default`→404); TP1 util0.85. **Never `pkill -f`**.
- p4297: **R1164 ~−0.36× REFUTE** SoftCtx HiRank Loβ LR exhausted → **R1180** MidCtx HiRank Loβ MidLR same pass on r338 GPUs4,5; reap :8003 by token. **Never `pkill -f`**.
- p4296: **R1159 ~−0.71× / R1162 ~−0.67× REFUTE** → MidLR / MidRank next same pass; empty `/proc/pid/cmdline` on finished chall ⇒ **skip tok-check, UUID-clear GPUs** (do not FATAL). **Never `pkill -f`**.
- p4295: **r340 R1142/R1144** TP2 NCCL-stall orphans (~1GiB, never CHALL_READY) + **R1156** TP1 util0.90 OOM on logprobs → kill-by-pid, relaunch **TP1 util0.85** + FORCE Triton + `/v1/completions` probe. **Never `pkill -f`**.
- p4294: **R1165 ~0.39× REFUTE** — MidCtx LoRank Loβ LR family exhausted (Hi/Mid/Ultra) → Hiβ UltraLoLR same pass; positive margin below bar still REFUTE. **Never `pkill -f`**.
- p4293: **R1160 ~−0.51× / R1161 ~−0.17× REFUTE** — ShortCtx LoRank Hiβ + MidRank MidLoβ LR families exhausted → SoftCtx/MidCtx UltraLoLR same pass; decision `note` may be wrong (trust train_launched). **Never `pkill -f`**.
- p4292: **R1155 ~0.91× / R1140 ~−0.04× REFUTE** → MidLR + ShortCtx UltraLoLR same pass; empty cmdline on finished chall ⇒ skip tok-check, clear by GPU uuid. **Never `pkill -f`**.
- p4291: **R1154/R1157/R1149/R1150 REFUTE** → MidLR/Hiβ next same pass (**R1169–R1172**); empty `/proc/pid/cmdline` on dying chall is OK if GPUs freed — do not FATAL. **Never `pkill -f`**.
- p4290: wait_merge_then_n80 with **wrong EXP dirname** leaves MERGE_DONE idle — fix path + relaunch lean_chall. **Never `pkill -f`**.
- p4289: **R1154** MERGE→Triton wipe+seed→probe→n80; sole B200=**fbb1135f** (BL). **Never `pkill -f`**.
- p4288: **R1138 REFUTE** ~**−0.70×** → do **not** MidLR-re-run (R1090 already) → Hiβ UltraLoLR. **Never `pkill -f`**.
- p4287–p4282: MidLR cascade after UltraLoLR REFUTEs. **Never `pkill -f`**.
- p4285: always wipe+seed Triton + probe `/v1/completions` before n80. **Never `pkill -f`**.
- p4281: blind `lium up --gpu` bypasses blacklist → **fbb1135f**; waiter = node-id+ngpu≥8. **Never `pkill -f`**.
- p4280–p4272: UltraLoLR cascade after HiLR REFUTEs. **Never `pkill -f`**.
- p4269: TP1 util**0.90** OOM → util≤0.85 on B200. **Never `pkill -f`**.
- p4268: TP2 NCCL stall → prefer TP1. **Never `pkill -f`**.
- p4265: positive margin below 2·SE still REFUTE. **Never `pkill -f`**.
- p4264: lean awk `/tmp/r…_merged/` breaks (regex `/` cut). **Never `pkill -f`**.
- p4259: crown **/tmp ENOSPC** → purge + `TMPDIR=/root/tmp`. **Never `pkill -f`**.
- p4259b: **R1064 LOST** knife-edge n80 not live-predictive.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


