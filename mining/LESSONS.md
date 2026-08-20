
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
- p4164: **R1014 REFUTE** m=−0.005437~−1.08× thought✓184 B✓0.435 + **R1019 REFUTE** m=+0.003346~0.95× thought✓165 B✓0.307 → exact-PID reap → **R1030 Ultra MidLR** r252 4,5 pid**125919** + **R1031 Ultra MidLR** crown 1,3 pid**189504**; R1021/R1022 waiters had **mega≠ultra** LEAN path → fixed + n80 rearm :8002/:8003; r926 teacher ConnectError mid-n80 → rearm util**0.82**/max_len**57344**. **Never `pkill -f`**.
- p4162: **R1012 REFUTE** m=+0.002536~0.54× thought✓171.5 B✓0.475 + **R1018 REFUTE** m=+6.11e-5~0.02× thought✓160 B✓0.429 → exact-PID reap r938 :8002 + r338 :8003 → **R1029 SoftCtx HiRank MidLoβ Ultra MidLR** pid**31208** + **R1028 MidCtx HiRank MidLoβ Mega MidLR** pid**114034** + MERGE→n80; B300×8=0 BL-only. **Never `pkill -f`**.
- p4161: **R1017 REFUTE** m=+0.000341~0.08× thought✓165 B✓0.426 → exact-PID reap r338 :8002 → **R1027 SoftCtx MidRank Hiβ Mega MidLR TRAIN** GPUs6,7 pid**112797** + MERGE→n80 (kept R1018 n80 :8003); B300×8=0 BL-only. **Never `pkill -f`**.
- p4160: **R1005 REFUTE** m=+0.003497~0.58× thought✓171.5 B✓0.425 → exact-PID reap r924 :8002 → **R1026 SoftCtx HiRank Hiβ Mega MidLR TRAIN** GPUs6,7 pid**78745** + MERGE→n80; B300×8=0 BL-only. **Never `pkill -f`**.
- p4159: r926 H100 teacher was **TP4 on 0,1,5,6** with GPU7 idle + R1013 on 3,4 — shrink **TP4→TP2 on 0,1** (exact-PID reap workers; keep king:8001) frees **5,6** → **R1025 SoftCtx MidRank Midβ Ultra MidLR TRAIN** pid**77945** + MERGE→n80; B300×8=0 BL-only. **Never `pkill -f`**.
- p4158: TTL+Soft/Dead extend all 7 mine-* via `POST /pods/{id}/schedule-removal` → **2026-08-21T13:26Z** (+24h); Soft was **~17:21Z** (would kill trains before old TTL); B300×8=0; sole B200=`8f34559f` BL. **Never `pkill -f`**.
- p4157: α→TAO→Lium r252 **147.6α→τ8.2526** → Lium +**~$1811** (bal **~$79088**); `lium fund` still `Subtensor.transfer` miss → `btcli`→`5FqACMt…zsThe`; catalog 8×B200=BL-only (`8f34559f`,`fbb1135f`); B300×8=0. **Never `pkill -f`**.
- p4156: **R1004 REFUTE** m=+0.002056~0.57× + **R1011 REFUTE** m=+0.004668~0.46× → exact-PID reap r337 :8002/:8003 → **R1023 SoftCtx Midβ Ultra MidLR** + **R1024 MidCtx Loβ Ultra MidLR** TRAIN pids**92389**/**92386** + MERGE→n80. **Never `pkill -f`**.
- p4155: **R1009 REFUTE** m=+0.000550~0.22× + **R1010 REFUTE** m=+0.001741~0.39× → exact-PID reap crown :8002/:8003 → **R1021+R1022 SoftCtx Ultra MidLR TRAIN** GPUs6,7+4,5 pids**182556**/**182562** + MERGE→n80. **Never `pkill -f`**.
- p4154: **R998 REFUTE** m=+0.001298~0.33× thought✓171.5 B✓0.4125 → exact-PID reap r252 :8003 → **R1020 SoftCtx HiRank MidLoβ Mega MidLR TRAIN** GPUs6,7 pid**119314** + MERGE→n80. **Never `pkill -f`**.
- p4153: **R1006 REFUTE** causality_fail B✗0.2875 m=+0.001417~0.71× → exact-PID reap crown :8004 → **R1019 MidCtx LoRank Midβ Mega MidLR TRAIN** GPUs1,3 pid**181332** + MERGE→n80 (kept R1009/R1010 n80). **Never `pkill -f`**.
- p4152: **R998** train→merge only (no merge→n80 waiter) sat MERGE_DONE ~22m with GPUs6,7 idle — always arm MERGE→n80 with TRAIN; launched lean :8003 chall pid**114993**. **Never `pkill -f`**.
- p4151: filled r338 GPUs4,5 with **R1018** MidCtx HiRank Hiβ Mega MidLR pid**104157** + MERGE→n80; R1008 **chal-00961**; B300×8=0 BL-only. **Never `pkill -f`**.
- p4150: **R1008 CROWN_OK** m=+0.005917 ~1.325× → **SUBMITTED** reveal **31475534**; **R1003**→**R1016**; **R1007**→**R1017**. **Never `pkill -f`**.
- p4149: **R1001 REFUTE** → **R1015 MidLR TRAIN** r924 1,3; B300×8=0 BL-only. **Never `pkill -f`**.
- p4148: **R996+R1002 REFUTE** → **R1013+R1014 MidLR TRAIN**. **Never `pkill -f`**.
- p4147: BL `fbb1135f` advertised 8× but nvidia-smi=3× → check executor_blacklist before `lium up`. **Never `pkill -f`**.
- p4121: cryptoDev HF `model.language_model.*` layout OK — vLLM mapper remaps. **Never `pkill -f`**.
- p4109: `lium fund` Subtensor.transfer fail → `btcli`→`5FqACMt…zsThe`. **Never `pkill -f`**.
- p4101/p4141: R959 **CROWN_OK**→**LOST** chal-00957. **Never `pkill -f`**.

