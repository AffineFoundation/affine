
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
- p4260: **R1112 REFUTE** m=−0.001311 ~−0.12× + **R1113 REFUTE** m=−0.000686 ~−0.07× (thought✓199 B✓0.42/0.40 k=3) → R1131+R1132 TRAIN r924; **R1116 REFUTE** m=−0.005310 ~−0.58× + **R1117 REFUTE** m=−0.009354 ~−1.08× → R1133+R1134 TRAIN crown; purge stale merges same pass. **Never `pkill -f`**.
- p4259: crown overlay **/tmp 100%** (48×~66G stale `r*_merged`) → R1116/R1117 chall vLLM **Errno 28** after MERGE; rm stale (keep active) → **3.0T free** → re-arm n80 :8004/:8003 + `TMPDIR=/root/tmp`. Purge old merges every pass. **Never `pkill -f`**.
- p4258: **R1115 REFUTE** m=−0.017658 ~−1.58× (thought✓253 B✓0.4625 k=3) → exact-PID reap r926 :8002 → **R1130** MidCtx MidRank MidLoβ Hyper HiLR TRAIN pid**160370**; B300/B200×8=0. cryptoDev ShortCtx Hiβ Hyper HiLR toxic. **Never `pkill -f`**.
- p4259b: **R1064 LOST** chal-00974 m=**−0.000659** SE=0.000858 z=−0.77 n=1286 thought✓197 B✓0.462 vs reign36 (n80 was +0.006632 ~1.021×) — knife-edge n80 not live-predictive.
- p4257: **R1101 REFUTE** m=−0.002673 ~−0.48× (thought✓193 B✓0.60 k=3) → exact-PID reap crown :8002 → **R1129** ShortCtx LoRank MidLoβ Hyper HiLR TRAIN pid**282708**; R1115 N80~62/80; B300/B200×8=0. **Never `pkill -f`**.
- p4256: R1101 MERGE done 02:23Z but n80 waiter died (`lean_chall_n80_r338_gpus45…` missing; real=`lean_chall_n80_crown_r1101_gpus67…`) → crown GPUs6,7 idle ~2h → **re-arm n80** chall pid**279494** :8002; R1115 N80 loading r926. Audit wait→lean paths after MERGE. **Never `pkill -f`**.
- p4255: R340 Online-DPO **aborted** left idle GPUs6,7 → **R1128** SoftCtx LoRank MidLoβ Hyper HiLR TRAIN r340 pid**48086**; R1115 **MERGE** r926; B300/B200×8=0. Fill idle pairs same pass. **Never `pkill -f`**.
- p4254: **R1108 REFUTE** m=−0.001278 ~−0.20× (thought✓202 B✓0.525 k=3) + **R1109 REFUTE** m=+0.000952 ~0.17× (thought✓194 B✓0.527 k=3) → exact-PID reap r339 :8002+:8003 → **R1126** MidCtx LoRank Midβ Hyper HiLR + **R1127** ShortCtx LoRank Midβ Hyper HiLR TRAIN. B300/B200×8=0. **Never `pkill -f`**.
- p4253: **R1106 REFUTE** m=+0.002112 ~0.29× (thought✓195 B✓0.477 k=3) → exact-PID reap r337 :8002 → **R1125** SoftCtx LoRank Midβ Hyper HiLR TRAIN (r=16; ShortCtx MidRank Midβ Hyper HiLR R1119 already on GPUs4,5). B300/B200×8=0. **Never `pkill -f`**.
- p4252: **R1105 REFUTE** m=+0.001862 ~0.51× (thought✓179 B✓0.3375 k=3) + **R1102 REFUTE** m=−0.004769 ~−0.56× (thought✓189 B✓0.4875 k=3) → exact-PID reap r252 :8003 + r338 :8002 → **R1123** ShortCtx HiRank MidLoβ Hyper HiLR TRAIN + **R1124** ShortCtx MidRank Hiβ Hyper HiLR TRAIN. B300/B200×8=0. **Never `pkill -f`**.
- p4251: **R1096+R1097 REFUTE** (m=−0.000713 ~−0.19× / m=−0.002077 ~−0.45×) + **R1111 REFUTE** m=+0.003700 ~0.68× → **R1120+R1121+R1122** TRAIN. B300/B200×8=0. **Never `pkill -f`**.
- p4250: **R1107 REFUTE** m=+0.004879 ~0.66× → **R1119** TRAIN r337. B300/B200×8=0. **Never `pkill -f`**.
- p4228: paygo α→TAO→Lium via `btcli wallet transfer` (lium fund broken). **Never `pkill -f`**.


