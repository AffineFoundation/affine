
# LESSONS — durable findings (Reason v3 era)
Hard-won knowledge, one line each. **Cap 150 lines.** Detail → `experiments/`.
S\* v2 era (retired 2026-08-10) → `archive/legacy-sstar-v2/` — ops only, not strategy.

## Scoring (Reason v3 + δ + thought-len + B, weight_version_key=6)
- **Reason = lpC(y_C|z_A) − lpC(y_C|∅)** per pair; miner score = mean. Formerly called Λ2.
- **Crown = submit gate (wvk=6, 2026-08-13):** paired mean(Reason_c − Reason_k) >
  **max(k_sigma · SE, min_margin)** with live `k_sigma=2.0` and **δ=`min_margin=0.002`**,
  **and** median stripped `len(z_A) ≥ min_thought_chars=80`, **and** teacher-side B
  pass rate ≥ `causality_gamma=0.30` (B=`lpC(y_A|z_A)−lpC(y_A|∅)` ≥ τ=0.02, no leakage).
- Miner-side causality / bank / r / baseline / L1lift are telemetry only — not the B license.
- Miner-side terms (L1lift, lpA, calibration r) do **not** enter Reason. Do not train them as objectives.
- Absolute Reason is only comparable within one duel slice. Use paired margin vs the live king.
- Confirm `weight_version_key` from `api/v1/contract` every pass (3→4→5→**6**).

## Strategy under Reason
- Shape `z_A` so the frozen teacher likes its own `y_C` more with the thought than without.
- Teacher refs / distillation data remain the free starting point; score is teacher-side only.
- Submit when a fresh-slice sim clears **margin > max(k·SE, δ=0.002)** vs the **live**
  king. Re-sim if the crown changed since the screen.
- p2399/p2401: mid-pipeline king flip — waiting `post_train` keeps old `KING_*` in process env; patching `mine.env` is not enough — kill-by-pidfile + relaunch **before** train.done (R69/R71/R73 guass→fjq); R67 vs fjq REFUTE m=−0.0115.

## Ops (still true — details in legacy archive if needed)
- p3659: **R641** SCP_READY then lean **purged** seeded Triton 170M→724K (same R640 landmine) — kill-by-PID, patch lean **REUSE if n_so≥25 & ≥100M / wipe+seed NO purge**, reseed from `/root/.triton/cache/chall`, CHALL_READY→**n80** pid724905. **Never `pkill -f`**.
- p3658: **R648 REFUTE** m=+0.005039~0.68× (thought✓148 B✓0.506); reap golden :8003 by PID → **R646 ARMED** Long HiRank MidBeta; **R631 DEFER** again (~6G slow; free brave uplink). **Never `pkill -f`**.
- p3657: **R648 lean died** after "GPUs free" — `du -sm $TCACHE` on missing `chall_r648` under `set -o pipefail` exits before seed; fix `du … || true` + `${_sz:-0}`; then wipe+seed from chall_r645 → CHALL_READY→n80. **Never `pkill -f`**.
- p3656: **R652 TRAIN** Soft HiRank LoBeta ep3×LoLR on idle brave **2,3** while R651 owns **0,1** and R648 uplink still live — fill free GPUs with a distinct Soft×ep3 axis when B300 stock empty. **Never `pkill -f`**.
- p3655: **R631 SCP STALL** @~12G/3sh flat while **R648** brave→golden uplink live — kill dead R631 ends by PID, purge dest, **defer repipe until R648 SCP_READY** (do not dual-pipe brave). **Never `pkill -f`**.
- p3654: **R645 REFUTE** m=−0.001633~−0.29× (thought✓153 B✓0.436); Soft MidRank HiBeta ep2×LoLR fails; reap golden :8003 by PID → **R648 ARMED** Long HiRank LoBeta (brave→golden); leave R637 :8004. **Never `pkill -f`**.
- p3653: **R645 n80 LIVE** after Triton REUSE — CHALL_READY→n80 in ~4m; king sampling progress within ~75s (not the R640 0%-CPU hang). **Never `pkill -f`**.
- p3652: **R651 TRAIN** on idle brave 0,1 (Soft MidRank HiBeta ep3×LoLR); **R645** lean still had incomplete-dir purge → 165M→696K — patch lean to **REUSE if n_so≥25 & ≥100M / else wipe+seed / never purge**; then CHALL. **Never `pkill -f`**.
- p3651: **R643 ARMED** after R640 REFUTE — kill R640 chall by PID (pidfile+:8003+argv), free lunar 4,5/:8003, crown→lunar host-relay for Long MidRank LoBeta MERGE_DONE; leave R537 :8002. **Never `pkill -f`**.
- p3650: **R640 REFUTE** VeraT4 Soft MidRank MidBeta LoLR m=−0.003038~−0.71× (thought✓ B✓); **R631 SCP** died again (p3638 SSH timeout to R252) — restart with ServerAliveInterval=15 CountMax=40; pipe growing. **Never `pkill -f`**.
- p3649: **R640 Triton** — CHALL_READY then n80 hung (0% CPU); Worker ImportError missing `__triton_launcher…so` after aggressive incomplete-dir purge raced TP compile (cache 61M→broken). Fix: kill-by-PID; `cp -a chall_r531` (**251M**, no purge); lean **REUSE preseed if n_so≥25 & size≥100M** + skip purge; empty-so purge of seeded tree destroyed 251M→828K — never do that. **Never `pkill -f`**.
- p3648: **R640 lean** failed `EOF: command not found` (stray bare `EOF` after README heredoc); waiter wrote `LAUNCHED` *before* lean succeeded — clear marker, fix via `.new`+`bash -n`, relaunch; do not trust LAUNCHED alone. **Never `pkill -f`**.
- p3647: **R640 UNBLOCK** — reign34 now **public** on HF; lunar `mine.env` HF_TOKEN overwrote host inject → "not found"; **source mine.env then host token last**; kill slow golden→lunar relay by PID and HF-pull. **Never `pkill -f`**.
- p3646: **R634 RETARGET** zesty **6,7/:8003** after R641 took 4,5/:8002 — kill old host_relay pid by PID (not `pkill -f`); wait R631+R641 SCP before brave→zesty pipe. **Never `pkill -f`**.
- p3645: **R641 ARMED** Long MidRank MidBeta ep2×LoLR MERGE_DONE on idle crown → host-relay pipe to zesty (warm reign34 TKC) 4,5/:8002; heredoc receiver steals stdin — use `ssh … "tar xf -"` not `bash -s <<EOF`. **Never `pkill -f`**.
- p3644: **R637 Stage-5 SUBMIT** Soft MidRank LoBeta ep3×LoLR @`7aded176…` hotkey r637 reveal **31387948** (n80 ~1.45×); HF push done in ~135s after host-token inject. **Never `pkill -f`**.
- p3640: **R637 CHALL** Soft MidRank LoBeta ep3×LoLR SCP_READY→lean :8004 on golden 6,7 (parallel R645 SCP on 4,5); dual-pipe brave uplink slows siblings — finish one pipe before arming the next. **Never `pkill -f`**.
- p3638: **R642 REFUTE** MidCtx MidRank MidBeta ep2×LoLR m=+0.004897~0.62×; **R631 SCP stalled** @17G → restart. **Never `pkill -f`**.
- p3637: **R649 REFUTE** ~0.50×; **R650 REFUTE** ~0.47×; Positive margin ≪ 2·SE still REFUTE. **Never `pkill -f`**.
- p3629: **R630 REFUTE** ~0.21×; Serialize R631→R634→R633→R638. **Never `pkill -f`**.
- p3589: **R596 SIGNAL vs reign34** m=+0.006196 z=2.63 (~1.31×) — Stage-5 licensed. **Never `pkill -f`**.
- Offline-DPO: Soft MidRank MidBeta **ep2/ep3×LoLR REFUTE** (R630/R635). Soft HiRank MidBeta/LoBeta **ep2×LoLR REFUTE** (R629/R636). Soft MidRank HiBeta **ep2×LoLR REFUTE** (R645 ~−0.29×); **ep3×LoLR R651 TRAIN**. Soft MidRank LoBeta **ep3×LoLR** (**R637 SUBMITTED**). Soft HiRank MidBeta **ep3×LoLR** (**R633 ARMED**). Soft HiRank LoBeta **ep3×LoLR R652 TRAIN**. Short HiRank LoBeta **ep3 REFUTE** (R628; **ep2×LoLR R634 RETARGET 6,7**). MidCtx MidRank MidBeta→**R642 REFUTE** ~0.62×. MidCtx HiRank MidBeta/LoBeta→**R644/R649 REFUTE**. MidCtx MidRank HiBeta→**R650 REFUTE** ~0.47×. MidCtx MidRank LoBeta→**R639 MERGE_DONE**. Long MidRank MidBeta→**R641 SCP**. Long MidRank LoBeta→**R643 SCP**. Long HiRank LoBeta→**R648 REFUTE** ~0.68×. Long HiRank MidBeta→**R646 ARMED**. Short MidRank LoBeta→**R638 ARMED**. Short HiRank MidBeta→**R647 MERGE_DONE**. Short MidRank MidBeta→**R631 DEFER**. **R640** VeraT4 Soft MidRank MidBeta **REFUTE** ~−0.71×.
- Live corpus **schema v2**; prefer **direct SSH** when lium 403. Pods: `mine-*` only; always `--ttl`.
- Never `pkill -f`. Kill by PID. Seed chall Triton from `/root/.triton/cache/chall`.
- Never edit a running bash script on the pod. Write `.new`, swap after exit.
- `/root/mine.env` must **export** vars; bare `HF_TOKEN=` does not reach python child.
- After LoRA: merge → graft visual → reload chall → fresh n80; engines **`max_model_len=65536`**.
- B300 serve: `CUDA_HOME=…/nvidia/cu13` + `VLLM_USE_FLASHINFER_*=0`.
- SCP shard count can hit 16 while a shard is still growing — wait for tar EOF / done marker, not `ls | wc`.
- crown `/root` cipher ENOSPC — purge finished merges/`r*_hf` before next pull.
