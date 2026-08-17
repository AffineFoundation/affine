
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
- p3675: **R660 TRAIN** MidCtx MidRank HiBeta ep3×LoLR on idle brave **4,5** (amplify R650 ~0.47× / R625 @5e-6; ≠ R654 LoBeta / ≠ R657 MidBeta / ≠ R659 Short HiBeta) — fill free train GPUs while R634 SCP uplink live (train ≠ dual-pipe); R634~47G/11sh; kept=225. **Never `pkill -f`**.
- p3674: **R659 TRAIN** Short MidRank HiBeta ep3×LoLR on idle brave **2,3** (amplify R622 ~0.85×; anti-overfit R627 ep3@5e-6 / R632 ep2×LoLR; ≠ R653 LoBeta / ≠ R658 MidBeta) — fill free train GPUs while R634 SCP uplink live (train ≠ dual-pipe); R634~43G/10sh. **Never `pkill -f`**.
- p3673: **R634 v4 ARM** before SCP finishes — replace pre-wvk7 lean (no k=3 assert / wrong Soft MidBeta card / Triton purge) with `*_wvk7` outs + fail-closed `n_teacher_samples==3` + Triton REUSE; restart wait by PID (not `pkill -f`); R652 MERGE_DONE but do not dual-pipe brave while R634 SCP live. **Never `pkill -f`**.
- p3672: **R658 TRAIN** Short MidRank MidBeta ep3×LoLR on idle crown **6,7** (amplify R592 ~0.65×; ≠ R653 LoBeta / ≠ R631 ep2 / ≠ R656 HiRank) — fill last free crown GPUs when B300 stock empty; R634~38G/9sh. **Never `pkill -f`**.
- p3671: **R657 TRAIN** MidCtx MidRank MidBeta ep3×LoLR on idle crown **4,5** (amplify R642 ~0.62×; ≠ R654 LoBeta / ≠ R635 Soft MidBeta) — fill free crown GPUs when B300 stock empty; R634~36G. **Never `pkill -f`**.
- p3670: **R656 TRAIN** Short HiRank LoBeta ep3×LoLR on idle crown **2,3** (amplify R617 ~0.64×; ≠ R653 MidRank / ≠ R634 ep2 / ≠ R628 @5e-6) — fill free crown GPUs when B300 stock empty; R634~34G/8sh. **Never `pkill -f`**.
- p3669: **R655 TRAIN** MidCtx HiRank LoBeta ep3×LoLR on idle R252 **6,7** (amplify R649 ~0.50×; ≠ R654 MidRank) — fill free TKC train slots when B300 stock empty; R634~32G/8sh. **Never `pkill -f`**.
- p3668: **R654 TRAIN** MidCtx MidRank LoBeta ep3×LoLR on idle crown **0,1** (R653/R652 on brave; R634~30G/7sh) — fill free crown GPUs with MidCtx×ep3 sibling of R637 when B300 stock empty. **Never `pkill -f`**.
- p3667: **R653 TRAIN** Short MidRank LoBeta ep3×LoLR on idle brave **0,1** (R652 still **2,3**; R634 uplink live) — fill free train GPUs with a distinct Short×ep3 axis when B300 stock empty. **Never `pkill -f`**.
- p3666: **R651 ARMED** after MERGE_DONE — lunar waiter+lean (`*_wvk7`); host-relay gated **R634 then R647** (never dual-pipe brave); R596 v4 REFUTE harvested. **Never `pkill -f`**.
- p3665: **R596 v4 n80** — warm chall already on R252 :8002; launch `relaunch_r596_n80_reign34_wvk7_p3665.sh` (new `*_wvk7` outs; assert stamp k=3); teacher GPUs 100% within ~30s; B300 stock still empty. **Never `pkill -f`**.
- p3664: **fleet v4 sync** — tar-deploy toml+score+terms+dueling+config+chat → all mine-* `affine_pkg`; zesty verify LME>mean; R634 still SCP~19G/5sh. **Never `pkill -f`**.
- p3662: **R643 lean died** after GPUs free — `du -sm $TCACHE` on missing `chall_r643` under `set -o pipefail` (same R648 landmine); fix `du … || true` + `${_pre_sz:-0}` → seed from chall_r640 → CHALL_READY→n80. **Never `pkill -f`**.
- p3661: **R646 REFUTE** m=−0.003615~−0.49× (thought✓154 B✓0.429); reap golden 4,5 → **R647 ARMED** (Short HiRank MidBeta) gated on R634 SCP; restart **R633** to also wait R647 SCP_READY (never dual-pipe). **Never `pkill -f`**.
- p3660: **R641 REFUTE** m=+0.001550~0.31× (thought✓169 B✓0.418); kill stuck **R634** host-relay waiting forever on **R631 DEFER** — bypass gate when R641 ready + GPUs 6,7 free; brave→zesty SCP. **Never `pkill -f`**.
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
- Offline-DPO: Soft MidRank MidBeta **ep2/ep3×LoLR REFUTE** (R630/R635). Soft HiRank MidBeta/LoBeta **ep2×LoLR REFUTE** (R629/R636). Soft MidRank HiBeta **ep2×LoLR REFUTE** (R645 ~−0.29×); **ep3×LoLR R651 ARMED**. Soft MidRank LoBeta **ep3×LoLR** (**R637 SUBMITTED**). Soft HiRank MidBeta **ep3×LoLR** (**R633 ARMED** after R647). Soft HiRank LoBeta **ep3×LoLR R652 MERGE_DONE**. Short HiRank LoBeta **ep3 REFUTE** (R628; **ep2×LoLR R634 SCP**; **ep3×LoLR R656 TRAIN**). MidCtx MidRank MidBeta→**R642 REFUTE** ~0.62×; **ep3×LoLR R657 TRAIN**. MidCtx HiRank MidBeta/LoBeta→**R644/R649 REFUTE**; **ep3×LoLR R655 TRAIN**. MidCtx MidRank HiBeta→**R650 REFUTE** ~0.47×. MidCtx MidRank LoBeta→**R639 MERGE_DONE**; **ep3×LoLR R654 TRAIN**. Long MidRank MidBeta→**R641 REFUTE** ~0.31×. Long MidRank LoBeta→**R643 REFUTE**. Long HiRank LoBeta→**R648 REFUTE** ~0.68×. Long HiRank MidBeta→**R646 REFUTE** ~−0.49×. Short MidRank LoBeta→**R638 ARMED**; **ep3×LoLR R653 TRAIN**. Short MidRank HiBeta→**R627/R632 REFUTE**; **ep3×LoLR R659 TRAIN**. Short HiRank MidBeta→**R647 ARMED**. Short MidRank MidBeta→**R631 DEFER**; **ep3×LoLR R658 TRAIN**. **R640** VeraT4 Soft MidRank MidBeta **REFUTE** ~−0.71×.
- Live corpus **schema v2**; prefer **direct SSH** when lium 403. Pods: `mine-*` only; always `--ttl`.
- Never `pkill -f`. Kill by PID. Seed chall Triton from `/root/.triton/cache/chall`.
- Never edit a running bash script on the pod. Write `.new`, swap after exit.
- `/root/mine.env` must **export** vars; bare `HF_TOKEN=` does not reach python child.
- After LoRA: merge → graft visual → reload chall → fresh n80; engines **`max_model_len=65536`**.
- B300 serve: `CUDA_HOME=…/nvidia/cu13` + `VLLM_USE_FLASHINFER_*=0`.
- SCP shard count can hit 16 while a shard is still growing — wait for tar EOF / done marker, not `ls | wc`.
- crown `/root` cipher ENOSPC — purge finished merges/`r*_hf` before next pull.
