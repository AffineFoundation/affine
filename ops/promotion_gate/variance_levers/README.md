# SN120 held-out gate: variance levers 1-3 (Jacob 2026-10-31 14:07 CR) - live from the first round opened after 2026-10-31 14:27 CR

Approved: "Go live with pooled sigma_run", "Go live with 8 candidate rollouts per task", "Go live with 2,000 tasks per round".
Not done (not approved): lever 4 (mixed-difficulty draw), the solo-vs-parallel load-bias test. Bar unchanged: GREEN iff gain > 0,
weighted z > 1.00 (strict), estimated cap-hit < 0.20, >= 100 kept tasks. Evidence: RA/20261031-variance-audit (README.md, summary.json).

| | before | rounds opened after 14:27 CR |
|---|---|---|
| tasks per round (n) | 1000 (~136 kept) | 2000 (~270 kept) |
| candidate rollouts per kept task (k_cand) | 4 | 8 (j < 4 share block-1 request seeds, CRN) |
| head | k4 selection on all n + 3 x k4 p_i on kept (12/task) | unchanged |
| sigma_run | max(floor 0, the round's own 2-df estimate) | pooled over the last 20 finished rounds (40 df), frozen at round open |
| expected SE (= bar in pp of weighted gain) | median 2.27 | ~1.26 |

## sigma_run policy (pooled-v1)
sigma_run^2 = max(0, df-weighted mean over the last 20 finished rounds of [between-block variance of the 3 head p_i block weighted
means - mean within-block sampling variance]); value = max(floor sigma_run = 0.0, sqrt(.)); computed once when a round opens
(heldout.sigma_policy -> ROUND.private.json "sigma_run_policy", published as current.json "sigma_run"); every turn of that round
uses it. Fallback to the old per-round rule if fewer than 5 pooled rounds. At the switch: 20 rounds, 40 df, excess variance
-0.016 pp^2 -> 0.000, one-sided 95 % upper bound 1.14 pp (reproduces the audit exactly). Why the point estimate and not the UB:
the per-round 2-df estimate is > 0 by chance in ~45 % of rounds (+0.95 pp^2 expected if the truth is 0); the leave-one-block-out
A/A with sigma_run 0 is calibrated (z rms 1.09, 5/36 > +1 vs 5.8 nominal); using the UB would re-inflate the bar without evidence.
Caveat (open, not tested per instruction): a possible load-dependent shift between parallel head blocks and solo candidate runs.

## Code (gate_service.py, heldout.py; patch_levers.py apply|revert, anchored, exact reverse)
Before: the turn read k_cand and the sigma_run floor from the LIVE flag, so a flag edit would have changed the open round. Now:
k_cand = ROUND.private.json k_cand; sigma_run = heldout.turn_sigma_run(round, cfg, baseline) (pooled value frozen in the round, or
the old rule for rounds without one). n was already per round (drawn at open). Rounds opened before the switch run unchanged.

## Files
activate_levers.sh (arm at a DECISION gap; MODE=revert; MODE=dryrun), revert.sh, watchdog.py (auto-revert: gate inactive 90 s,
>= 3 gate errors / 20 min, TRAINING-RESULT +40 min without DECISION, new round baseline not done +35 min, fail-closed in a new round,
turn not using the round's k_cand / sigma_run), setflag.py, patch_levers.py, patch_texts.py (+ ll_old.txt; mining.json/training.json
heldout.se + rule, gate.json rule, k1 green.json notes, llms.txt s6/s7), kit_patch.py / readme_patch.py (miner kit).
Revert: bash RA/20261031-variance-levers/revert.sh (closes a still-open new-rules round, never revealed; NOCLOSE=1 keeps it).
Dry run of the revert path (copies only): MODE=dryrun bash revert.sh.
