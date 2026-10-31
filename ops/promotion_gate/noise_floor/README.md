# SN120 held-out gate: SE uses the measured run noise (no fixed floor), live from E354

Approved by Jacob (pt900u, 2026-10-31 07:03 CR).

## What changed
The GREEN rule is unchanged: gain > 0, weighted_z > 1.00 (strict), estimated cap-hit < 0.20, at least 100 kept tasks.

The gate computes weighted_z = gain / SE, with

    SE = sqrt(se_task^2 + sigma_run^2 * (1 + 1/3))
    sigma_run = max(floor, measured run noise of this round)

The measured run noise comes from the round's 3 replicate head runs on the kept tasks (between-run variance of the weighted mean minus within-run sampling variance, floored at 0).

- E271-E353: floor = 0.022 (2.2 points). With se_task ~1.6-1.9 points, SE was 2.9-3.3 points, so the effective bar was about +3 points of weighted gain.
- From E354: floor = 0.0, so sigma_run is the measured value. Over the 78 decided held-out turns, SE would have been 1.42-2.68 points (median 1.87).

This is a config-only change: `PROMOTION-GATE.json` gets `heldout.sigma_run = 0.0`, with the history in `heldout.sigma_run_history`. The gate reads the flag at every decision. Each turn records the sigma_run it used, so round reveals recompute exactly. The gate code (`heldout.py` 80eeadb3) is unchanged.

## Evidence (`summary.json`, scripts `analyze.py`, `retro.py`)
- **Replicate runs.** 12 production rounds x 3 replicate head runs (24 df). The pooled run-level excess variance is -0.55 pp^2, so the measured sigma_run is 0 (95% upper bound about 0.7 points). The floor of 2.2 points was about 3x above anything measured in production.
- **A/A, leave one replicate block out (36 cases).** The gain SD is 1.61 points (k4 vs k8; about 1.5 points for the real k4 vs k12).
  - With se_task alone: z rms 0.85, 4/36 above 1.
  - With the old floor: z rms 0.49, 0/36 above 1.
- **Retro check.** Overnight, 58 turns from E293 on were all RED. None would have been GREEN without the floor. Earlier, three turns would have flipped: E276, E282 (same round) and E291, each with gain +2.4 to +2.6 points.
- **Caveat.** In production the 3 replicate runs share one serve session. Cross-session drift is not measured separately. The per-round estimate stays in the max(), so a round whose replicates disagree still widens its SE.

## Public text
- mining.json / training.json: `training.promotion_gate.heldout.se`
- gate.json: `rule`
- llms.txt: section 6

All are applied by `patch_texts.py`, an anchored, reversible patch.

## Commands (on affinevalidator, RA = ~/subnet120-rewrite/state/root-audits)
    WAIT_EPOCH=<E> nohup bash RA/20261031-gate-noise-floor/activate_noise_floor.sh &   # install in the gap after gate DECISION E
    MODE=revert bash RA/20261031-gate-noise-floor/activate_noise_floor.sh              # floor back to 0.022 + reverse text patches

## Live
Installed 2026-10-31 07:26:42 CR (13:26:42 UTC), after the e353 DECISION and in its own gap. First decision on the new SE: e354 GREEN.
- gain +8.36 points, se_task 2.13, sigma_run 0.26 points (measured), SE 2.15, z 3.88
- under the old floor: SE 3.32, z 2.52, so GREEN either way
- the round reveal recompute matches
