# SN120 gate test-design study (SHADOW / offline only, 2026-11-02)

Nothing here changes the live gate. Code only. Private inputs (prior values, selection probabilities, holdout ids, per-task grades)
are read on the validator host and outputs are written to `RA/20261102-seqgate-shadow/private/` (0700). Nothing private is committed.

## 1. Betting e-process (seqgate.py, replay.py) - dropped (too expensive)
Anytime-valid PrPl/mixture betting CS at alpha 0.03 on 59 prior-v1 turns (E497-562): 0 GREEN at the live ~490 tasks (all 7 live GREENs flip);
bootstrapped to 2000 tasks it needs mean ~1,424 tasks/candidate. The live shadow was never started; its pod was released after 23 min.

## 2. Equal-budget comparison (cmp/)
`compare.py`: simulation from the live round's population (2PL IRT truth for the head, logit-shift candidate with per-task heterogeneity
s_eps=0.28 calibrated to the real per-task gain spread tau^2=0.0010), Poisson draw, head 12 / candidate 8 rollouts, today's per-env Hajek z.
Designs: A today (z>1 @500, floor 0.003), B CUPED (cross-fit, X = prior P + IRT P), C discordance-targeted 2-stage + IPW,
D group-sequential OBF/Pocock with futility, combos, paired seeds (shared-uniform rho), weighting prior / irt-info / irt clipped 0.05-0.95.
Each rule's constant is calibrated to today's simulated false-GREEN at Delta=0 (and to 5%).
`calib.py`: real-data calibration (tau^2, paired-seed correlation, real-turn CUPED variance ratio). `realcheck.py`: GS rule on the real
A/A 8v8 splits and on the live turns. `summ_cmp.py`: tables.

Headline (sim, false-GREEN held at today's 11.4%): power at +0.3/+0.5/+1 pt
- A today @500:                        31 / 50 / 86 %, 500 tasks
- D 300/600/900 OBF, futility z<0:      37 / 59 / 93 %, mean 449 tasks on the live gain mix (536 at Delta=0, max 900)
- A fixed @700:                        36 / 58 / 92 %, 700 tasks
- CUPED: no gain (sim SE +2%; real-turn cross-fit residual variance ratio median 1.01-1.05). Stratified+IPW: lower power at equal cost.
- Paired seeds: SE -7% at rho 0.08 (head-vs-itself within-task correlation measured 0.08).
Revert: delete priors/seqgate (nothing live depends on it).
