# K=1 "Pass rate & cap-hit per turn" chart: pass rates for every decided turn (Jacob pt904u, 2026-10-31)

**Symptom.** On affine.io the Head and Candidate lines stopped after E270 while the cap-hit bars ran on to the latest turn.

**Root cause.** The chart reads `turns[].h_pass / c_pass` from `/k1.json`, built by
`state/root-audits/20261027-k1-per-uid/code/k1_service.py` (user unit `affine-k1`) from the gate's `DECISION.json`
`evaluated.{h_pass,c_pass,n}`. With the held-out gate (first decided turn E272) the gate redacts `n` and the pass rates
from `DECISION.json` while the round is open ("published with the round reveal"). It keeps only `z`, `delta` and `c_cap_rate`.
Nothing went back to fill them in after the round closed, so every turn from E272 had `null` pass rates.

**Fix.** `pass_rates()` in `services/k1_service.py`:
- old gate format: unchanged (`pass_basis: gate-sample`, the ~1,000-task private gate sample with the turn batch removed);
- held-out format: once `/gate/heldout/<round>.json` (or `/gate/heldout/index.json`) says the round is `closed`, publish the gate's
  own aggregates `h_pass, c_pass, n` from `EVALUATED.private.json`. These are the numbers the decision used, on the round's kept tasks
  (`pass_basis: heldout`, `heldout_round`). Only these three aggregates are published, with no task ids or per-task cells;
- open round: `h_pass/c_pass` stay null with `pass_note` "held-out round open: ...". Fail-closed (E274) or a pending decision: null, with `pass_note`.

`c_pass` and `n` for every held-out turn were checked against the already-public reveals: kept tasks from `/gate/heldout/<round>.json`
and the candidate cells from `/gate/reveal/<e>.json`. All 81 match exactly (`tests/check_pass_rates.py`).

Site (`site/k1.js`): this chart only. One x slot per epoch. Lines break at gaps and are not interpolated.
A dashed marker at the switch to the held-out set (a harder, smaller set, so the level drops from ~85% to ~62%).
The tooltip shows the basis, n and why a point is missing. `index.html` only changes the k1.js cache-buster.

Scores and payees are unaffected (`score_sha256` unchanged).

Revert: `ops/k1_pass_rates/revert.sh` on the validator host (backups in `state/root-audits/20261031-pass-chart-fix/backup`).
