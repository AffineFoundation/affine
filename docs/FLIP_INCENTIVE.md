# Flip incentive (LIVE scoring — 2026-10-23)

**Status: LIVE.** On-chain mech-0 weights now use the flip+surprise formula.
Public scores at https://affine.io/scores.json carry
`version` / formula field `sn120-flip-incentive-live-v1` and `live_weights: true`.
`scores-shadow.json` remains a mirror of the same flip document during
transition. The MiMo complete-group mining contract is unchanged.

Miner-facing contract text (exact copy of the live site section published
2026-10-23) lives in `dashboard/public/llms.txt` under
“Flip-incentive LIVE scoring — October 23, 2026”, also served at
https://affine.io/llms.txt. Keep that section byte-aligned with the site when
editing either side.

## Client requirements

**No miner client change is required.** The LIVE cutover only changes which
score shares the hourly writer consumes. Submission format, nonces, runtime,
and proof transport stay the October 13 MiMo complete-group contract.

- Keep nonces 0–7, at least one completed correct and one completed wrong,
  same upload / TOPLOC / proof rules.
- Stock task search may stay a random shuffle over signed `environments[].indices`.
- Optional local preference of frontier / currently-failing tasks is strategy
  only; use `--indices` with an authorized subset if desired. Preference never
  expands the epoch or changes group size.

## What LIVE scores measure

Live shares reward recovering **failing** tasks (fail→pass), weighted by
surprise `w = −log(max(p̂, ε))`. Pass→fail is debited. Eligibility is hard_fail
on baseline probe status `{unresolved, wrong, error, mixed}`. Unearned /
unregistered mass burns to UID 0. Full numbered formula, previous-vs-now table,
and probe URLs are in `llms.txt` (do not fork wording here).

Dashboard charts on https://affine.io/ include Incentive (chain emissions),
Flip credit, Flip share, Performance, and Batches per epoch. #120 notice is
already posted.

## Cutover record (operator)

Cutover completed 2026-10-23: the hourly best-assessment writer reads flip
shares from the LIVE scores path. This docs PR does **not** change runtime
weight writers or secrets; it syncs miner-facing text to the already-live
contract.

## Optional read-only audit helper

```bash
.venv/bin/python ops/flip_incentive_audit.py summary
.venv/bin/python ops/flip_incentive_audit.py suggest-indices --limit 64
```

Fetches public LIVE scores and lagged `/probe/tasks/` dumps. Does not write
chain weights, change submissions, or alter the signed miner source.

## Related URLs

- https://affine.io/llms.txt
- https://affine.io/mining.json
- https://affine.io/scores.json (LIVE flip, `sn120-flip-incentive-live-v1`)
- https://affine.io/scores-shadow.json (mirror during transition)
- https://affine.io/probe/index.json · https://affine.io/probe/series.json
