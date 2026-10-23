# Flip incentive (shadow → LIVE cutover)

**Status: SHADOW ONLY.** LIVE on-chain weights remain v9 / formula sequence 7 at
https://affine.io/scores.json. Parallel flip+surprise scores publish at
https://affine.io/scores-shadow.json with `live_weights: false`. This package
does not change on-chain submission and does not claim LIVE weights have flipped.

Miner-facing contract text (exact copy of the live site section published
2026-10-23) lives in `dashboard/public/llms.txt` under
“Flip-incentive SHADOW scoring — October 23, 2026”, also served at
https://affine.io/llms.txt. Keep that section byte-aligned with the site when
editing either side.

## Client requirements

**No miner client change is required** for the shadow path or for a later LIVE
cutover that only swaps which score file the hourly writer consumes.

- Keep the October 13 MiMo complete-group contract: nonces 0–7, at least one
  completed correct and one completed wrong, same upload / TOPLOC / proof rules.
- Stock task search may stay a random shuffle over signed `environments[].indices`.
- Optional local preference of frontier / currently-failing tasks is strategy
  only; use `--indices` with an authorized subset if desired. Preference never
  expands the epoch or changes group size.

## What the shadow scorer measures

Shadow shares reward recovering **failing** tasks (fail→pass), weighted by
surprise `w = −log(max(p̂, ε))`. Pass→fail is debited in the shadow formula.
Eligibility is hard_fail on baseline probe status
`{unresolved, wrong, error, mixed}`. Full numbered formula, LIVE vs SHADOW
table, and probe URLs are in `llms.txt` (do not fork wording here).

## LIVE cutover (operator only — not authorized by this doc)

When signed off, **one** change:

1. **Path swap (simplest):** point the LIVE weight reader / hourly submit from
   `scores.json` to `scores-shadow.json`, **or**
2. **Config flag:** set shadow `live_weights: true` **and** wire the hourly
   entry to consume shadow shares instead of the v9 vector.

Until then treat `scores-shadow.json` as preview only. Rollback restores the
LIVE path to `scores.json` (LIVE seq7 needs no restore if never modified).
Docs rollback for `llms.txt` is the ArbosLife backup
`llms.txt.before-flip-shadow-20261023`.

## Optional read-only audit helper

```bash
.venv/bin/python ops/flip_incentive_audit.py summary
.venv/bin/python ops/flip_incentive_audit.py suggest-indices --limit 64
```

Fetches public shadow scores and lagged `/probe/tasks/` dumps. Does not write
chain weights, change submissions, or alter the signed miner source.

## Related URLs

- https://affine.io/llms.txt
- https://affine.io/mining.json
- https://affine.io/scores.json (LIVE)
- https://affine.io/scores-shadow.json (SHADOW)
- https://affine.io/probe/index.json · https://affine.io/probe/series.json
