# Changelog

## Unreleased

### Flip-incentive LIVE cutover docs (2026-10-23)

- Document that on-chain mech-0 weights now use flip+surprise scoring.
  Public scores: https://affine.io/scores.json
  (`sn120-flip-incentive-live-v1`, `live_weights: true`).
- Sync `dashboard/public/llms.txt` with the live site section
  “Flip-incentive LIVE scoring — October 23, 2026” (exact wording).
- Update `docs/FLIP_INCENTIVE.md`, miner quickstart, and README from SHADOW
  preview to LIVE status.
- Point read-only `ops/flip_incentive_audit.py` at LIVE scores (still never
  writes chain weights).
- **No miner client protocol change.** The MiMo 8-nonce complete-group contract
  is unchanged; no new client is required.

### Flip-incentive SHADOW package (2026-10-23)

- Documented the flip+surprise shadow scorer for miners and operators.
  LIVE weights were still v9/seq7; shadow scores were at
  https://affine.io/scores-shadow.json (`live_weights: false`).
- Synced `dashboard/public/llms.txt` with the live site section
  “Flip-incentive SHADOW scoring — October 23, 2026” (exact wording).
- Added `docs/FLIP_INCENTIVE.md` for cutover notes and client guidance.
- Added read-only `ops/flip_incentive_audit.py` to summarize shadow shares and
  suggest lagged failing probe indices for optional `--indices` preference.
- **No miner client protocol change.** Superseded by the LIVE cutover entry
  above after 2026-10-23 activation.
