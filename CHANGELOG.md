# Changelog

## Unreleased

### Flip-incentive SHADOW package (2026-10-23)

- Document the flip+surprise shadow scorer for miners and operators.
  LIVE weights remain v9/seq7; shadow scores are at
  https://affine.io/scores-shadow.json (`live_weights: false`).
- Sync `dashboard/public/llms.txt` with the live site section
  “Flip-incentive SHADOW scoring — October 23, 2026” (exact wording).
- Add `docs/FLIP_INCENTIVE.md` for cutover notes and client guidance.
- Add read-only `ops/flip_incentive_audit.py` to summarize shadow shares and
  suggest lagged failing probe indices for optional `--indices` preference.
- **No miner client protocol change.** The MiMo 8-nonce complete-group contract
  is unchanged; no new client is required for the shadow path.
