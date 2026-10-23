# Changelog

## Unreleased

### Authenticity couple for flip credit + GRPO (2026-10-23)

- Couple TOPLOC/sampling authenticity to **both** flip credit eligibility and
  GRPO packing: fail or unverified ⇒ no flip score and no train pack use.
- Keep `no_authenticity_guard=true` (no authenticity bonus/penalty term).
- Verify **all** batches that can earn flip credit or enter GRPO (not spot-only).
- Never produce/submit all-zero weights: `subnet/flip_share_floor.py` retains
  last-good shares or refuses submit.
- Wire train pack filter in `subnet/mimo_training_authenticity.py` and
  `make_training_job` (`mimo/train-pack.json`); trainer refuses unaudited packs.
- Update miner-facing `dashboard/public/llms.txt`, `docs/FLIP_INCENTIVE.md`,
  README, and MiMo GRPO notes. Nonces 0–7 + mixed correct/wrong unchanged.

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
