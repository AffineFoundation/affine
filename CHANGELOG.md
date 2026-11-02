# Changelog

## Unreleased

### GREEN files public by default + cleanup sweep (2026-11-02)

- Every GREEN miner file is published at https://affine.io/k1/turns/<epoch>/batch.json at the decision (default on).
  Fixes the publisher skipping `affine-batch-v2` turns; backfilled E485, E537, E544, E552, E555, E557. RED files and RED
  lr / kl stay private. See `ops/green_publish_default/`.
- Removed https://affine.io/llms-legacy.txt and 86 old `.bak` / `.before` copies from the live site (backed up privately).
- mining.json `training.promotion_gate` top-level numbers now match the live gate (z > 1 flat, gain > 0, pooled cap-hit < 0.33,
  >= 100 scored); the old E168-E270 values sit under `e168_e270_flag`.
- README: current trainer (v8.3.3j) and gate rule, inactive designs moved under history; removed four unreferenced
  `*-before-*` history files from docs/.

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
