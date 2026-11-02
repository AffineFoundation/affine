# GREEN files published by default (2026-11-02, Jacob pt1081u)

Every GREEN turn's miner file is published byte-exact at `https://affine.io/k1/turns/<epoch>/batch.json` when the
gate decision is final, with its lr / kl and the trainer report in `/k1/turns/<epoch>/turn.json`. This is on by default
(`K1T_PUBLISH_GREEN_FILES`, default `1`; `0` only pauses it). RED files, RED lr / kl and RED per-rollout data are never
published; a RED `turn.json` carries only the decision, counts, hashes and step statistics (the applied `kl_coefficient`
is removed from RED reports because it would reveal the miner's `kl`).

## Root cause of the missing files

`k1_transparency.py` routed only `format == "affine-batch-v1"` to the free-file publisher. `affine-batch-v2` turns (optional
`train` lr / kl, live from E362) fell through to the old K=16 path, found no `TRAIN-DISPATCH-K1.json` and returned without
an error, so v2 GREENs (E485, E537, E544, E552, E555, E557) were never published. Now every `affine-batch-*` format (or any
turn with a `TRAIN-DISPATCH-FF.json`) takes the free-file path, the turns index links `batch`, and a watchdog logs
`GREEN file not published` hourly for any final GREEN free-file turn without a public file.

## Files

- `services/k1_transparency.py`: the live publisher (sidecar `affine-k1-transparency.service`).
- `services/training_params.py`: the live mining.json `training` block; `promotion_gate` top-level numbers now show the live
  held-out rule (z > 1 flat, gain > 0, pooled cap-hit < 0.33, >= 100 scored) and the old E168-E270 flag values moved to
  `promotion_gate.e168_e270_flag`.

## Revert (on the validator host)

- publisher: `~/subnet120-rewrite/state/root-audits/20261102-green-publish-default/revert.sh`
- mining.json block: `~/subnet120-rewrite/state/root-audits/20261102-site-cleanup/revert-training-params.sh`
- removed website backup files and llms-legacy.txt: `~/subnet120-rewrite/state/root-audits/20261102-site-cleanup/revert-site-files.sh`
