# SN120 lifecycle host v8.2.9h — miner collection window 1800 s -> 1200 s

Authorized by Jacob Steeves on 2026-10-26 at 13:38 CR: "Cut gather to 20 minutes".

## What changed
- The affine.io "Gather" bar is the signed collection window: `manifest.deadline - manifest.start`.
  - It is set as `deadline = start + collection_seconds` in `subnet/mimo_lifecycle.py` `_plan`.
  - It is not download time or verification time.
- v8.2.9h is the v8.2.9g launcher plus three exact-site edits (`v829h_patch.py`, `v829g-to-v829h.diff`).
  - `COLLECTION_SECONDS` (1200 live; 1800 in the sibling revert release).
  - A `Lifecycle._current` wrapper that uses the same epoch-boundary pattern as the v8.2.4 `max_tasks` 128/256 switch.
  - CURRENT.json `collection_seconds` is re-stamped only while the current epoch has no PLAN.json. An in-flight epoch keeps its window.
  - Only 1200 or 1800 are accepted; any other value fails closed through the original
    "same original lifecycle release/configuration" check.
- Unchanged: `max_tasks` (256 groups = 2048 rollouts per update), the trainer, the train gate, the audit sidecar, bans, TOPLOC, nonces, the worker roster and the signed lifecycle config.
- Miners need no client change. The deadline comes from the signed manifest and `mining.json`.
- llms.txt now says collection targets 20 minutes from epoch e152.

## Activation
- `activate_v829h.sh` waits for e150 FINALIZED and e151 collecting, plus the concurrent trainer v8.3.2 activation to finish.
- Then it backs up, swaps the lifecycle drop-in and restarts.
- Activated 2026-10-26 at 13:54:43 CR. e151 kept 1800 s; e152 is the first 1200 s epoch.

## Revert
`REL=release-v829h-collect1800-signed FROM_EPOCH=<last finalized> activate_v829h.sh`
re-stamps 1800 at the next boundary.

## Not published
ROOT-signed private operator configs, the seed, and host-private paths.
