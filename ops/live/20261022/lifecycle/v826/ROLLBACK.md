# Rollback: MiMo v8.2.6 lifecycle host capture layer (2026-10-21, authorized by Jacob 12:08 CR)

Changed: ONLY the lifecycle systemd drop-in `zzzz~~~~~~~~~~v826-release.conf` (new file) in
`~/.config/systemd/user/affine-mimo-lifecycle.service.d/`. It runs the byte-identical v8.2.4 uid launcher files (copied
into `release-v826-signed/`) with `uid-operator.v8.2.6` -> capture launcher v2 + `mimo_host_capture_cache.py` (v826) +
`capture-operator.v8.2.6` (workers 16). Lifecycle config `lifecycle.v8.2.4.ACTIVE` (c38a9fcf), CURRENT.json, policy dfe0cbf7,
source closure, trainer (v8.2.5), writer, v9 scorer, chain: UNCHANGED. The v8.2.4 drop-in `zzzz~~~~~~~~~v824-release.conf`
was left in place; nothing was overwritten or deleted.

## Roll back to v8.2.4 (previous exact composition)
Prefer a boundary (current epoch FINALIZED, next not yet sealed). A restart during collection is idempotent (original
open/collect/seal recovery) but loses the in-memory capture cache, so the next pass is a full original collect.

    rm ~/.config/systemd/user/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~~~v826-release.conf
    systemctl --user daemon-reload
    systemctl --user restart affine-mimo-lifecycle.service
    # check: MainPID cmdline contains release-v824-signed/uid-operator.v8.2.4.ROOT-ADMITTED; journal shows "component": "host-v824"
    tr '\0' ' ' < /proc/$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)/cmdline

Never restart while a TRAINING-JOB is waiting for the trainer only if you can avoid it (it is safe for the lifecycle, the
trainer is a separate unit), and never touch the trainer unit for this rollback.

## Backups
`backup-lifecycle-<UTC>/` in this folder: tar of the lifecycle drop-in directory before install, CURRENT.json, ExecStart and
MainPID before restart, BACKUP.sha256. Restore = `tar -C ~/.config/systemd/user -xf systemd-user-units.tar` + daemon-reload + restart.

## What v8.2.6 changes (host process only)
1. Capture launcher v2 wraps `Lifecycle.tick`: if a tick returns `collecting` but the wall clock is already past the deadline
   (a pass started before the deadline and ended after it), the ORIGINAL tick runs again at once with a post-deadline time
   (complete listing, probes, collect of changed miners, original seal). Nothing is sealed from a pre-deadline observation.
2. CapturePass v826: per-miner fresh observation time; resolve_task memo; parallel read-only prefetch (GET/HEAD) consumed once
   by the original collect; probe transport errors fall back to an original full collect.
3. Root-cause fix: thread-local memo (capture thread only, inside one original collect / one pass) of pure derivations of the
   immutable opening: forced_sampling.binding, protocol.entries, complete_group_admission.scope, environments._source_hash.
4. workers 8 -> 16 (probe/prefetch read threads).
