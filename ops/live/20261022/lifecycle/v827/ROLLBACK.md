# Rollback: MiMo v8.2.7 lifecycle host capture layer (2026-10-21, authorized by Jacob)

Changed: ONLY a new lifecycle drop-in `zzzz~~~~~~~~~~~v827-release.conf` in
`~/.config/systemd/user/affine-mimo-lifecycle.service.d/`. It runs the byte-identical v8.2.4 uid launcher files (copied into
`release-v827-signed/`) with `uid-operator.v8.2.7` -> capture launcher v3 + `mimo_host_capture_cache.py` (v827) +
`capture-operator.v8.2.7` (workers 16). Lifecycle config `lifecycle.v8.2.4.ACTIVE` (c38a9fcf), CURRENT.json, policy dfe0cbf7,
source closure, trainer (v8.2.5), writer, v9 scorer, chain: UNCHANGED. The v8.2.6 drop-in (`zzzz~~~~~~~~~~v826-release.conf`)
and v8.2.4 drop-in are left in place; nothing was overwritten or deleted.

## A. Roll back to v8.2.6 (previous live composition)
Prefer an epoch boundary (current epoch FINALIZED, next one without captures). Never needs the trainer.

    rm ~/.config/systemd/user/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~~~~v827-release.conf
    systemctl --user daemon-reload
    systemctl --user restart affine-mimo-lifecycle.service
    tr '\0' ' ' < /proc/$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)/cmdline   # expect uid-operator.v8.2.6
    journalctl --user -u affine-mimo-lifecycle.service --since -2min | grep '"component": "host-v824"'

## B. Roll back to v8.2.4 (pre-today)
Do A, then also remove `zzzz~~~~~~~~~~v826-release.conf` (see ../20261021-v826-capture-deadline/ROLLBACK.md), daemon-reload, restart.

A restart during collection is idempotent (original open/collect/seal recovery) but drops the in-memory caches, so the next
pass is a full original collect. Restarting after POPULATION.json exists is also safe (seal resumes from the frozen population).

## Backups
`backup-lifecycle-<UTC>/` in this folder: tar of the lifecycle drop-in directory before install, CURRENT.json, ExecStart and
MainPID before restart, BACKUP.sha256. Restore = `tar -C ~/.config/systemd/user -xf systemd-user-units.tar` + daemon-reload + restart.

## What v8.2.7 adds over v8.2.6 (host process only)
1. Incremental validation on the capture thread inside ORIGINAL capture(): validate_document / validate_commitment results
   are reused only for byte-identical inputs (sha256 of the bytes just read from R2 + canonical claim + miner + task + exact
   opening digest); every document GET, LastModified window check and proof HEAD still runs for every slot. Snapshot writes
   already done/verified by this process for the same key+bytes are skipped. Exceptions are never cached.
2. Evaluator opening-parse cache: on the `mimo-evaluation-scheduler` thread only, json.loads of >=1 MiB byte-identical input
   (length + SHA-256) returns the earlier parse. Every other caller gets stdlib json.loads.

## Disarm (before activation)
    pkill -f "^bash /home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/activate_v827.sh --arm"
Nothing is installed until the armed script logs "lifecycle restarted on v8.2.7".
