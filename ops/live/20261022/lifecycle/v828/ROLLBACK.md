# Rollback: MiMo v8.2.8 lifecycle host layer (2026-10-21, authorized by Jacob)

Changed: ONLY a new lifecycle drop-in `zzzz~~~~~~~~~~~~v828-release.conf` in
`~/.config/systemd/user/affine-mimo-lifecycle.service.d/`. It runs `release-v828-signed/mimo_v828_launcher.py` with
`v828-operator.ROOT-ADMITTED` (56e57851...), which loads the signed v8.2.7 uid launcher + uid operator IN PLACE (pinned by
sha256; release-v827-signed untouched) and adds, in this host process only:
1. the original `Lifecycle._audit_population` runs in a background thread after DISPATCH (trainer job dispatched first);
   FINALIZED waits for it, so `epochs/E/AUDIT-POPULATION.json` (read by the v9 scorer + dashboard) and its R2 copy are
   byte-identical to before; per-tick re-authentication (~110 s/tick) no longer repeats;
2. audit scheduler disabled (no audit jobs scheduled/queued). Config `audit_scheduler.enabled` and `run()` unchanged;
3. native grading pool workers 8 -> 32 (original ordered pool).
Lifecycle config `lifecycle.v8.2.4.ACTIVE` (c38a9fcf), CURRENT.json, policy, source closure, trainer, writer, v9 scorer,
chain: UNCHANGED. The v8.2.7 / v8.2.6 / v8.2.4 drop-ins are left in place; nothing was overwritten or deleted.

## A. Roll back to v8.2.7
Prefer an epoch boundary (current epoch FINALIZED, next one without captures). Restart after POPULATION.json exists is
also safe (original seal/audit-population/training recovery is idempotent).

    rm ~/.config/systemd/user/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~~~~~v828-release.conf
    systemctl --user daemon-reload
    systemctl --user restart affine-mimo-lifecycle.service
    tr '\0' ' ' < /proc/$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)/cmdline   # expect uid-operator.v8.2.7
    journalctl --user -u affine-mimo-lifecycle.service --since -2min | grep '"component": "host-v824"'

The original audit scheduler then resumes from its durable state (auditor/ directory + roles.sqlite3). If the three
verify workers were stopped (see WORKERS.md), start them again so audits are executed:

    for n in 1 2 3; do systemctl --user start affine-mimo-verifier-$n-tunnel.service affine-mimo-verifier-$n-supervised.service; done

## B. Further back
Do A, then follow ../20261021-v827-incremental-capture/ROLLBACK.md.

## Backups
`backup-lifecycle-<UTC>/` in this folder: tar of the lifecycle drop-in directory before install, CURRENT.json, ExecStart and
MainPID before restart, BACKUP.sha256. Restore = `tar -C ~/.config/systemd/user -xf systemd-user-units.tar` + daemon-reload + restart.

## Disarm (before activation)
    pkill -f "^bash /home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/activate_v828.sh --arm"
Nothing is installed until the armed script logs "lifecycle restarted on v8.2.8". The arm also refuses to install unless
v8.2.7 has ACTIVATED, its first epoch finalized clean on the same pid, and the GO file `E<n>-VERIFIED.ok` exists.
