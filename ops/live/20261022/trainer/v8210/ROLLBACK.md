# Rollback: MiMo trainer host v8.2.10 optimizer_persistence=none (2026-10-22, authorized by Jacob)

Changed: ONLY the trainer host layer on the trainer pod eager-fox-7b (150.136.210.177:20301). v8.2.5 host layer +
`engine.optimizer_persistence` ('none' = the optimizer state stays in trainer memory; each epoch stages the model exactly
as before plus a small marker optimizer/NOT-PERSISTED.json instead of ~107 GB / 1,372 optimizer shards; cold recovery
rebuilds fresh AdamW moments with the step kept when no state was persisted). New files: host-v8210/
mimo_host_trainer_v8210.py, host-v8210/v8210_optimizer_offpath.py; all other host files byte-identical to v8.2.5.
Lifecycle, trainer_config (lr etc.), policy, source closure, writer, v9 scorer, chain, R2 layout, published model bytes:
UNCHANGED. Nothing on the pod or Arbos was overwritten (new pod dir host-v8210, new owned-worker-v8210s*.json,
new drop-in zzzz~~~~~~~~v8210s-owned-worker.conf which sorts after zzzz~~~~~~~pod-20261022-owned-worker.conf).

NEVER restart the trainer while a TRAINING-JOB is in flight (epochs/<CURRENT epoch_index>/TRAINING-JOB.json without
TRAINING-RESULT.json). Wait for the TRAINING-RESULT, then do A or B, then C. Tell Jacob before any rollback.

IMPORTANT for any rollback after v8.2.10 has run an epoch: the current generation has NO persisted optimizer state.
The restarted v8.2.5 worker cannot cold-restore it (its recovery requires optimizer/descriptor.json) -> the next job
would fail with a RecoveryRequired / missing-file error. Prefer A (sync variant: it keeps the v8.2.10 recovery code, so
it rebuilds fresh moments at the kept step once, then persists synchronously as v8.2.5 did). For B, first rebuild the
current generation's optimizer the e88 way (20261022-trainer-pod-migration/rebuild_trainer_state_pod.py method: fresh
moments, step kept) or keep A.

## A. Flip to synchronous persistence (stay on v8.2.10 code, pre-signed 'synchronous' = exact v8.2.5 staging)
    D=/home/const/subnet120-rewrite/state/root-audits/20261022-async-optimizer-save
    U=~/.config/systemd/user/affine-mimo-trainer-supervised.service.d; export XDG_RUNTIME_DIR=/run/user/$(id -u)
    sed 's/owned-worker-v8210s.json --sha256 [0-9a-f]*/owned-worker-v8210s-sync.json --sha256 8d36329f88446289078b9f9e188fe3ec1abc3c1276169b51688248a373977a5a/' \
        "$U/zzzz~~~~~~~~v8210s-owned-worker.conf" > /tmp/sync.conf && install -m 600 /tmp/sync.conf "$U/zzzz~~~~~~~~v8210s-owned-worker.conf"
    systemctl --user daemon-reload
    then drain-restart (C) with CFG=owned-worker-v8210s.json SHA=6c5ebcb7f74c65632a3b420d502e7bd1cb3b9c09ce65338a3ed45fd6ce0b7108.

## B. Full rollback to v8.2.5 (previous exact config owned-worker-v825s.json sha 8d796a92...)
    rm ~/.config/systemd/user/affine-mimo-trainer-supervised.service.d/zzzz~~~~~~~~v8210s-owned-worker.conf
    systemctl --user daemon-reload        # zzzz~~~~~~~pod-20261022-owned-worker.conf (left in place) applies again
    (see the IMPORTANT note above first), then drain-restart (C) with CFG=owned-worker-v8210s.json SHA=6c5ebcb7...

## C. Drain-restart (same as activate_v825.sh / activate_v8210.sh)
    R=/home/const/subnet120-rewrite/state/root-audits/20261022-trainer-pod-migration/rssh.sh   # retry if sshd refuses
    W=/workspace/affine-mimo-trainer-prod
    systemctl --user stop affine-mimo-trainer-supervised.service
    $R "$W/runtime-exact/venv/bin/python -I -B $W/owned-worker-v7.py --config $W/$CFG --sha256 $SHA --stop-owned-after-drain"
    poll the same command with --status until "alive": false, then
    systemctl --user start affine-mimo-trainer-supervised.service
    check: pod owned-worker-v7.log shows the expected component line (trainer-host-v8210 with optimizer_persistence
    synchronous, or trainer-host-v825).

Backups: backup-trainer-<UTC>/ in this folder (systemd drop-in tarball, trainer ExecStart, lifecycle CURRENT.json,
pod trainer CURRENT.json, worker status before). A cold optimizer restore at the next job is expected after any trainer
restart (exact moments if the generation has optimizer/descriptor.json, fresh moments + kept step if it has the marker).
