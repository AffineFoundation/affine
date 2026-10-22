# Rollback: MiMo v8.2.5 trainer drop_unresolved (2026-10-21, authorized by Jacob 11:29 CR)

Changed: ONLY the trainer host layer on the pod (<TRAINER_HOST>:<TRAINER_SSH_PORT>). v8.2.4 host layer + `engine.drop_unresolved=true`
(new files host-v825/mimo_host_trainer_v825.py, host-v825/v825_batched.py; all other host files byte-identical to v8.2.4).
Lifecycle, trainer_config (learning rate etc.), policy dfe0cbf7, source closure, writer, v9 scorer, chain: UNCHANGED.
Nothing on the pod or Arbos was overwritten (new dir host-v825, new owned-worker-v825s*.json, new drop-in).

NEVER restart the trainer while a TRAINING-JOB is in flight (epochs/<CURRENT epoch_index>/TRAINING-JOB.json without
TRAINING-RESULT.json). Wait for the TRAINING-RESULT, then do one of:

## A. Flip the flag off (stay on v8.2.5 code, pre-signed `drop_unresolved=false` = exact v8.2.4 loss path)
    D=/home/const/subnet120-rewrite/state/root-audits/20261021-v825-drop-unresolved
    U=~/.config/systemd/user/affine-mimo-trainer-supervised.service.d
    sed 's/owned-worker-v825s.json --sha256 [0-9a-f]*/owned-worker-v825s-off.json --sha256 1564a0f0940a1140f84552d688ea34461f990514b9e89db8a1dddb82bbda10e8/' \
        "$U/zzzz~~~~~~v825s-owned-worker.conf" > /tmp/off.conf && install -m 600 /tmp/off.conf "$U/zzzz~~~~~~v825s-owned-worker.conf"
    systemctl --user daemon-reload
    then drain-restart (step C).

## B. Full rollback to v8.2.4 (previous exact config)
    rm ~/.config/systemd/user/affine-mimo-trainer-supervised.service.d/zzzz~~~~~~v825s-owned-worker.conf
    systemctl --user daemon-reload        # zzzz~~~~~v824s-owned-worker.conf (left in place) applies again
    then drain-restart (step C).

## C. Drain-restart (same as activate_v824.sh --trainer-restart)
    P=ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/home/const/.lium/known_hosts/0a7aba3d-9443-445b-9601-c78ac0d14b1c -p <TRAINER_SSH_PORT> root@<TRAINER_HOST>
    systemctl --user stop affine-mimo-trainer-supervised.service
    $P "/workspace/affine-mimo-trainer-prod/runtime-exact/venv/bin/python -I -B /workspace/affine-mimo-trainer-prod/owned-worker-v7.py \
        --config /workspace/affine-mimo-trainer-prod/owned-worker-v825s.json --sha256 8d796a925cb71981c4339b0cef710885b3925060ed7023cb639a2bd1670484a9 --stop-owned-after-drain"
    poll the same command with --status until "alive": false, then
    systemctl --user start affine-mimo-trainer-supervised.service
    check: pod owned-worker-v7.log shows the expected component line (trainer-host-v824, or trainer-host-v825 with drop_unresolved false).

Backups: backup-trainer-<UTC>/ in this folder (systemd drop-in tarball, trainer ExecStart, CURRENT.json, worker status before).
Cold optimizer restore at the next job is expected after any trainer restart (same as v8.2.4 activation).
