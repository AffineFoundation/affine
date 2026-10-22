# SN120 env probe — ROLLBACK / STOP  (2026-10-22, CR = UTC-6)
Nothing here is on the weight path. The trainer, lifecycle, evaluator worker, v9 writer/scorer (seq7), hyperparameters, R2,
Tobias mech and the chain are untouched. Rollback = stop the probe and (optionally) delete its public files.
R=~/subnet120-rewrite/state/root-audits/20261022-env-probe

## Stop (Arbos Life)
systemctl --user disable --now affine-env-probe.service
rm ~/.config/systemd/user/affine-env-probe.service && systemctl --user daemon-reload

## Stop the pod side (evaluator pod 38.255.28.20:20092; frees the probe GPU memory, evaluator unaffected)
POD="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/home/const/.lium/known_hosts/6e469697-f3a7-4a6a-a1ae-40dde7120581 -p 20092 root@38.255.28.20"
$POD "python3 /root/affine-env-probe-20261022/podctl.py pause; for d in /root/affine-env-probe-20261022/runs/*; do python3 /root/affine-env-probe-20261022/podctl.py abort-run \$(basename \$d); done; python3 /root/affine-env-probe-20261022/podctl.py stop-serve"
Optional full removal (~40 GB: own venv, own checkpoint copies, env snapshots): $POD "rm -rf /root/affine-env-probe-20261022"
The evaluator dirs (/root/affine-mimo-20261013, /root/affine-mimo-public-evaluator-20261013) were only READ
(checkpoint cache copied with sha256 verification, source imported read-only); nothing was written there.

## Unpublish (optional)
rm -rf ~/subnet120/affine/website/probe     # https://affine.io/probe/* disappears; no other site file was changed

## Pause without stopping
Edit $R/code/config.json: "enabled": false  (the daemon re-reads it every 30 s), then run the pod "pause" above.

## Chain
No transaction was submitted. Staged commitments: $R/state/staged-commitments.jsonl (status STAGED-NOT-SUBMITTED).
