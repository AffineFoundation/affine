# SN120 disk guard

Policy (2026-10-30): on trainer, probe and evaluator keep only the current king (gate head) and the next
model queued to be checked (trainer CURRENT checkpoint / probe serve.ckpt). Anything a process has open,
mapped, uses as cwd, or names in its argv is never touched. Entries younger than MIN_AGE_MIN (180) are skipped.

Pruned: probe `ckpt/<id>` weight dirs and `*.partial` (plain-disk dir and `*.gocryptfs-orig` copy);
trainer `private/promotion-gate/keep/<id>` for non-protected heads. Caps listed logs at LOG_CAP_MB (copytruncate).
Alerts (`RA/20261030-disk-cleanup/ALERT.json`, journal) when free space is below threshold
(trainer 150 GB; a training job needs >= 135 GB).

Accumulation sources found 2026-10-30 (not all automated here): probe ckpt dirs per candidate (18 GB each, never
pruned by the gate), trainer keep/ (3 x 18 GB), fp32-master (FP32_KEEP=2, 34 GB each), experiment dirs left on pods
(/workspace/shadow 257 GB, evaluator /var/tmp qualification trainer-state 118 GB, k1replay 54 GB), full
roles.sqlite3 backups per recovery (52 GB each on affinevalidator), trainer-pod-archive tars, *.bak code copies.

Install: copy the unit files to ~/.config/systemd/user, `systemctl --user enable --now sn120-disk-guard.timer`.
Dry run: `python3 ops/disk_guard/disk_guard.py`. Revert: `systemctl --user disable --now sn120-disk-guard.timer`.
