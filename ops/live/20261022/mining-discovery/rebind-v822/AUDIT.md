# 20261016 mining-discovery rebind to MiMo v8.2.2

Applied: 2026-10-16 18:44:59 UTC (13:44 CT). Operator approval: Jacob (via parent agent), scope = public mining observer only.

## Problem
https://affine.io/mining.json reported `discovery_binding_error` continuously since 2026-10-16 04:21:14 UTC
(Oct 15 23:21 CT) = the v8.2 release-succession activation (activation-backup-20261016T042113Z).
The observer (`affine-mining-discovery.service`, drop-in `zzzz~mimo-observer-v7.conf`) ran
`20261013-null-pow-v1/uid1000-host-prospective/publish_mimo_public_discovery.py` with
`observer.CANDIDATE.private.json`, which pins the v7 lifecycle config/entry and v7 capture/uid operators.
The v8.2 / v8.2.1 / v8.2.2 succession changed the lifecycle argv to
`release-v822-signed/uid-operator.v8.2.2.ROOT-ADMITTED.private.json`, but never re-bound the observer, so
`selector()` raised `ValueError('physical MiMo lifecycle config/entry/activation differs')` every 5 s.
Also: the observer looked for host wrapper files next to each operator JSON (v8.2.2 operators live in
release-v822-signed/, the unchanged host files stay in capture-latency-v1/ and uid1000-host-prospective/),
and the v7 ReleasePolicy rejects the v8.2.2 policy, so it must run against the v8.2.2 source.

## Change (no new signatures; no chain tx; lifecycle/trainer/writer not restarted)
1. `publish_mimo_public_discovery.py`: optional observer keys `host_capture_files_dir`, `host_uid_files_dir`
   (default = operator file's directory, i.e. old behaviour). All files still sha256-pinned; all ROOT
   admissions/opening/discovery signatures still verified against authority 3301134b…. Diff: publisher.patch.diff
2. New observer config: this folder/observer.v822.private.json (lifecycle.v8.2.2.ACTIVE 943ed077…,
   entry deployment-final-source-v822/ops/mimo_lifecycle.py d068abbd…, capture-operator.v8.2.2 0f0bfbd6…,
   uid-operator.v8.2.2, source_sha256 a2a03cb3…).
3. New drop-in `~/.config/systemd/user/affine-mining-discovery.service.d/zzzz~~mimo-observer-v822.conf`
   (WorkingDirectory/PYTHONPATH = release-v822-signed/deployment-final-source-v822). Old v7 drop-in left in place (overridden).
4. `systemctl --user daemon-reload`; restarted only affine-mining-discovery.service.

Backups: backup-20261016T184438Z/ (original script, observer.CANDIDATE.private.json, all drop-ins, unit-cat.before.txt);
also `uid1000-host-prospective/publish_mimo_public_discovery.py.bak-20261016T184438Z`.

## Result
Before: status=discovery_binding_error, accepting_submissions=false.
After:  status=epoch_closed, epoch affine-mimo9b-grpo-20261013-e000043, checkpoint e28d7496, error=null
(correct: e43 deadline passed, training). Expect status=mining / accepting_submissions=true when e44 opens.

## Rollback (one line)
B=/home/const/subnet120-rewrite/state/root-audits/20261016-mining-discovery-rebind-v822/backup-20261016T184438Z; rm -f ~/.config/systemd/user/affine-mining-discovery.service.d/zzzz~~mimo-observer-v822.conf && cp -p $B/publish_mimo_public_discovery.py /home/const/subnet120-rewrite/state/root-audits/20261013-null-pow-v1/uid1000-host-prospective/publish_mimo_public_discovery.py && systemctl --user daemon-reload && systemctl --user restart affine-mining-discovery.service

## Follow-up
Release activation scripts (activate_v8xx.sh) must also re-bind this observer config, or the next release will break mining.json again.
