# 20261018 dashboard v8.2.4 publication fix — progress
Cause: e052+ TRAINING-RESULT publication receipts are affine-mimo-model-publication-v2 (v8.2.4 trainer);
dashboard ran v8.2.2 validate_result (v1 only). Diag: v822 rejects e052-e056, v824 validator accepts e048-e056.
Fix: publisher installs ONLY the ROOT-admitted v8.2.4 publication validator (uid-operator admission sig verified,
mimo_uid_capacity.py sha pinned); lifecycle_config -> lifecycle.v8.2.4.ACTIVE (same chain 460a1f5e). No new signing.
[x] 1 new dir + source copy + patch (succession.diff), config sha 9e63ea63
[x] 2 backup/ (unit, drop-ins, network-data.json, network.sqlite, SHA256SUMS); service-override.ROLLBACK.conf
[x] 3 preview OK (44m33s, exit 0): 58 epochs, e052-e056 trained, e057 collecting
[x] 4 drop-in swapped + dashboard-only restart 2026-10-18 13:33:18Z (pid 3524175)
[x] 5 first new export live on affine.io 14:18:07Z (58 epochs, e056 trained, e057 closed)
[ ] 6 next refresh: running since ~14:18Z (continuous loop, 15 s sleep; cold refresh ~45 min), no errors as of 14:32Z; watch refresh-watch.log (export mtime) + journal
Rollback: ./ROLLBACK.sh
Evidence-publication: different cause (single 20261013 v7 ReleasePolicy, no chain; also signs+uploads R2). Report only.
