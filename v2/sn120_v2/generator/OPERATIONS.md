# Generator control-plane operations

SN120 **netuid 120, mechanism 1**. All published timestamps are UTC. These
files are delivery artifacts; they are not installed or started by the PR.

## Layout and credentials

The validator runs the timers as `const`, with Python
`/home/const/sn120-generator/venv/bin/python` and
`PYTHONPATH=/home/const/sn120-generator/src`. Copy the PR's `v2/` tree to that
source directory; do not change the live `~/affine` checkout. Use the isolated
Generator venv, not the live validator venv. Install `v2` with
its `links`, `r2`, `chain`, and `generator` extras.

Keep R2 access tokens out of the repository, unit files, command lines and logs.
The tick unit reads `/home/const/sn120-generator/r2.env`; use the same
`SN120_R2_ENDPOINT` and `SN120_R2_BUCKET` as the existing links job. A separate
credentials file may hold the two token fields using `R2_ACCESS_KEY_ID=` /
`R2_SECRET_ACCESS_KEY=` or their `SN120_R2_*` spellings. Restrict it to the
service user. Config example:

```json
{
  "site_dir": "/home/const/subnet120/affine/website",
  "work_dir": "/home/const/sn120-generator/work",
  "pool_dir": "/home/const/sn120-generator/pool",
  "network": "finney",
  "public_url": "https://affine.io",
  "credentials_file": "/home/const/sn120-generator/credentials",
  "ssh": {
    "host": "scorer-host-from-operator",
    "port": 22,
    "key_path": "/home/const/sn120-generator/scorer_ed25519"
  }
}
```

The SSH key is used only to reach the scorer, with strict host-key checking and
batch mode. The scorer has no R2 credentials and never publishes. Its job is
`/root/gen/jobs/<epoch>/job.json`; its output is collected privately, verified
against the deadline snapshot, and published by the validator. Before launch,
dispatch runs `ensure_head` with a 15-minute timeout. It accepts only the
requested current published head or matching read-links checkpoint, verifies
every base-file size and SHA-256, atomically replaces `/root/dl/base`, writes
`.head_version`, and stops/starts all eight servers via `serve.sh`. The scorer
command refuses any mismatch between that marker and the job. Unknown heads,
bad hashes or unhealthy servers prevent scoring. Preparation failures may be
retried; after a launch attempt, only collection is resumed.

The pool cache must contain all eight verified `sn120-math-prior-v1` shards,
`MANIFEST.json`, and the complete ID index. Use `PriorPool(...,
build_index=True)` once in an offline preparation step; never substitute another
pool. The index may be removed and rebuilt if damaged; the published manifest
hash remains `45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115`.

## Commands

All commands run with `PYTHONPATH=/home/const/sn120-generator/src` and the venv
Python. `$CONFIG` is the JSON file above.

```sh
python -m sn120_v2.generator.links issue --site-dir /home/const/subnet120/affine/website \
  --credentials-file /home/const/sn120-generator/credentials --network finney --mirror-r2
python -m sn120_v2.generator.snapshot --config "$CONFIG" --due
python -m sn120_v2.generator.dispatch --config "$CONFIG" --epoch YYYYMMDDTHHMMZ
python -m sn120_v2.generator.collect --config "$CONFIG" --epoch YYYYMMDDTHHMMZ
python -m sn120_v2.generator.publish --config "$CONFIG" --epoch YYYYMMDDTHHMMZ
python -m sn120_v2.generator.tick --config "$CONFIG"
python -m sn120_v2.generator.weights \
  --payout /home/const/subnet120/affine/website/generator/payout.json \
  --runs /home/const/sn120-generator/runs.jsonl --network finney
```

`weights` is a **dry run** unless `--execute` is explicitly supplied. The
systemd weights unit includes that flag; do not enable it until the separate
go-live authorization. It never addresses mechanism 0. There is no hand-set
weight command and no all-zero submission. Float shares are passed to the
Bittensor SDK for u16 normalization, and only `mechid=1, version_key=0` is used.
Missing/invalid/stale (>6h) payout or a changed earning hotkey selects UID 0.
The mech-0 activity, minute >=25, 240-block and 45-minute-success gates remain
in force. Each scheduled instant has a four-minute tolerance: `[35:20,39:20)`
or `[50:20,54:20)` UTC, including interpreter startup delay. For a manual
operator run, `--ignore-window` skips only the minute >=25 and timer-window
checks; all other gates still apply.

The standalone snapshot command expects a frozen king record captured at T;
use the tick driver for normal operation. Tick commits a seed at window open,
captures the public head pointer at T, issues links at most every 10 minutes,
snapshots at T+1m, dispatches, collects, and publishes. A lock prevents
overlapping runs. Links expire at T; within the final ten minutes, a separately
sealed `next` grant is provided. Conditional first-upload PUT is opt-in and
requires a support probe plus an `If-None-Match`-signed URL; otherwise links
explicitly allow overwrite until T, and the version at T is scored.

`jobs/<epoch>/evidence.json` is private audit/payout evidence;
`jobs/<epoch>/job.json` is the scorer's strict private input. Neither is public.
An empty epoch is published without dispatching a scorer. Scorer output must
cover exactly the eligible unique tasks. Unscored tasks and rejected tasks never
produce public full-text rollouts. `intake.jsonl` contains only
`problem`, `answer`, `source_id`, `prior_p`. It is written after results,
accepted records and accepted rollouts, followed by `intake.sha256` in the form
`<hex>  intake.jsonl`, then `state.json`. `published_epochs` is the durable
completion index for the box-side consumer. Empty intake files are still
published with the SHA-256 of the empty byte string.

Every local public write uses a same-directory temporary file and `os.replace`.
Public objects are mirrored to `public/generator/` in R2. The accepted ledger is
append-only, and a retry never appends a task twice. Once-ever ownership comes
from the earliest valid store upload (exact tie: lower submission file hash).

## Install and rollback (operator only, after qualification)

```sh
sudo cp v2/systemd/sn120-generator-{tick,weights}.{service,timer} /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/sn120-generator-*.service \
  /etc/systemd/system/sn120-generator-*.timer
sudo systemctl daemon-reload
sudo systemctl enable --now sn120-generator-tick.timer
# Qualify the dry run, inspect runs.jsonl, then explicitly enable:
sudo systemctl enable --now sn120-generator-weights.timer
```

To roll back the control plane, disable both timers. Do not delete accepted
history, in-flight jobs, seed records, or the last valid payout. Existing public
files remain available and the last committed weights remain on chain.

```sh
sudo systemctl disable --now sn120-generator-tick.timer sn120-generator-weights.timer
```

Restore a previously qualified `src/` tree before re-enabling. A partial epoch
publication may be resumed from its immutable private snapshot and scorer
output; a fully published epoch is never rewritten. Do not remove
`intake.sha256` as a recovery trick. If results are wrong, stop the timers and
preserve evidence for an explicit protocol/operator decision. Do not start any
legacy burn unit or change the mech-0 writer.

The specified `/home/const/sn120-slots-mech1/slots_mech1_weights.py` reference
was not available in the build environment or repository history. The supplied
gates are implemented explicitly in `weights.py`; a byte-for-byte comparison
could not be performed and should be done by the operator before execution.
