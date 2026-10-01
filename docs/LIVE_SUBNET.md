# Live registration and weights

The live adapter defaults to Finney SN120, with operator wallet `default/default`.
It never registers paid test miners. `ChainAdapter.registrations()` returns only
currently registered hotkeys whose on-chain `affine2|activate|HOTKEY|SIGNATURE`
reveal verifies as Ed25519 over the existing Affine registration domain. The
commitment author must equal HOTKEY. Public keys are decoded from SS58 and used
with the supported Ed25519-to-X25519 conversion for sealed mailboxes. No private
miner key is needed by the validator. Ordinary sr25519 subnet membership alone
does not qualify for an encrypted mailbox.

Miner activation CLI: `python -m subnet.register --wallet NAME --hotkey ED_HOTKEY`
prints the signed activation payload without submitting. Append `--execute` to
publish its 180-second timelock reveal using the miner's own already-registered
hotkey. This performs no paid subnet registration and refuses the validator owner
as a test miner. Chain commitment capacity/rate errors remain explicit failures.
Run `python -m ops.new_subnet_checks` for nine offline payout/registration tests.

## Controller integration

`subnet.chain.ChainAdapter(state_dir).registrations()` returns a dictionary keyed
by SS58 hotkey, with `uid`, `public_key` (hex), `activate_block`, and `snapshot_block`.
Pin that identity snapshot at epoch opening. `submit_hour(points, registrations,
window_end, execute=False)` returns an inspectable status. Fresh current chain
registration and UID/hotkey mapping are rechecked before payout; recycled or changed
UIDs deny the whole payout rather than silently changing the reward distribution.
The subnet owner and operator wallet must match the expected public address.

`hourly_points(reports, window_end)` aggregates controller-authenticated final
reports with fields `epoch_id`, `finalized_at` (Unix seconds), and `points`
(hotkey-to-integer). UTC windows are half-open `[end-3600,end)`; one finalized epoch
may occur once. A report at exactly the hour belongs to the following window.
Only verified live points belong here; mock identities and unauthenticated uploaded
reports must never reach this adapter. Finalized report and identity snapshots
are private local controller outputs, not data accepted directly from miners.

The deployment worker reads `state/finalized-reports.json` and
`state/epoch-registrations.json`. Until they exist it waits without submitting.
Zero total points skips the extrinsic. Chain cardinality/max-weight policy denials
are reported without padding recipients. SDK 11 SetWeights plan/execute handles
commit-reveal; chain weights version and rate limit are queried rather than assumed.
A private flock prevents overlapping invocations. Successful hourly windows are
durably recorded, so retries cannot double-pay the same window. The timer retries
the most recently completed UTC hour every five minutes. Backlogged windows older
than the current completed hour require deliberate controller/operator recovery;
the timer does not blindly replay old payouts.

## Read-only inspection

```
.venv/bin/python -m ops.new_subnet_weights --inventory
.venv/bin/python -m ops.new_subnet_cutover
.venv/bin/python -m ops.new_subnet_weights
```

## Single writer cutover, after successful mock

Do not activate until the end-to-end mock and private controller output validation
pass. Preserve `~/.local/state/affine-transition/active`: it suppresses the original
validator payout path while keeping its intake/evaluation running.

```
.venv/bin/python -m ops.new_subnet_cutover --activate --mock-evidence state/e2e-final/report.json
install -m 600 ops/new_subnet_weights.service ops/new_subnet_weights.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now new_subnet_weights.timer
```

Mock evidence must be a locally produced JSON object with `success: true`.
Activation stops/disables the transition and owner-burn timers, stops any in-flight
prior writer service, records their original status, and enables this writer's local
marker. Installation is intentionally a separate reviewable step. No legacy eval,
bench, teacher, or original validator process is stopped by this cutover.

Rollback: stop/disable `new_subnet_weights.timer`, stop its service, remove
`state/live-chain/writer.enabled`, and restore the former timer from
`state/live-chain/cutover-before.json`. Keep the original payout guard unless
intentionally restoring legacy competition payouts. Never run both payout writers.
Do not use broad process-kill patterns.

Controller writes private payout inputs under `state/live/`; the service unit passes those exact paths. See `state/live-config.example.json` and `systemd/affine-epoch-controller.service`. A reachable HTTPS public_url is required for actual miners. The quick-tunnel template is a temporary ingress option and its generated hostname changes on restart; a named tunnel is recommended for durable operation.

## Configurable multi-environment controller

`examples/controller-multi-environment.json` is the production-capable service
contract, with actual versioned source/harness definitions, signed numerical
profile, full-audit policy, separate held-out indices and disk headroom. Supply
operator R2/HTTPS settings in a private config. This example remains permanently
nonpayable and never calls a chain weight method. Existing production services
were not changed or deployed by writing the example.

The controller accepts declarative `source`/`config` entries or fully pinned
`EnvironmentSpec` dictionaries. It validates unique environment IDs, task ranges,
harness budgets and held-out disjointness before opening. Every manifest signs
those definitions, the audit policy, numerical profile and evaluation contract.
The service uses the published deadline; it cannot silently close a payable epoch
early. `--once` explicitly permits early closure only for a nonpayable trial.

A stream pointer is available at
`/public/streams/nonpayable-service/current.json`. Use miner `--current-url` with
that signed endpoint when running a nonpayable service, or the usual global
`/public/current.json` for an explicitly payable service. The miner retries finite
search sweeps until the signed deadline/batch quota, reusing its pinned model and
cumulative cache. Exhausting one search sweep is not an epoch-completion claim.

Opening is journaled before capabilities are issued. Restart republishes a committed
local signed manifest or freezes an interrupted unpublished opening before a fresh
attempt. Cached training requires an exact previously recorded changed-checkpoint
binding. Pretraining checks require the greater of configured minimum headroom
(default2GiB) and twice model-file size plus512MiB. Capacity shortage retains the
frozen epoch for retry; it does not fabricate training or consume another epoch.
Provisional sampled scores and incomplete duplicate coverage are excluded from
hourly payout inputs, in addition to permanent nonpayable test exclusions.

### Direct R2 transport

Set `direct_r2: true` on the controller. The operator creates seven-day GET
capabilities for the signed stream pointer and signed epoch manifest. Bootstrap
miners with `state/<controller>/direct-discovery.json`'s `current_url` and trusted
`authority`; running miners adopt refreshed `current_url` capabilities only from an authority-verified signed pointer. Clients offline beyond seven days need a fresh operator bootstrap. Each
manifest pins `transport_policy: direct-r2-v1` and an exact `checkpoint.read_urls`
map. These shorter-lived GET URLs expire after the epoch plus one hour. File
hashes and the signed model file map remain the trust anchor. Direct-mode clients
reject absent URLs, non-R2 hosts and transport downgrades instead of falling back
to a tunnel.

Each sealed miner mailbox contains a single-object R2 PUT URL, its signed
Content-Type header and the deadline. The miner replaces its cumulative ZIP at
that one private staging key; it receives no bucket/list/read credentials.
Presigned URL expiry restricts request starts, not completion of an in-flight
upload. At closure, the controller reads body, ETag, length and LastModified from
one atomic R2 GET. It only accepts completed objects within `[start, deadline)`,
limits the ZIP to 100 MB, saves an operator-only SHA-addressed frozen copy, and
persists its receipt before publishing the public frozen object. Retries read the
persisted snapshot. Late PUTs can replace staging but cannot modify frozen bytes
or scores. A late completion visible before snapshot is excluded; a still-running
PUT can leave the last completed in-window object as the accepted snapshot.
Frozen receipts include expiring read capabilities for external audit.

The trial `ops/direct_r2_miner.py` fails on any HTTP request outside the R2 S3
endpoint and writes allowlisted host/method/status evidence. Its bootstrap file
and dedicated miner seed stay private; request evidence excludes query strings.
The conformance trial runs with its quick tunnel stopped. Small metadata and
large artifacts both travel directly to R2. No chain weights are submitted.

References: [R2 presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/),
[R2 S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/), and
[R2 conditional copy semantics](https://developers.cloudflare.com/r2/api/s3/extensions/).

### Public audits over direct R2

The signed stream pointer includes a renewable `history_url`. Its signed history
contains read capabilities for finalized manifests, frozen batch ZIPs, score and
audit reports, receipt challenges, training records, and both approved input and
trained output checkpoint file maps. Verify the operator authority first, then the
artifact signatures and SHA hashes. These GET capabilities confer no bucket list
or write privileges; the bucket remains private. The live client renews discovery
and history routes while online; an offline client beyond seven days needs a fresh
bootstrap. Historical source references are labeled `reviewed-compatible-reference`
and do not assert the exact worker bundle originally used. New manifests pin their
source archive before opening and the history labels those `epoch-signed`.

An empty, closed nonpayable epoch is journaled and advances using the same approved
checkpoint, with no optimizer step. A nonempty epoch with no verified training pairs
pauses at the training barrier. Nonpayable test results never become payout eligible.
