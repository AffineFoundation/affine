# AffineII dedup release kit

This is an **operator-only release kit**, not a deployment. It does not grant
permission to arm a round, change weights, or access production. Building and
qualifying this kit makes no production change.

## Current r000093 release

This r000093 incident-repair addendum supersedes the historical release and
round pins below. The production source is
`71d287f61338596ee24ad1dca654f8cf2cafe6e7`, with GPU digest
`07e946950889a1dfc9d8e59ce6b98fd955e2729b6ecf3b25912bf5660f556fda`,
canonical params hash
`4c83d52f7489f8eae22e550e226edc17fb6f2f0d9f6dce5a4d333369e5b1a3be`,
and `new-params.json` file SHA-256
`4183e5976910628dd162946600d756b43dc0a262320fa6dec414ad524f5f5309`.
The offline gate is `r000093`, with first health round `r000094` and second
open `r000095`.

The compatibility overlay accepts exactly the historical `d5979c30…` /
`28f1fb8b…`, repaired `8caddd9c…` / `2f7fd4ec…`, and current
`4c83d52f…` / `07e94695…` pairs, including the exact infrastructure-failed
r000092 close. It still restores old `c5252865` / `50276206…` /
`bba7a409…`; unknown pairs remain refused. Qualification uses fake
GPU drivers and isolated journals, not a physical-GPU or production test.
No deployment, chain write, merge, or `main` change is included.

## Historical release and rehearsal boundary

The approved release enables dedup, round-scoped training seeds, charged
null updates and **merged-candidate evaluation**, raises the crown threshold to
**`z_bar = 1.5`**, and makes **both
admission floors no-ops**:

- `min_signal_rollouts = 1` preserves the legacy requirement for at least one
  active rollout. `recipe.minibatches` adds no admission floor: old code used
  fewer nonempty minibatches when fewer active rows were available.
- `min_step_norm = 0.0` admits every valid nonnegative norm cap and accepts
  every finite stored norm, including zero. Non-finite stored norms remain
  charged `null_update` decisions.

`z_bar` is the existing crown threshold. Final judging requires a strictly
larger `z`, and early-look elimination uses the same threshold. The publisher
renders this value in `llms.txt`; `mining.json` includes it in `params`.
Merged evaluation builds each candidate with the install-fold arithmetic
`round_bf16(fp32 king master + LoRA update)`, rather than scoring a runtime LoRA
on an already-rounded bf16 base. Scored merged results must carry their tested
base hash and fold kernel, and install-fold promotion checks that identity.

Thus the previous concerns about UID 488's `global_norm_cap = 0.01` and
58-rollout batches no longer block admission. Dedup duplicates and non-finite
stored updates are still charged. The health gate fails if duplicate/null-update
cuts remove every selected batch or if a required healthy close does not occur.

The default `r000091` gate, first health round `r000092` and second open
`r000093` are **offline rehearsal identities only**. `GATE_WANT` may be
explicitly overridden, but the selected gate must immediately precede the
required health round in `expect.json`. Any real use still requires a separate
explicit operator `go` and `go-confirmed`; this kit does not deploy or authorize
production access.

## Frozen identities

| Item | Value |
| --- | --- |
| Old commit | `c525286580c84d795b3ca7cbdf2302e96fce31ba` |
| Old GPU source digest | `5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47` |
| Old canonical params hash | `bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a` |
| New commit | `50afaca32ea982796728a0b07cb6e3c579ef378b` |
| Integrated AFF-5 PR #310 head | `1fa67a9770884eb679ffeb91fd133ac49575d117` |
| New GPU source digest | `28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a` |
| New canonical params hash | `d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c` |
| `old-params.json` file-byte SHA-256 | `86c9cd5b5e696c6c10411fd828b7e26a641d06bd39f94a4ea98eb69de3e78f56` |
| `new-params.json` file-byte SHA-256 | `4acb73b3cf45e2240c147150e854ac27ab3f80851903b7adbb15b08452bc2784` |

A canonical `Params.params_hash` is not the SHA-256 of a pretty-printed params
file. The new params pin the combined worker recipe
`28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a`.
AFF-5 supplies the merged-candidate worker change; the release contains the
floor correction and no-op-floor publication as well. Its GPU tree has no diff
from `1fa67a97`.

The effective production-copy changes are exactly:

```text
dedup_enabled        false -> true
train_seed_scope     batch -> round
z_bar                1.0 -> 1.5
eval_merged          false -> true
recipe.code_digest   5027620615... -> 28f1fb8b...
```

The other five explicit dedup fields retain their legacy-effective values.
Expected rehearsal constants are `batch_cap=6`, `n_tasks=2000`, two null
strings per leased box, `active_size=10000`, and `irt_draw_rule=sqrt_fisher`.
`r000092` is the first health round and `r000093` the second-open proof.

## Production parameter route and journal durability

Production `autodeploy` uses the **ordinary journaled `set-params` CLI**, only
after the sole loop has stopped and `apply` has independently refused to run if
a loop is alive. It runs against the real production state directory with the
normal `sn120_v2.assembly:stack`, not with the copied-state isolation wrapper.
The complete seven-field dedup record plus `z_bar=1.5` and `eval_merged=true`
is required; the incomplete legacy file is never parsed by new code.

This is not an unjournaled local edit. `Journal` synchronizes against the one
existing R2 journal prefix, appends its leader and `params_set` records there,
then appends the same records to the local journal. No second state directory,
second journal prefix or alternate mirror is constructed, so no stray R2
journal copy is created. The guarded isolated path below is separately
qualified for operator inspection of a copied journal; it must not be pointed
at production state. Its copied `LocalStore` cannot write R2.

After apply, first-round health requires a clean close, all 2,000 tasks scored,
the new params hash, a successful open check, two 2,000-character null strings
per leased box, no `lease_failed`, and journal-logged duplicate/null-update
decision counts. Public evaluation timing comes from the real `run.log` and
public `eval_wall_s`; optional phase records are not required. Admission-cut
counters are diagnostics only because both floors are no-ops.

The external box-side validator-sync watcher can consume these stdout/log
markers without reaching the validator from production:

```text
APPLY OK: ...
PASS: dedup ...
ROLLBACK BLOCKED: ...
ROLLBACK FAILED: ...
ROLLBACK RECOVERY FAILED: ...
```

`APPLY OK` is emitted only after new code and params apply. `PASS:` follows the
complete first-round health gate. Every rollback failure path includes
`ROLLBACK`; a successful rollback additionally logs `old-code loop started
after rollback`. These markers are evidence hooks, not authorization or proof
of production execution during qualification.

## Rollback is not bare old code

**Rollback requires the reviewed compatibility overlay. Bare `c5252865` is not
sufficient after dedup journal history exists.** The old decoder rejects new
params fields, duplicate/null-update decisions, new round fields, and tagged
non-finite evidence. Do not rewrite history or delete sealed evidence to make
old code accept it.

Recovery is code-first:

1. Stop the sole loop using the real `run-loop.stop-after-round` request.
2. With the new checkout and new params, run the reviewed controller `run` to
   finish every open round, including an undecided or decided round with
   durable sealed evidence. Do not discard results or copy results across
   rounds.
3. Check out `c5252865` only after that finish succeeds.
4. Apply the **complete** old params file through `rollback-python`, never
   through new code. New-code production `set-params` rejects an incomplete
   legacy file; omission of new fields must not activate defaults.
5. Verify the old params hash through the same compatibility loader used by the
   old loop. The loader first validates the historical new pin, then exposes
   old behavior without changing historical records. The complete new pin
   includes `eval_merged=true` and the combined worker digest. After validating
   that pin, replay removes the seven dedup fields, historical `z_bar=1.5`
   override and `eval_merged=true`, and translates the reviewed combined worker digest to the reviewed
   old worker digest **in memory**. Journal bytes are never rewritten; native
   release readers still see the original new worker identity. The restored
   old behavior has `z_bar=1.0` and the old worker recipe. Unknown or newer
   parameters remain refused.
6. Start exactly one old loop through `rollback-loop.sh` and preserve its
   existing transaction cursor, UID mapping, best-assessment cadence and
   six-hour contribution half-life. Never enable burn or owner-only fallback
   weights. Retain the last valid assessment through an evidence/learning
   outage.

`manifest.json` pins every compatibility file. `rollback_cli.py` verifies the
overlay and old GPU source before importing it. The overlay validates charged
duplicate/null-update accounting and refuses open dedup work before writing a
leader fence. This is replay compatibility, not permission for old code to
finish a dedup ticket.

The separate validator/weights reader is **restored to `c5252865` as part of
rollback**. Native release `50afaca3` can also decode this history, but that is
not the required rollback configuration.

The existing `/etc/systemd/system/sn120-v2-weights.service` uses ordinary
`python -m sn120_v2.weights_writer`, and the operator cannot edit that unit.
The required route is therefore a **complete staged physical package**, not a
runtime import shim. `restore_validator.sh` reads an operator-supplied exact
`c5252865` package plus this kit and creates a new directory whose
`sn120_v2/` is the rsync source. It verifies all 123 old source files and the
old GPU digest, overlays exactly seven reviewed modules, and emits a
`restore-validator-manifest.json` covering every staged source:

```text
canonical.py  types.py  params.py  state.py  controller.py
dedup_compat.py  journal.py
```

`journal.py` includes the open-dedup-round pre-fence guard that the controller's
runtime loader otherwise installs. This restores old code plus schema
compatibility, not dedup execution behavior. There is no `sitecustomize.py`,
test hook or runtime import shim in the staged package.

**Never rsync onto an arbitrary mixed-release tree, and never run this script
against production state.** Use an operator-created archive/copy of the exact
old `c5252865` package as input. The stager refuses missing, extra, changed or
symlinked old source. It refuses to overwrite its output or tarball. It writes
only those new staging paths; it does not alter the source, systemd, a journal,
a store or chain state.

```bash
# Operator-only staging. OLD_PACKAGE contains sn120_v2 source, not the v2 root.
KIT=/OPERATOR-SELECTED/dedup-deploy
OLD_PACKAGE=/OPERATOR-SUPPLIED/c5252865/sn120_v2
STAGE=/OPERATOR-SCRATCH/restored-validator-c5252865-v1       # new path
TARBALL=/OPERATOR-SCRATCH/restored-validator-c5252865-v1.tar.gz  # new path
DEDUP_RESTORE_PY=/home/const/affine/v2/.venv/bin/python \
  "$KIT/restore_validator.sh" "$OLD_PACKAGE" "$STAGE" --tar "$TARBALL"
# Inspect the PASS, hashes, staged manifest and:
#   $STAGE/sn120_v2/
#   $TARBALL  (sn120_v2/ plus restore-validator-manifest.json)
```

If the service account cannot create files under `/home/const/affine/v2`, an
operator who owns that directory must perform the following replacement. No
sudo or unit change is assumed, but **the existing interpreter must already
resolve this source directory**. An adjacent source checkout is not
automatically on a venv's `sys.path`; its existing editable-install or `.pth`
configuration must point here. Before stopping any job, verify that fact:

```bash
/home/const/affine/v2/.venv/bin/python -c \
  'import sn120_v2; print(sn120_v2.__file__)'
```

The expected resolved package is `/home/const/affine/v2/sn120_v2/__init__.py`.
If it is elsewhere, a file-only rsync into this path cannot restore the running
jobs. The operator must repair that package-path installation or provide the
actual resolved target; this kit does not change the interpreter installation.
Do not assume the unit's working directory will supply the package.

After the new-code controller finish, complete old-params re-pin and exactly
one old-controller loop are established through `rollback-loop.sh`, quiesce
**all jobs sharing this package**, including the hourly weights writer, links
and site-pull. Stop/disable only their existing scheduler triggers during the
replacement; do not start a second weights writer or alter its unit. Then let
the directory owner rsync the complete staged package and verify its manifest:

```bash
# Replace only the package, after quiescing every job that imports it.
# Existing installed-package destination must be confirmed by the check above.
rsync -a --delete "$STAGE/sn120_v2/" /home/const/affine/v2/sn120_v2/

# Restore with the unit's existing state-dir and other read-side arguments;
# omit only --wallet for this no-chain-write check. Do not add --force.
/home/const/affine/v2/.venv/bin/python -m sn120_v2.weights_writer \
  --state-dir /OPERATOR-SELECTED/validator-state \
  --from-store --network finney --dry-run
```

Inspect the exact `status: "would set"` result, resolved package path and
weights before re-enabling only the original hourly writer and the other
quiesced jobs. Preserve the existing unit and arguments for real runs, its
transaction cursor/state directory, UID mapping, hourly cadence, six-hour
contribution half-life and last valid assessment through learning/evidence
outages. Never enable burn or owner-only fallback weights. `--dry-run` needs no
wallet and writes no chain weights, writer state or lock. This package does not
stop, restart or deploy any service; the boxed procedure is operator-only.

`rollback-validator-weights` remains available only as an optional diagnostic
for a separate old checkout through `rollback_cli.py`; it is not a solution for
the immutable `python -m` systemd unit.

`qualify-old-validator.py` and the complete `qualify-rollback.py` fixture prove
the physical package against the post-repin journal. The real module entry
point returns `would set`, with zero chain submissions, an unchanged journal
and no writer state/lock. The fixture explicitly selects the staging directory
with `PYTHONPATH`; only an external `sitecustomize.py` supplies its in-memory
chain. It does not test the production interpreter's package resolution, live
chain submission or live validator state.

## Operator environment and preservation

Use the operator's existing environment and the explicit environment settings
for `DEDUP_V2`, `DEDUP_HERE`, `DEDUP_STATE`, `DEDUP_WORK_DIR`, `DEDUP_PY`,
`DEDUP_PUBLIC`, `DEDUP_ENV_FILE`, reserve/generator paths and loop settings.
`GATE_WANT` defaults to `expect.json["gate_round"]`; set it only for an explicit
alternate gate whose successor is still `expect.json["first_health_round"]`.
Defaults in `dedup_operator.py` and `expect.json` describe the intended host;
inspect and override them rather than assuming a path.

Keep credentials in 1Password or the operator environment. Never put them in
the kit, plan, command line, logs or git. `set-params`, loop launch and
store-evidence inspection inherit only the operator's explicit environment.
Do not print it.

The driver refuses competing king-x2, genfeed or dedup deployment processes.
It must not stop, replace or reseed ingestd, `reserve-live`, the generator
inbox, the prior index or generator state. The ingest service needs a stable
operator service/launcher path across checkout; a command path embedded in a
mutable checkout is not preservation evidence. R2 is durable storage. Preserve
sealed evidence and do not turn machine cleanup into a rollback mechanism.

This kit does not modify the hourly best-assessment writer or any weights
service. Only an already authorized, separately reviewed cutover should use
`autodeploy` against production; the normal path here is the offline dry run.

## Real CLI entry points

```bash
# Offline: refuses a dirty kit unless DEDUP_DRY_DEVELOPMENT=1 (which suppresses
# the final PASS). D must be a fresh directory under /home/bb/scratch.
D=/home/bb/scratch/dedup-dry-normal bash v2/tools/dedup-deploy/dry-run

# Failure path: crash immediately after opening the first health round, finish
# it under new code, then restore compatibility-shimmed old code and one loop.
DEDUP_DRY_ABORT_AFTER_OPEN=1 D=/home/bb/scratch/dedup-dry-abort \
  bash v2/tools/dedup-deploy/dry-run

# Additional evidence recovery: a durable sealed result exists before the
# undecided crash; the same new-code run judges that result on restart.
DEDUP_DRY_STORE_RESULT_AFTER_OPEN=1 D=/home/bb/scratch/dedup-dry-sealed \
  bash v2/tools/dedup-deploy/dry-run

# A real round_settled record exists before the simulated process crash.
DEDUP_DRY_DECIDED_OPEN=1 D=/home/bb/scratch/dedup-dry-decided \
  bash v2/tools/dedup-deploy/dry-run
```

The normal dry run uses real controller, admission, worker/result, CLI,
journal, stop-after-round and public-rotation code over an isolated copied
fixture. Training/evaluation and wall-clock behavior are deterministic local
seams; this is not a GPU throughput test. Merged worker results carry
deterministic tested-base proofs and the simulated install fold supplies the
matching base bytes. The real `assert_tested_result` and `assert_fold_matches`
checks remain mandatory; neither is bypassed. The fixture intentionally preserves
an expired copied `r000086` lease as a genuine `no_decision`; it does not
fabricate a clean close for that lease. Fresh predecessor rounds are driven
through real worker/result behavior.

Public checks authenticate the journal-named `Rotation`, active bytes and IRT
snapshot. They never rely on `pool/state`; private `public/pool/state` is
excluded from the dry public projection. HTTP 404, 5xx and timeout publication
delays are retryable. Schema, digest and integrity failures are not.

A deployment attempt remains stopped and watchdog-disarmed when recovery
cannot be proven. `autodeploy` requires both the explicit `go` and
`go-confirmed` marker files before stopping a loop. Qualification creates only
disposable markers inside its isolated scratch fixture; never create either
marker in an operator deployment directory as part of qualification.

## Isolated copied-production-journal check

First copy the supplied journal to a new non-symlink state directory owned by
the operator. Do not use the source journal or a live state directory. Keep
the reviewed full params file outside that state directory.

```bash
python -I v2/ops/dedup_isolated_set_params.py \
  --state-dir /OPERATOR-SELECTED/COPY \
  --params /OPERATOR-SELECTED/new-params.json
```

The real production assembly is retained, but `SN120_V2_ISOLATED=1` selects a
new local store before any R2 construction or mirror catch-up. A write guard
refuses network connections, R2 construction, journal mirror pushes and local
store writes outside that isolated store. Open-round refusal is checked before
`Journal()` appends a fence. A refused probe leaves the copied journal
byte-identical.

The local qualification source journal had SHA-256
`1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca`, 574
records, and an open `r000088`. A separate copy was legitimately advanced to
IDLE before the guarded transition: 577 records became 579 and the params
changed `bba7a409… -> d5979c30…`. The effective diff was exactly
`dedup_enabled`, `train_seed_scope`, `z_bar`, `eval_merged` and
`recipe.code_digest`. No production journal or R2 object was written. See
`QUALIFICATION.md` for the actual evidence and limitations.

## Frozen inputs: separate, GET-only operator fetch

`frozen_inputs.py` derives private frozen-batch and result/LoRA references for
`r000085`–`r000088` from the supplied hash-chained journal. It performs no
production fetch while making a plan. The fetch command permits only S3
`GetObject(Bucket, Key)`, bypasses the normal R2 list preflight, and has no
list, put, copy or delete capability.

Run this **separately in an operator-selected production worktree**, with
credentials already in its environment. Do not run it from this qualification
session or against the copied state directory.

```bash
KIT=/OPERATOR-SELECTED/dedup-deploy
V2=/OPERATOR-SELECTED/production-worktree/v2
JOURNAL=/OPERATOR-SELECTED/supplied-journal.jsonl
OUT=/OPERATOR-SELECTED/private-dedup-inputs

python -I "$KIT/frozen_inputs.py" \
  --journal "$JOURNAL" \
  --journal-sha256 1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca \
  --root "$OUT" --params "$KIT/new-params.json" --v2-root "$V2" \
  --output "$OUT/plan.json"

python -I "$KIT/frozen_inputs.py" fetch \
  --plan "$OUT/plan.json" --target "$OUT/objects" --v2-root "$V2"

# The plan prints exact round-specific commands. Do not add --lora unless every
# selected seat has a complete, fetched three-file LoRA directory.
python -I "$KIT/frozen_input_dryrun.py" --v2-root "$V2" \
  dedup-dryrun --batch REPEAT_PER_SEAT --params "$KIT/new-params.json"
```

`OUT` must be a new private parent path chosen by the operator; the fetch
target must not already exist. For first-seen evidence, use a private plan
with `--fetch-first-seen` on a separate results-only fetch, then regenerate the
plan using the fetched local ledger. Alternatively pass a reviewed local
`--first-seen` ledger. Never invent missing timestamps.

LoRA directories are required to contain exactly the needed evidence files:
`adapter_model.safetensors`, `adapter_config.json`, and `train_report.json`.
Pending directories may have been removed by a normal close. A journal-derived
plan with zero candidates means that no previously fetched sealed results were
available to derive them, **not** that sealed results are absent in production.
Use the documented results-only flow to establish that separately; no pending
LoRA directory is promised to exist.

## Standalone rollback qualification

Prepare separate operator-owned new- and old-code worktrees/archives. Never
use a production checkout as the fixture input.

```bash
python -I v2/tools/dedup-deploy/qualify-rollback.py \
  /OPERATOR-SELECTED/new-scratch-output \
  --new-v2 /OPERATOR-SELECTED/new-worktree/v2 \
  --old-v2 /OPERATOR-SELECTED/old-c5252865/v2 \
  --python /OPERATOR-SELECTED/python
```

This uses a real new-controller synthetic fixture, tests bare-old rejection,
re-pins the old params after historical new params, proves refusal before an
open-round fence, and starts one bounded old loop through the compatibility
loader. Its PASS lines are limited to those checks.

`QUALIFICATION.md` records the actual commands, results and remaining limits.
