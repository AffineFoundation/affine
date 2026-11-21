# AffineII dedup release kit

This is an **operator-only release kit**, not a deployment. It does not grant
permission to arm a round, change weights, or access production. Building and
qualifying this kit makes no production change.

## Approved release and rehearsal boundary

The approved release enables dedup, round-scoped training seeds and charged
null updates, raises the crown threshold to **`z_bar = 1.5`**, and makes **both
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
| New commit | `5e9df83842f34cc3204d8003169d72f5ed4121ab` |
| New GPU source digest | `b68a3005fd9837e9474349ca086088d9a7649406772e141b169952e7ef7f6e01` |
| New canonical params hash | `8543962a34e8fecb0da3663e05b0c222ae612cbe4a8cef44fedf24f42f0504e5` |
| `old-params.json` file-byte SHA-256 | `86c9cd5b5e696c6c10411fd828b7e26a641d06bd39f94a4ea98eb69de3e78f56` |
| `new-params.json` file-byte SHA-256 | `f8073812849133f5d0e81728ef361d3e98b0466b4b76e3b6259a4a858d891d4b` |

A canonical `Params.params_hash` is not the SHA-256 of a pretty-printed params
file. The new params intentionally retain the old worker recipe code digest;
the release's own GPU source digest is a different identity.

The effective production-copy changes are exactly:

```text
dedup_enabled        false -> true
train_seed_scope     batch -> round
z_bar                1.0 -> 1.5
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
The complete seven-field dedup record plus `z_bar=1.5` is required; the
incomplete legacy file is never parsed by new code.

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
   old behavior without changing historical records. The reviewed historical
   `z_bar=1.5` override is removed along with the seven dedup fields, restoring
   the old `z_bar=1.0` behavior; unknown or newer parameters remain refused.
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
seams; this is not a GPU throughput test. The fixture intentionally preserves
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
changed `bba7a409… -> 8543962a…`. The effective diff was exactly `dedup_enabled`,
`train_seed_scope` and `z_bar`. No production journal or R2 object was written. See
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
