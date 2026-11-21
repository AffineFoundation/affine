# AffineII dedup release kit

This is an **operator-only release kit**, not a deployment. It does not grant
permission to arm a round, change weights, or access production. Building and
qualifying this kit makes no production change.

## Current decision: do not arm `r000089`

`r000089` is an **offline rehearsal gate only**. The reviewed new parameters
would cut the live `r000087` king:

- UID 488 has `global_norm_cap = 0.01`, below `min_step_norm = 0.02`; admission
  returns `NULL_UPDATE`.
- The 58-rollout batches fail `min_signal_rollouts = 64`.

A cut is not itself a health failure. The health gate fails if dedup and
null-update cuts remove every selected batch, or if a required healthy close
does not occur. Do not manufacture acceptance by lowering the floors.

The kit targets the later `r000089` boundary only in an offline fixture. It
requires a separate explicit operator `go` and `go-confirmed` decision before
any real use. No live `r000089` arming is authorized.

## Frozen identities

| Item | Value |
| --- | --- |
| Old commit | `c525286580c84d795b3ca7cbdf2302e96fce31ba` |
| Old GPU source digest | `5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47` |
| Old canonical params hash | `bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a` |
| New commit | `e96566341dfa463d0a2f1853f038dac9877b2c9a` |
| New GPU source digest | `b68a3005fd9837e9474349ca086088d9a7649406772e141b169952e7ef7f6e01` |
| New canonical params hash | `0b2f390b0e416e9b1ff2523b81976777d827c1da92238f9147cbae7d8472a5f6` |
| `old-params.json` file-byte SHA-256 | `86c9cd5b5e696c6c10411fd828b7e26a641d06bd39f94a4ea98eb69de3e78f56` |
| `new-params.json` file-byte SHA-256 | `5fd2ea0c4cabca97e0edf999c0381910e7a1b79ce2cef76ac88141410cd00c10` |

A canonical `Params.params_hash` is not the SHA-256 of a pretty-printed params
file. The new params intentionally retain the old worker recipe code digest;
the release's own GPU source digest is a different identity.

The effective production-copy changes are exactly:

```text
dedup_enabled        false -> true
min_signal_rollouts  1 -> 64
min_step_norm        0.0 -> 0.02
train_seed_scope     batch -> round
```

Expected rehearsal constants are `batch_cap=6`, `n_tasks=2000`, two null
strings per leased box, `active_size=10000`, and `irt_draw_rule=sqrt_fisher`.
`r000090` is the first health round and `r000091` the second-open proof.

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
   old behavior without changing historical records.
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
changed `bba7a409… -> 0b2f390b…`. No production journal or R2 object was
written. See `QUALIFICATION.md` for the actual evidence and limitations.

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
