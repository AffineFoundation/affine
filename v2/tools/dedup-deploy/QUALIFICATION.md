# Dedup release qualification record

This record describes **offline qualification only**. It does not report a
production deployment, production health, or authorization to arm a round.
Final committed-kit rehearsal results are recorded only from actual output;
no development completion line is promoted to a final PASS.

## Current approved release

The approved design enables dedup, `train_seed_scope=round`, charged null
updates and merged-candidate evaluation, raises the crown threshold to
`z_bar=1.5`, and makes both admission floors no-ops:

```text
dedup_enabled          true
train_seed_scope       round
dedup_rollout_overlap  0.5
dedup_cos_shared_seed  0.135
dedup_cos_own_seed     0.002
min_signal_rollouts    1
min_step_norm          0.0
z_bar                  1.5
eval_merged            true
recipe.code_digest     28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a
```

The default offline gate is `r000091`, with `r000092` as the first health round
and `r000093` as the second open. `GATE_WANT` is overridable, but `autodeploy`
refuses a gate whose successor differs from `expect.json`'s required health
round. Qualification and a PASS line do not themselves authorize production.
The separate `go` and `go-confirmed` operator actions remain mandatory.

### Legacy-equivalence proof and minimal release correction

At old commit `c525286580c84d795b3ca7cbdf2302e96fce31ba`, admission required
at least one active rollout but did **not** impose `recipe.minibatches` as an
admission floor. The trainer's `minibatches_of` returns
`range(min(minibatches, len(mixed)))`: fewer active rows produce fewer
nonempty minibatches. Therefore setting `min_signal_rollouts=1` is not enough
at the previous release candidate `e9656634`, which incorrectly used
`max(min_signal_rollouts, recipe.minibatches)`.

The original minimal correction was commit
`8afd574736f6a2bac351cd3aa0edcd6295c35e44`, directly on
`e96566341dfa463d0a2f1853f038dac9877b2c9a`. It changes only
`v2/sn120_v2/admission.py` and `v2/tests/test_min_rollouts.py`, removing the
implicit minibatch floor and retaining the explicit configured floor. The
combined lineage retains that functional correction through `39cc17cf`; the
only admission-code delta from `8afd5747` is explanatory rule text.

Release commit `50afaca32ea982796728a0b07cb6e3c579ef378b` combines the dedup
release, no-op-floor publication and AFF-5 PR #310 head
`1fa67a9770884eb679ffeb91fd133ac49575d117`. Its GPU tree has no diff from that
AFF-5 head. Merged evaluation uses the install arithmetic
`round_bf16(fp32 king master + LoRA update)`; the GPU source manifest is:

```text
28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a
```

The offline diagnostic `dedup.dryrun` now reports the same explicit
`min_signal_rollouts` floor as `admission.check_batch`. It no longer reports
`max(min_signal_rollouts, recipe.minibatches)`. This non-GPU diagnostic change
does not change the release commit, GPU digest or params identity. A regression
uses two active rollouts with `minibatches=71`, `min_signal_rollouts=1` and
`min_step_norm=0.0`; both real admission and the diagnostic accept it.

At `min_step_norm=0.0`, static admission accepts every valid nonnegative cap
because `cap >= 0.0`. Judging rejects no finite stored norm because a
nonnegative norm cannot be `< 0.0`; zero and small positive updates remain
valid. Non-finite stored norms still become charged `null_update` decisions.
Old `c5252865` instead raised `BatchRejected("training gave a loss or an
update that is not finite")` for a non-finite training loss or step norm. The
new release intentionally seals tagged non-finite stored evidence for judging;
it is not scored as an ordinary update.

## Frozen params identities

The generated full params artifact is byte-identical to `new-params.json`:

```text
source                 v2/ops/dedup_params_step.py -> params_dedup_full.json
file SHA-256           4acb73b3cf45e2240c147150e854ac27ab3f80851903b7adbb15b08452bc2784
canonical params_hash  d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c
old file SHA-256       86c9cd5b5e696c6c10411fd828b7e26a641d06bd39f94a4ea98eb69de3e78f56
old canonical hash     bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a
```

A canonical `Params.params_hash` is not the SHA-256 of the pretty JSON file.
The full params pin the combined worker recipe
`28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a`,
which enables merged evaluation. The old params pin worker recipe
`5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47`.

Independent generation from the reconstructed original IDLE copied journal at
`/home/bb/scratch/dedup-combined-50afaca3-v1/from-original-idle.json` was
byte-identical to the full params artifact. The five effective changes are
`dedup_enabled`, `train_seed_scope`, `z_bar`, `eval_merged` and
`recipe.code_digest`. All disabled-floor values remain explicitly present.

The earlier `5fd2ea0c…` file contained `min_signal_rollouts=64` and
`min_step_norm=0.02`, with canonical hash `0b2f390b…`. The intermediate
no-op-only file `eb6bad85…` had canonical hash `ea3ca1f1…` and `z_bar=1.0`.
Neither is the approved z=1.5 artifact.

## Copied real-journal isolation evidence

The supplied source is `/home/bb/inbox/dedup-prodstate/journal.jsonl`:

```text
SHA-256 1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca
574 records, sequence 0..573, r000088 open
```

The source SHA-256 was unchanged after testing. Combined-release isolation
artifacts are under `/home/bb/scratch/dedup-combined-50afaca3-v1`.

An independent copied open journal was tested with the real guarded isolation
script. The refusal returned exit 2 and left its journal byte-identical at the
source SHA-256 above. Its captured stdout and stderr are
`open-refusal.stdout` and `open-refusal.stderr` in that artifact directory:

```text
a round is open; set the parameters when it has closed (a decided round takes new digests only)
```

A distinct copy was legitimately advanced to IDLE as `no_decision`, without
reading or discarding a sealed result. Its 577 records became 579 during the
guarded transition, which produced:

```json
{
  "after_params_hash": "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c",
  "before_params_hash": "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a",
  "exit": 0,
  "isolated": true,
  "journal_records_after": 579,
  "journal_records_before": 577,
  "live_journal_writes": 0,
  "network_connections": 0,
  "r2_writes": 0
}
```

The effective changes are exactly:

```json
{
  "dedup_enabled": {"before": false, "after": true},
  "train_seed_scope": {"before": "batch", "after": "round"},
  "z_bar": {"before": 1.0, "after": 1.5},
  "eval_merged": {"before": false, "after": true},
  "recipe.code_digest": {
    "before": "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47",
    "after": "28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a"
  }
}
```

The guard blocks socket connections, R2 construction, journal mirror pushes
and local-store writes outside the selected isolated store. The isolated
store contained only the copied close's `public/head.json`, one versioned head
and `public/rounds/r000088/round.json`; no journal object was created there.
No production journal, network or R2 write occurred.

## Production parameter route

Production `autodeploy` uses the ordinary journaled `set-params` CLI against
the real state and `sn120_v2.assembly:stack`, after stopping the sole loop and
checking again that it is not alive. It does not set `SN120_V2_ISOLATED`.
`Journal` synchronizes and appends its leader/`params_set` records to the one
existing R2 journal prefix and the local journal; no alternate state or mirror
is created. Consequently this route creates no stray R2 journal copies.
The isolated wrapper is only for copied-state qualification, never a command
to point at production state.

## Regression evidence for the combined release

Earlier z=1.5-only focused 30-test and broader 97-test runs passed against the
superseded `5e9df838`/`8543962a…` proposal. They are **not current
combined-release or final-suite evidence**.

The combined focused params/operator/publication/min-rollouts selection passed
27 tests with exit 0. After the merged-fixture and diagnostic-floor corrections,
a combined focused controller/operator/params/publication/isolation/admission
selection passed 46 tests with exit 0. Its log is
`/home/bb/scratch/dedup-combined-focused-after-fix-v2.log`, with complete output
`.............................................. [100%]`. The preceding four-test
failure was a dropped `layout` import in the test module, not a controller
failure; restoring that import fixed all four. A later expanded selection of
frozen inputs, isolation, params, operator, controller, minimum-rollouts,
merged evaluation and publisher modules passed 80 tests with exit 0. Its log
is `/home/bb/scratch/dedup-kit-combined-focused-final-v1.log`. These focused
runs do not replace the required final complete suite against the final
committed source.

AFF-5 reported its uninterrupted suite at `1fa67a97`: **1 failed, 1,691 passed,
1 skipped**. Its actual terminal traceback identifies
`test_public_queue_files.py::test_every_key_that_the_snapshot_had_before_is_the_same_byte_for_byte`,
not the different public-queue test suggested by an earlier positional
estimate. The expected compatibility hash is `a5ffb03f…`; the actual hash is
`7b07d90a…`. AFF-5 and this qualification session independently reproduced the
same test failure and hashes on an untouched archive of `4d337954`; the
combined-tree public-queue module has the same single failure. This supports
its classification as pre-existing rather than an AFF-5 regression. It remains
a recorded suite failure, not a green suite or a justification for weakening
the public-queue contract. The local release-base log is
`/home/bb/scratch/dedup-base-4d337954-public-snapshot-v1.log`; the combined
module log is `/home/bb/scratch/dedup-kit-public-queue-isolated-v1.log`.
The earlier AFF-5 Generator-recovery failures did not recur in its final run.

The clean `50afaca3` complete suite log is
`/home/bb/scratch/dedup-combined-release-50afaca3-full-suite-v1.log`. It reaches
100% with one skip and the known public-snapshot failure, followed by the
failure summary. It has **no terminal pass/fail count line**; do not infer an
exact local total from the progress display or AFF-5's separate count.

The corrected integrated-kit suite log is
`/home/bb/scratch/dedup-kit-combined-full-suite-v2.log`. It also reaches 100%
with one skip and no terminal count line. Its two failure-summary entries are
the same public-snapshot failure and
`v2/tests/test_no_legacy.py::test_v2_code_imports_no_legacy_module`. The latter
reports explicit `sys.path` use in `frozen_input_dryrun.py:24`,
`frozen_inputs.py:462`, `frozen_inputs.py:489` and `rollback_cli.py:120`.
Those paths predate the physical-validator change, but the integrated tree is
still **not green**. Neither failure is omitted or counted as a qualification
PASS; the public-snapshot failure has separate pre-existing-base evidence.

The corrected assembly module passed all 20 tests with exit 0 in
`/home/bb/scratch/dedup-kit-assembly-explicit-legacy-and-groups-v1.log`. An
earlier ordinal estimate incorrectly attributed the integrated-suite failure
marker to the round-seed test; the exact two-test run and assembly traceback
identified the smoke-copy test instead. The production set-params guard
correctly rejected its incomplete fixture. The test now serializes all seven
explicit disabled legacy dedup/seed fields and requests 32 groups explicitly,
preserving its original 64-rollout smoke coverage independently of the
release's no-op admission floor. This module result does not replace the final
complete suite.

Coverage includes no implicit minibatch floor, finite zero/small norms at the
zero floor, charged non-finite stored updates, the exact five effective params
changes, explicit seven-field dedup serialization plus `z_bar=1.5` and
`eval_merged=true`, merged tested-base/install-fold identities, stopped-loop
isolation, public-only authenticated checks, 404/5xx/timeout retries, and
health using journal decision cuts plus `run.log`/public `eval_wall_s`.
Publication tests verify the dedup rules, round training-seed formula, both
no-op floors and the literal `z=1.5` final/early-look rules in `llms.txt`, the
`z_bar=1.5` value in `mining.json`, merged-evaluation disclosure, and a clean
`publisher.check_pair` result.

## Combined working-kit rollback requalification

Earlier z=1.5-only rollback development v3 and committed-kit
`a8f5b5dcdaac2421c44b7ac399d40f0798874f54` evidence used `8543962a…` and
pre-merged-release code. Those six-PASS records are superseded by the combined
development run below; they are not current committed-kit PASS evidence.

The combined fixture supplies deterministic tested-base bytes and matching
simulated install folds. It calls the real `gpu.merged.assert_tested_result`
and `assert_fold_matches` checks, including Stage-B duplicate scoring, merged
winner promotion and fold publication. Neither production proof is bypassed.
The fixture completed `null_update`, `winner`, `duplicate` decisions and an
applied close. Early attempts with recursive submit wrappers, incomplete
`eval_merged` recognition and a stale old-worker expectation failed and are
not counted as qualification.

The corrected **working-tree development v19** command completed with exit 0:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -I \
  v2/tools/dedup-deploy/qualify-rollback.py \
  /home/bb/scratch/dedup-combined-rollback-development-v19 \
  --old-v2 /home/bb/scratch/dedup-old-c5252865/v2 \
  --python /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python
```

Its log is
`/home/bb/scratch/dedup-combined-rollback-development-v19-qualification.log`.
The actual seven PASS lines were:

```text
ROLLBACK QUALIFICATION PASS: real new-controller fixture: failed lease, duplicate-only, null-update, Stage-B duplicate, sealed nonfinite evidence
ROLLBACK QUALIFICATION PASS: closed dedup history replays with charged duplicates/null update and uncharged failed-lease survivors
ROLLBACK QUALIFICATION PASS: bare old code separately rejects new params, duplicate/null_update, new round fields and tagged nonfinite values
ROLLBACK QUALIFICATION PASS: old-code compatibility CLI re-pins bba7a409 after historical d5979c30; unknown/newer params refused
ROLLBACK QUALIFICATION PASS: native release reads post-rollback dedup history at old bba7a409/z1.0; validator weights dry-run would set without shim or chain write
ROLLBACK QUALIFICATION PASS: undecided/decided open dedup rounds refuse before fence; local and mirrored journals unchanged
ROLLBACK QUALIFICATION PASS: unmodified old loop uses the compatibility executable and exits after exactly one bounded start
```

The stable committed-kit rollback rerun at `bef1e0a8` also completed with exit
0 and emitted the same seven PASS lines above. Its actual output is
`/home/bb/scratch/dedup-combined-rollback-committed-v2.log`. An earlier
rollback-v1 attempt overlapped the history-preservation merge and failed while
reading conflict-marker-contaminated generated params; it is invalid evidence
and is not counted as a replay failure.

The compatibility loader validates the exact full `d5979c30…` pin,
including `eval_merged=true` and the combined worker digest, before exposing
old behavior. It removes the seven dedup fields, historical `z_bar=1.5` override
and `eval_merged=true`, then translates only the reviewed combined worker digest
to the reviewed old worker digest **in memory**. Journal bytes are never rewritten.
The replay assertion checks final old `state.params.z_bar == 1.0`.

**Bare `c5252865` cannot replay dedup history.** Rollback is code-first:
finish open work under new code, check out `c5252865`, apply the complete old
`bba7a409…` params through the hashed compatibility loader, and start exactly
one old-code loop. Sealed evidence is never deleted to make old code accept
history.

### Validator rollback reader

Jacob requires rollback to restore the validator too. The validator therefore
uses old `c5252865` plus the same reviewed compatibility behavior, rather than
staying on native release code. Native release readers remain compatible with
this journal, but that is not the required rollback configuration.

The immutable systemd invocation `python -m sn120_v2.weights_writer` cannot use
the earlier external runtime loader. The final route is therefore a physical
package staged by `restore_validator.sh`: all 123 exact old source files plus
seven physically overlaid compatibility modules, including `journal.py` with
the pre-fence open-dedup-round guard. `weights_writer.py` and the old package
`__init__.py` remain unchanged. The stager pins every input and output hash and
never reads or writes production state.

The complete working-kit physical rollback v5 completed with exit 0 and all
eight checks. Its actual output is
`/home/bb/scratch/dedup-combined-rollback-physical-v5.log`. The focused physical
proof v4 completed with exit 0 and the following lines:

```text
RESTORE VALIDATOR PASS: complete c5252865 plus seven physical compatibility files; 124 source files staged
ROLLBACK QUALIFICATION PASS: c5252865 physical package runs python -m sn120_v2.weights_writer --dry-run: would set; no runtime shim, zero chain writes, journal unchanged
kings: read from the journal file
{"block": 5000, "blocks_since_last_write": 5000, "dry_run": true, "kings": 2, "kings_from": "the journal file", "last_write_block": 0, "status": "would set", "validator_hotkey": "5SimValidator", "validator_registered": true, "weights": {"1": 0.5, "7": 0.5}}
{"chain_submissions": 0, "final_params_hash": "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a", "journal_unchanged": true, "kings": 2, "old_modules_physically_overlaid_count": 7, "runtime_import_shim": false, "weights": {"1": 0.5, "7": 0.5}, "weights_dry_run": "would set", "z_bar": 1.0}
```

The focused log is `/home/bb/scratch/dedup-physical-validator-proof-v4.log`.
This v5/v4 record predates the final archive-manifest and direct physical
open-round tests; the final command below is the current committed-kit proof.

The reader proves exact historical `d5979c30…` identity through
`dedup_compat.digest_params`; compatibility-translated old `Params` exposes
`bba7a409…`/`z_bar=1.0` and has no dedup/seed attributes. It decodes historical
charged decisions and sealed non-finite evidence. An in-memory chain returned
`would set`, with zero submissions, unchanged journal and no writer state/lock.
A missing eagerly imported `canonical.py` also makes the ordinary module entry
point fail rather than silently borrowing another installed package.

The ordinary module uses `PYTHONPATH=<stage>:<external-fixture>` in this offline
proof. That explicitly selects the staged source; the only `sitecustomize.py`
is external and supplies the in-memory chain. No fixture hook is included in
the staged package. A real venv does not automatically put an adjacent source
checkout on `sys.path`. On the box, the operator must prove that the existing
interpreter resolves `/home/const/affine/v2/sn120_v2/__init__.py` before using
file-only restoration. If it resolves elsewhere, that package-path installation
must be repaired or the actual target supplied; this kit does not alter it.

The deterministic tarball includes `sn120_v2/` and its complete
`restore-validator-manifest.json`. The final proof verifies every archived
byte against that directory. The staged package contains no `sitecustomize.py`,
bytecode or test hook. This is synthetic offline evidence, not a live chain,
production interpreter or validator-state test.

The final rollback command is:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -I \
  v2/tools/dedup-deploy/qualify-rollback.py \
  /home/bb/scratch/dedup-combined-rollback-physical-v8 \
  --new-v2 /home/bb/scratch/affine-dedup-release-combined/v2 \
  --old-v2 /home/bb/scratch/dedup-old-c525-v2 \
  --python /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python
```

Its log is `/home/bb/scratch/dedup-combined-rollback-physical-v8.log`.
This working-tree run completed with exit 0 and **nine rollback PASS lines**:
the eight original checks plus the direct **physical** `journal.py` check for
undecided and decided open dedup rounds. Its core validator lines are:

```text
RESTORE VALIDATOR PASS: complete c5252865 plus seven physical compatibility files; 124 source files staged
ROLLBACK QUALIFICATION PASS: c5252865 physical package runs python -m sn120_v2.weights_writer --dry-run: would set; no runtime shim, zero chain writes, journal unchanged
ROLLBACK QUALIFICATION PASS: physical journal.py refuses undecided/decided open dedup rounds before fence; local and mirrored journals unchanged
kings: read from the journal file
{"block": 5000, "blocks_since_last_write": 5000, "dry_run": true, "kings": 2, "kings_from": "the journal file", "last_write_block": 0, "status": "would set", "validator_hotkey": "5SimValidator", "validator_registered": true, "weights": {"1": 0.5, "7": 0.5}}
```

The v8 deterministic tarball SHA-256 is
`31d2b376fc2ec18a5e49d840ab2c7be0011a35d1d0593ab1c657d83924b14591`;
its staged-manifest SHA-256 is
`232ca5b9de0b7a05c85d76f28afa3f963bcc03c0d1eff488a97a00ff7ccd258e`.
The generated source manifest verifies all 124 source files. The final
committed-kit command/output below supersedes v8 only as provenance evidence;
source and overlay identities are unchanged.

Development v6 stopped at an incorrect shell argument-count check for
`--tar PATH`; v7 stopped at a package-relative/archive-path mismatch in its new
archive assertion. A focused command reusing v7's staging path was correctly
refused because outputs must be new. None is qualification PASS evidence.
The fresh v8 run fixed the wrapper and archive assertion and passed.

For the physical journal, both local journals and all mirrored store objects
stay byte-identical, with no store writes before the fence refusal. The archive
is checked for all 124 source files plus its staged manifest. This is not
The exact committed-kit run at `70cd06f5262f340463b882f72d70490673fcba11`
completed with exit 0 and the same nine rollback PASS lines. Its actual log is
`/home/bb/scratch/dedup-physical-validator-committed-v1.log`. The committed-kit
`restore_validator.sh` also produced the same deterministic archive and staged
manifest, byte-for-byte. Its final operator artifact is:

```text
/home/bb/scratch/restore-validator-c5252865-kit70cd06f5.tar.gz
SHA-256 31d2b376fc2ec18a5e49d840ab2c7be0011a35d1d0593ab1c657d83924b14591
staged manifest SHA-256 232ca5b9de0b7a05c85d76f28afa3f963bcc03c0d1eff488a97a00ff7ccd258e
```

The final committed-kit SHA-256 inventory is
`/home/bb/scratch/dedup-physical-validator-kit70cd06f5-file-hashes.json`.
No production access, systemd change, website publication or chain-weight
write occurred during staging or qualification.

Native release `50afaca3` independently reads the same post-repin journal. Its
`Params.from_dict` supplies effective legacy defaults for omitted fields and
its result types decode historical decisions and sealed evidence. The separate
native-reader PASS remains useful compatibility evidence but no longer defines
the required validator rollback path.

`qualify-rollback.py` runs the post-repin journal through a fresh `-I`
interpreter with only the native new-v2 source on its import path. It asserts
the historical `d5979c30…` params canonical hash, all dedup fields,
`eval_merged=true`, the combined worker digest and `z_bar=1.5`, new round
fields and charged duplicate/null-update decisions, a tagged non-finite
sealed report, and final IDLE params `bba7a409…`, `z_bar=1.0`,
`dedup_enabled=false`, `train_seed_scope=batch`. It then invokes the release
weights-writer dry-run with an in-memory chain. The actual combined report was
`status="would set"`, with zero submissions, an unchanged journal and no
writer state/lock. This synthetic fixture does not test a live chain write or
a production journal mirror.

No external `/validator-ready` or `/validator-rollback-ready` marker gate was
added. The cancelled proposal would have made the new or old loop depend on a
separate box watcher; the existing `APPLY OK`, `PASS:` and `ROLLBACK` lines
remain evidence markers only, not readiness dependencies.

## Frozen-input planning evidence

The offline plan summary is
`/home/bb/scratch/dedup-frozen-inputs-summary-final.json`:

```text
journal SHA-256  1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca
rounds           r000085, r000086, r000087, r000088
batch_count      24
lora_candidates  0
```

Zero candidates means the planner had no previously fetched sealed results
from which to derive LoRA files. It does not establish their absence in
production. Fetching remains a separate operator-only GET-only operation; no
production object or credential was accessed here.

## Current rehearsal fixture

The dry run starts from the copied `r000086` open predecessor, preserves its
expired lease as genuine `no_decision`, re-pins the full old params, then
executes real old-code rounds `r000087` through `r000090` before opening the
offline `r000091` gate. After apply it requires the real new-code `r000092`
health close and `r000093` second open.

All eight selected no-op-floor fixtures are admitted, including 58 active
rollouts with `minibatches=71` and a `global_norm_cap=0.01` batch. Expected
admission cuts are zero. The selected decision plan has one normalized-output
duplicate, one independently simulated tagged non-finite stored update and
six surviving batches. A finite zero norm is not used to simulate a null
update. Public checks authenticate rotation, active bytes and IRT snapshot;
`pool/state` is absent. Generator/reserve bytes and the sole-loop behavior are
checked independently of the training/evaluation simulation.

The real controller, admission, worker/result, CLI, journal and
stop-after-round paths are used. Training/evaluation and wall time are local
deterministic seams, so this is not a production GPU or throughput test.

### Failed development attempts retained as limitations

Working-kit normal v1 and v2 did not reach apply: v1 exposed a stale
compatibility manifest pin; v2 copied the kit before that manifest update.
Neither is a PASS. The manifest now pins the actual compatibility file. The
initial no-op dry-run round rewrite also omitted `r000087`–`r000088`, and its
stored-null fixture used finite `step_norm=0.0`; both were corrected before
fresh normal and abort rehearsals. Do not count attempts before those fixes
as qualification.

## Historical evidence from the superseded floor kit

The following records apply to the earlier `e9656634`/`0b2f390b…` proposal
with 64-rollout and 0.02-norm floors. They explain development coverage but
are **not evidence for the current params identity or default gate**.

Development v17 and v18 completed the old normal rehearsal with offline gate
`r000089`, health `r000090`, and second open `r000091`. Their actual health
included `duplicate=1`, `null_update=1`, admission `min_rollouts=1` and
`null_update=1`, with four of six selected batches surviving and a
1,234-second evaluation wall. Their only top-level completion line was
`DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no final PASS`.

Abort development v5 completed a genuine undecided-open crash, new-code finish,
stop-obeyed confirmation, compatibility-shimmed old checkout/params and one
bounded old loop. Sealed-result development v1 proved a durable result
survived an undecided crash and was judged on restart. Decided-open development
v2 proved `round_settled` survived a crash before `round_closed`, then the same
bounded code-first recovery completed. These runs printed only development
completion lines, not committed-kit PASS lines.

Earlier old-floor attempts v10–v16 and abort v1–v4 failed for fixture,
selection, timeout or harness reasons. None is counted as a qualification PASS.
Working-kit old-floor rollback runs v23–v26 passed their six named rollback
checks against `0b2f390b…`; the later z=1.5-only `8543962a…` runs and the
current combined `d5979c30…` development v19 supersede their params identity.

## Current combined development rehearsals and remaining qualification

The superseded z=1.5-only normal development rehearsal completed with exit 0
and emitted `DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no
final PASS`. Its health was 2,000 scored tasks, one duplicate, one charged
null update, zero admission cuts, no lease failures, and a clean `r000093`
second open. Its log is `/home/bb/scratch/dedup-z15-development-normal-v1.log`.
This is **superseded pre-merged fixture development evidence**, not a current
committed-kit PASS.

The first combined normal working-tree development attempt failed before
apply: an unconditional new-module import made the real old-code `r000087`
predecessor finish as `no_decision`, so every later old-code predecessor lacked
sealed results and the `r000091` gate correctly refused as not clean. Its
offline autodeploy log records `FAIL before go-live: gate close was not clean`;
it did not apply new code or params and emitted no PASS. The fixture now
imports merged proof checks only when `eval_merged` is active, while still
requiring both merged tested-base and install-fold checks in the new release.
This failed attempt and any subsequent development run are not final PASS
evidence.

Combined normal working-tree development v2 also failed before apply. Its
actual terminal traceback was `AttributeError: 'Params' object has no attribute
'eval_merged'` in the simulated `MergedSimFolder.fold` during old-code
`r000087`, after winner publication but before a clean close. The fixture now
uses `getattr(params, "eval_merged", False)` at that legacy boundary. The
production merged tested-base and install-fold guards remain mandatory under
new params. Normal development v3 was restarted from a fresh isolated root;
its outer log's early `REFUSED` line is the intentional competing-deployment
guard check, not the rehearsal's terminal result. It subsequently completed
with exit 0 and the terminal line
`DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no final PASS`.
Its real old-code `r000091` gate closed applied with 2,000 scored tasks and no
lease failures; apply pinned `50afaca3`, then `r000092` closed applied with
2,000 scored tasks, one duplicate, one charged null update, zero admission
cuts, evaluation wall 1,234 seconds, and no lease failures. Public checks
passed, and `r000093` opened in the same single loop. The log is
`/home/bb/scratch/dedup-combined-development-v3.log`. This remains development
evidence, not a committed-kit PASS.

The earlier normal-v1/v2 stale-manifest attempts described below predate these
combined merged-fixture runs; the names refer to different attempts.

Combined abort-after-open development v1 completed with exit 0 and terminal
`DRY DEVELOPMENT COMPLETE: bounded new-code finish and shimmed old rollback`.
After `50afaca3` applied and opened `r000092` under `d5979c30…`, the simulated
unhealthy-public response correctly caused `autodeploy exit 1 (no PASS)`. New
code finished the undecided `r000092` as applied without discarding its stored
result; compatibility-shimmed old `c5252865` and old params were restored, and
exactly one old-code loop started. Its log is
`/home/bb/scratch/dedup-combined-abort-development-v1.log`. This is development
recovery evidence, not a final committed-kit PASS.

At `2026-11-21T14:21:47Z`, both GitHub PR #310 and `refs/pull/310/head` still
resolved to `1fa67a9770884eb679ffeb91fd133ac49575d117`. The integrated release
is `50afaca32ea982796728a0b07cb6e3c579ef378b`, its GPU tree matches AFF-5,
and `c5252865` remains an ancestor. AFF-5's terminal report is recorded above.

### Committed-kit normal rehearsal

The clean committed-kit normal rehearsal launched from `be725d3a` completed
with exit 0. Its frozen kit is content-identical to the ancestry-preserving
`bef1e0a8` revision. It ran without `DEDUP_DRY_DEVELOPMENT`, compared all 24
kit files recursively, and preserved generator/reserve bytes. The actual
terminal output is `/home/bb/scratch/dedup-combined-committed-normal-v1.log`:

```text
2026-11-21T14:33:10Z gate r000091; old HEAD c5252865
2026-11-21T14:34:18Z journal gate r000091: yes applied n_tasks 2000 lease_failed 0 why []
2026-11-21T14:35:01Z APPLY OK: new HEAD 50afaca3; params applied through set-params CLI
2026-11-21T14:36:03Z first open r000092: OK active 10000 params d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c n_tasks 2000 snapshot sqrt_fisher eval_wall_s None cuts None admission_cuts None why []
2026-11-21T14:36:06Z first-open public PASS; same loop must close r000092 and open r000093
2026-11-21T14:36:28Z first new-code round r000092 close: yes applied n_tasks 2000 lease_failed 0 why []; second r000093 opened in same process
2026-11-21T14:36:29Z first-round health: OK active None params None n_tasks 2000 snapshot None eval_wall_s 1234 cuts {'duplicate': 1, 'null_update': 1} admission_cuts {'min_rollouts': 0, 'null_update': 0} why []
2026-11-21T14:36:29Z PASS: dedup 50afaca3; merged evaluation enabled; r000092 closed n_tasks=2000 with exactly 2 null strings x 2000 chars for every box, no lease_failed, eval wall 1234s; duplicate/null-update decision cut counts logged; both admission floors are no-ops; crown threshold z_bar=1.5; r000093 opened in the same single loop; public files healthy
generator and reserve bytes preserved
health evidence verified {'duplicate': 1, 'null_update': 1} {'min_rollouts': 0, 'null_update': 0} no-op admission fixtures accepted
2026-11-21T14:36:39Z DRY AUTODEPLOY PASS: real dedup/no-op-floor/z1.5 run/gate/apply/open/public/health; r000091 boundary was offline-only
```

### Committed-kit abort-after-open rehearsal

The clean committed-kit abort-after-open rehearsal completed with exit 0 at
`bef1e0a8`, without `DEDUP_DRY_DEVELOPMENT`, after comparing all 24 kit files
recursively. Its simulated unhealthy public response correctly produced
`autodeploy exit 1 (no PASS)`; the offline recovery rehearsal itself then
completed with the final PASS below. The undecided `r000092` was finished as
applied by reviewed new code without discarding sealed evidence, the
compatibility overlay restored `c5252865` and old params, and exactly one old
loop started. Generator and reserve bytes remained intact. The actual log is
`/home/bb/scratch/dedup-combined-committed-abort-v1.log`:

```text
2026-11-21T14:33:11Z gate r000091; old HEAD c5252865
2026-11-21T14:34:21Z journal gate r000091: yes applied n_tasks 2000 lease_failed 0 why []
2026-11-21T14:35:02Z APPLY OK: new HEAD 50afaca3; params applied through set-params CLI
2026-11-21T14:36:02Z first open r000092: OK active 10000 params d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c n_tasks 2000 snapshot sqrt_fisher eval_wall_s None cuts None admission_cuts None why []
2026-11-21T14:36:10Z FAIL: public files not healthy after 8 seconds: BAD public-check failed or used the remaining retry window
2026-11-21T14:37:19Z autodeploy exit 1 (no PASS)
14:37:16Z rollback: reviewed new-code run finished undecided r000092 as applied; no stored result was discarded
2026-11-21T14:37:19Z old-code loop started after rollback
shimmed old params/code verified
generator and reserve bytes preserved
2026-11-21T14:37:22Z DRY POST-OPEN ROLLBACK PASS: sealed/undecided evidence finished under new code; shimmed old params/code and one old loop
```

Pending items were replaced by the terminal suite records above and the
physical-validator qualification below. Complete-suite failures remain
explicit; the focused PASSes are not represented as a green full suite.

The final rehearsals compared the complete recursive 24-file kit and ran
without `DEDUP_DRY_DEVELOPMENT=1`. Only their verbatim output is recorded as
committed-kit PASS evidence. Normal, abort and the earlier seven rollback
checks completed; the final physical-validator suite extends that rollback
coverage. No production deployment is implied. The historical complete
24-file SHA-256 inventory is
`/home/bb/scratch/dedup-combined-final-kit-hashes.json`; it does not include
the final physical-validator additions.

### Final physical-validator additions

The staged package contains 124 source files: 123 pinned `c5252865` files plus
`dedup_compat.py`. Seven named modules replace old source bytes; all other
source files retain their exact old hashes. The optional deterministic archive
includes the entire package and its source manifest. The shared-package tests
`test_weights_writer.py`, `test_links_core.py`, `test_links_cli.py`,
`test_site_pull.py` and `test_site_pull_pool.py` passed **113 tests** with exit
0 against a complete old-v2 fixture layout whose `sn120_v2/` was replaced by
the staged package. Its log is
`/home/bb/scratch/dedup-physical-validator-shared-jobs-v2.log`.

The first shared-test attempt ran those tests with only the package staging
parent available. It reached 100% with one failure because
`test_links_cli.py` expects sibling `systemd/` deployment fixture files, which
are intentionally not part of the package-only archive. The successful rerun
used a copy of the full old-v2 tree to supply those fixture files, replacing
only `sn120_v2/`. Neither result is a production-systemd or live-service test.
