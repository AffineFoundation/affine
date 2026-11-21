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

The first integrated-kit suite was stopped after its known smoke-copy fixture
failure and before terminal output; it is not complete-suite evidence. A new
corrected integrated-kit suite and a clean `50afaca3` release suite are running.
The complete-suite failure above remains expected until separately reconciled;
its presence must not be omitted from final counts.

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

This is **working-tree development evidence only**, not a final committed-kit
PASS. The compatibility loader validates the exact full `d5979c30…` pin,
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

### Native release reader after rollback

The validator/weights host can stay on native release `50afaca3` after a
controller rollback; it does not need to go back to `c5252865`. The native
release `Params.from_dict` gives an old record that omits the dedup fields
their effective legacy values, including `dedup_enabled=false`,
`eval_merged=false` and batch-scoped training seeds. Release state/result
types decode historical duplicate decisions, duplicate clusters and tagged
non-finite stored evidence. The compatibility overlay is needed by the **old
controller**, not by native release readers.

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
guard check, not the rehearsal's terminal result. No v3 completion is claimed
here without its terminal output.

The earlier normal-v1/v2 stale-manifest attempts described below predate these
combined merged-fixture runs; the names refer to different attempts.

At `2026-11-21T14:21:47Z`, both GitHub PR #310 and `refs/pull/310/head` still
resolved to `1fa67a9770884eb679ffeb91fd133ac49575d117`. The integrated release
is `50afaca32ea982796728a0b07cb6e3c579ef378b`, its GPU tree matches AFF-5,
and `c5252865` remains an ancestor. AFF-5's terminal report is recorded above.

Pending at this source revision:

- Terminal complete combined-release and integrated-kit suites, retaining the
  pre-existing public-snapshot failure explicitly.
- Combined normal and abort-after-open development rehearsal terminal output.
- Clean committed-kit normal and abort-after-open rehearsals.
- Committed-kit rollback qualification.
- Exact final kit HEAD and every kit-file SHA-256.

Final rehearsal must compare the complete recursive 24-file kit and run without
`DEDUP_DRY_DEVELOPMENT=1`. Only verbatim output from those runs may be added
here or reported as final PASS evidence. No current committed-kit normal,
abort or rollback PASS is claimed.
