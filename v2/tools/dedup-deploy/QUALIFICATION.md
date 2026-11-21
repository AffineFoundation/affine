# Dedup release qualification record

This record describes **offline qualification only**. It does not report a
production deployment, production health, or authorization to arm a round.
Final committed-kit rehearsal results are recorded only from actual output;
no development completion line is promoted to a final PASS.

## Current approved release

The approved design enables dedup, `train_seed_scope=round` and charged null
updates, raises the crown threshold to `z_bar=1.5`, and makes both admission
floors no-ops:

```text
dedup_enabled        true
train_seed_scope     round
dedup_rollout_overlap 0.5
dedup_cos_shared_seed 0.135
dedup_cos_own_seed    0.002
min_signal_rollouts  1
min_step_norm        0.0
z_bar                1.5
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

The minimal correction is commit
`8afd574736f6a2bac351cd3aa0edcd6295c35e44`, directly on
`e96566341dfa463d0a2f1853f038dac9877b2c9a`. It changes only
`v2/sn120_v2/admission.py` and `v2/tests/test_min_rollouts.py`, removing the
implicit minibatch floor and retaining the explicit configured floor.
`git diff e0ab853a 8afd5747 -- v2/sn120_v2/gpu` is empty. Release commit
`5e9df83842f34cc3204d8003169d72f5ed4121ab` sits directly on `8afd5747` and
updates only `publisher.py` and its tests: the `llms.txt`/`mining.json` contract
now states the explicit signal floor, disabled norm floors and dynamic `z_bar`.
The release pin in `expect.json` selects this commit, independently of the kit
commit, so it can later be re-pinned to AFF-5's combined rounding-fix release.
The GPU source manifest remains:

```text
b68a3005fd9837e9474349ca086088d9a7649406772e141b169952e7ef7f6e01
```

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
file SHA-256           f8073812849133f5d0e81728ef361d3e98b0466b4b76e3b6259a4a858d891d4b
canonical params_hash  8543962a34e8fecb0da3663e05b0c222ae612cbe4a8cef44fedf24f42f0504e5
old file SHA-256       86c9cd5b5e696c6c10411fd828b7e26a641d06bd39f94a4ea98eb69de3e78f56
old canonical hash     bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a
```

A canonical `Params.params_hash` is not the SHA-256 of the pretty JSON file.
The full params retain king-x2's worker recipe digest
`5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47`;
that is distinct from the release's GPU source digest above.

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

The source SHA-256 was unchanged after testing. Evidence for this approved
z=1.5 transition is under `/home/bb/scratch/dedup-z15-actual.fIOu3YMp`, in
particular `guarded-result.json`.

An independent copied open journal was tested with the real guarded isolation
script. The refusal returned exit 2 and left its journal byte-identical at the
source SHA-256 above:

```text
a round is open; set the parameters when it has closed (a decided round takes new digests only)
```

A distinct copy was legitimately advanced to IDLE as `no_decision`, without
reading or discarding a sealed result. Its 577 records became 579 during the
guarded transition, which produced:

```json
{
  "after_params_hash": "8543962a34e8fecb0da3663e05b0c222ae612cbe4a8cef44fedf24f42f0504e5",
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
  "z_bar": {"before": 1.0, "after": 1.5}
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

## Regression evidence for the z=1.5/no-op-floor revision

On 2026-11-21, the focused 30-test selection passed:

```bash
env PYTHONPATH="$PWD/v2" PYTHONDONTWRITEBYTECODE=1 \
  /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -m pytest -q \
  v2/tests/test_dedup_params_step.py \
  v2/tests/test_dedup_publication.py \
  v2/tests/test_dedup_deploy_operator.py \
  v2/tests/test_dedup_isolated_set_params.py \
  v2/tests/test_min_rollouts.py
```

Output:

```text
..............................                                           [100%]
```

The broader selection also completed with exit 0 and all 97 tests passed:

```bash
env PYTHONPATH="$PWD/v2" PYTHONDONTWRITEBYTECODE=1 \
  /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -m pytest -q \
  v2/tests/test_dedup.py v2/tests/test_dedup_controller.py \
  v2/tests/test_dedup_params_step.py v2/tests/test_dedup_publication.py \
  v2/tests/test_dedup_deploy_operator.py \
  v2/tests/test_dedup_isolated_set_params.py \
  v2/tools/dedup-deploy/test_frozen_inputs.py v2/tests/test_min_rollouts.py
```

Output:

```text
........................................................................ [ 74%]
.........................                                                [100%]
```

Coverage includes no implicit minibatch floor, finite zero/small norms at the
zero floor, charged non-finite stored updates, only three effective params
changes for a legacy king-x2 record, explicit seven-field dedup serialization
plus `z_bar=1.5`, stopped-loop isolation, public-only authenticated checks,
404/5xx/timeout retries, and health using journal decision cuts plus
`run.log`/public `eval_wall_s`. Publication tests verify the dedup rules, round
training-seed formula, both no-op floors and the literal `z=1.5` final/early-look
rules in `llms.txt`, the `z_bar=1.5` value in `mining.json`, and a clean
`publisher.check_pair` result.

## Current working-kit rollback requalification

The z=1.5 development v1 exposed a decoder problem: adding the existing
`z_bar` to the removal map made its presence look like a new unknown field.
The correction distinguishes the seven genuinely new dedup fields from the
reviewed historical `z_bar=1.5` override, validates the complete pin, then
removes all eight release values before decoding old behavior. Development v2
also exposed that the accounting validator must use an actual dedup field to
recognize new history; an ordinary old params file can legitimately contain
`z_bar`. Neither v1 nor v2 is a PASS.

The corrected working-kit v3 command completed with exit 0:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -I \
  v2/tools/dedup-deploy/qualify-rollback.py \
  /home/bb/scratch/dedup-z15-rollback-development-v3 \
  --old-v2 /home/bb/scratch/dedup-old-c5252865/v2 \
  --python /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python
```

Its log is `/home/bb/scratch/dedup-z15-rollback-development-v3.log`. The actual
six PASS lines were:

```text
ROLLBACK QUALIFICATION PASS: real new-controller fixture: failed lease, duplicate-only, null-update, Stage-B duplicate, sealed nonfinite evidence
ROLLBACK QUALIFICATION PASS: closed dedup history replays with charged duplicates/null update and uncharged failed-lease survivors
ROLLBACK QUALIFICATION PASS: bare old code separately rejects new params, duplicate/null_update, new round fields and tagged nonfinite values
ROLLBACK QUALIFICATION PASS: old-code compatibility CLI re-pins bba7a409 after historical 8543962a; unknown/newer params refused
ROLLBACK QUALIFICATION PASS: undecided/decided open dedup rounds refuse before fence; local and mirrored journals unchanged
ROLLBACK QUALIFICATION PASS: unmodified old loop uses the compatibility executable and exits after exactly one bounded start
```

The replay assertion explicitly checks `state.params.z_bar == 1.0` after the
reviewed z=1.5 history is decoded. This is working-kit development evidence;
the committed-kit rerun at `a8f5b5dcdaac2421c44b7ac399d40f0798874f54`
also completed with exit 0. Its log is
`/home/bb/scratch/dedup-z15-rollback-final.log`; its fixture journal SHA-256
is `69fb00a5eca85f7fc4286dbe761e5a443d82440d4df8518765fb4c5c41e7a0dc`.
It emitted the same six PASS lines above against the reviewed `8543962a…`
history. **Bare `c5252865` cannot
replay dedup history.** Rollback is code-first: finish open work under new
code, check out `c5252865`, apply the complete old `bba7a409…` params through
the hashed compatibility loader, and start exactly one old-code loop. Sealed
evidence is never deleted to make old code accept history.

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
checks against `0b2f390b…`; the current `8543962a…` run above supersedes their
params identity.

## Final committed-kit qualification

The z=1.5 normal development rehearsal completed with exit 0 and emitted
`DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no final PASS`.
Its health was 2,000 scored tasks, one duplicate, one charged null update,
zero admission cuts, no lease failures, and a clean `r000093` second open.
Its log is `/home/bb/scratch/dedup-z15-development-normal-v1.log`. This is
fixture development evidence, not a final committed-kit PASS.

Committed-kit rollback qualification is complete at `a8f5b5dc`, as recorded
above. Pending at this source revision: clean normal and abort-after-open
rehearsals, exact final kit HEAD and every kit-file SHA-256.
Final rehearsal must compare the complete recursive 24-file kit and run without
`DEDUP_DRY_DEVELOPMENT=1`. Only verbatim output from those runs may be added
here or reported as final PASS evidence.
