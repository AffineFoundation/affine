# Dedup release qualification record

This record describes **offline qualification only**. It does not report a
production deployment, production health, or authorization to arm `r000089`.
The exact committed-kit rehearsal results and file hashes are reported
separately against their final HEAD; this source document is not rewritten to
claim its own post-commit results.

## Non-arming boundary

The reviewed live parameters would cut the `r000087` king: UID 488 has
`global_norm_cap=0.01 < min_step_norm=0.02`, producing admission
`NULL_UPDATE`; the 58-rollout batches also fail `min_signal_rollouts=64`.
Therefore `r000089` is not armed. Its appearance in the kit is an offline gate
boundary only.

The operator decision requires both the code-first deployment design and the
failure rehearsal. A kit or a PASS line does not itself authorize a live run.

## Frozen release verification

The release remains `e96566341dfa463d0a2f1853f038dac9877b2c9a`.
`git diff e0ab853a e9656634 -- v2/sn120_v2/gpu` is empty. Independent local
manifest recomputation gives:

```text
b68a3005fd9837e9474349ca086088d9a7649406772e141b169952e7ef7f6e01
```

The changed paths between those two commits are the params-step helper/tests,
publisher `llms.txt` prose, and its test line. No release GPU source changed.

The external `bb/dedup-set-params-isolated` branch was checked with
`git ls-remote --heads origin bb/dedup-set-params-isolated`, including on
2026-11-21, and was not returned. The reviewed local guarded isolation
implementation was therefore used. No remote branch was assumed to exist or
to have passed review.

## Copied real-journal isolation evidence

The supplied source journal is:

```text
/home/bb/inbox/dedup-prodstate/journal.jsonl
SHA-256 1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca
574 records, sequence 0..573, r000088 open
```

The source was not modified. The evidence directory is
`/home/bb/scratch/dedup-prod-journal-qual-20261121`.

An independent copied open journal was tested with the real guarded isolation
script. Its refusal returned exit 2, retained all 574 records byte-identically,
and reported:

```json
{
  "after_params_hash": "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a",
  "before_params_hash": "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a",
  "exit": 2,
  "isolated": true,
  "journal_records_after": 574,
  "journal_records_before": 574,
  "live_journal_writes": 0,
  "network_connections": 0,
  "r2_writes": 0
}
```

A distinct copy was legitimately advanced to IDLE (577 records) before the
transition. The guarded transition returned exit 0, added one leader and one
`params_set` record, and produced:

```json
{
  "after_params_hash": "0b2f390b0e416e9b1ff2523b81976777d827c1da92238f9147cbae7d8472a5f6",
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

The changed effective keys are exactly:

```text
dedup_enabled        false -> true
min_signal_rollouts  1 -> 64
min_step_norm        0.0 -> 0.02
train_seed_scope     batch -> round
```

The guard blocks socket connections, R2 construction, journal mirror pushes
and local-store writes outside the selected isolated store. No production
journal, network or R2 write occurred in these rehearsals.

## Rollback qualification, working-kit development run v23

Command boundary: real new code generated a fresh isolated fixture;
`qualify-rollback.py` was run under Python `-I` with explicit
`--old-v2 /home/bb/scratch/dedup-old-c5252865/v2`. Output is under
`/home/bb/scratch/dedup-rollback-qualification-v23`; its log is
`/home/bb/scratch/dedup-rollback-qualification-v23.log`.

The actual six PASS lines were:

```text
ROLLBACK QUALIFICATION PASS: real new-controller fixture: failed lease, duplicate-only, null-update, Stage-B duplicate, sealed nonfinite evidence
ROLLBACK QUALIFICATION PASS: closed dedup history replays with charged duplicates/null update and uncharged failed-lease survivors
ROLLBACK QUALIFICATION PASS: bare old code separately rejects new params, duplicate/null_update, new round fields and tagged nonfinite values
ROLLBACK QUALIFICATION PASS: old-code compatibility CLI re-pins bba7a409 after historical 0b2f390b; unknown/newer params refused
ROLLBACK QUALIFICATION PASS: undecided/decided open dedup rounds refuse before fence; local and mirrored journals unchanged
ROLLBACK QUALIFICATION PASS: unmodified old loop uses the compatibility executable and exits after exactly one bounded start
```

The generated fixture journal SHA-256 is
`e692ddcd946c64a19f21564be3ff0a3ba89414adfafae54905ec64177853d21f`.
This proves the named synthetic compatibility behavior; it is not production
state recovery. The final committed-kit rollback rerun is required before
release sign-off.

**Bare old commit `c5252865` cannot perform this rollback.** The compatibility
overlay is required. Sealed evidence is retained independently of the normal
pending-file close lifecycle; it is never deleted to force old-code replay.

## Frozen-input planning evidence

The offline plan summary is
`/home/bb/scratch/dedup-frozen-inputs-summary-final.json`:

```text
journal SHA-256  1a7475525b8c176b445927fe864b1df6e7abf31b8083e011e9a3e4c2c3acf1ca
rounds           r000085, r000086, r000087, r000088
batch_count      24
lora_candidates  0
```

Zero candidates here means the planner had no previously fetched sealed
results from which to derive LoRA files. It does not establish their absence
in production. No pending LoRA directory is promised to remain available. No
production object or credential was accessed. The unit tests exercise the
GET-only transport and explicit `--batch`, `--lora`, `--params`, and
`--first-seen` behavior with synthetic stores.

## Focused and broad regression evidence

On 2026-11-21, the focused suite completed with exit 0 and all 25 tests passed,
including the bounded dry-recovery stop-environment and confirmation checks:

```bash
env PYTHONPATH="$PWD/v2" PYTHONDONTWRITEBYTECODE=1 \
  /home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -m pytest -q \
  v2/tests/test_dedup_deploy_operator.py \
  v2/tests/test_dedup_isolated_set_params.py \
  v2/tools/dedup-deploy/test_frozen_inputs.py
```

Output:

```text
.........................                                                [100%]
```

The broader dedup/deployment selection also completed with exit 0 and all 86
tests passed:

```text
........................................................................ [ 83%]
..............                                                           [100%]
```

It covered `test_dedup.py`, `test_dedup_controller.py`,
`test_dedup_params_step.py`, `test_dedup_publication.py`, the three focused
files above, using the same isolated Python environment.

The final 2026-11-21 working-tree rerun also completed with exit 0 and all 86
tests passed after the CLI help correction and the test-isolation correction.
The legacy-params test now stages its command output only in a temporary kit
copy, so the test suite no longer creates `set-params-old-params.json` in the
source kit. Its focused seven-test rerun passed independently.

## Dry-run development evidence and limitations

Early development attempts did not emit a deployment PASS. In particular,
v10 was stopped by its background execution time limit during old-code
predecessor setup. It had preserved the copied `r000086` lease as genuine
`no_decision` and returned the expected old params hash, but it did not reach
the health gate. It is not counted as a pass.

v11 failed before a clean predecessor round: the synthetic batch initially
used one rollout per prompt group, yielding zero advantage and `no_signal` for
every rollout. v12 genuinely closed the copied `r000086` as `no_decision`
after its real expired lease, then closed fresh `r000087`–`r000089` with 2,000
tasks and no `lease_failed`; it failed before apply because the compatibility
status command lacked an explicit old-code package path. v13 failed before the
gate because the synthetic-input helper rejected the legitimately open copied
`r000086`. v14 opened `r000089` but failed before gate execution or apply
because a fixture incorrectly assumed waiting selection matched queue
insertion order. v15 closed fresh `r000087`, `r000088` and the offline
`r000089` gate, applied new code/params, passed the `r000090` public/open and
basic health checks, and observed `r000091`. It then failed the harness's
required duplicate-cut assertion: its stale fixture wrote the duplicate to a
non-selected queue seat, so the actual health counts were `duplicate=0` and
`null_update=1`. Its nested basic `PASS` line is not a deployment-kit
qualification PASS; the process exited 1. The selected-seat duplicate write
was corrected afterward. v16 reached genuine clean `r000087` and `r000088`
closes with the complete old params pin, but its harness background command was
stopped at its execution time limit before the offline gate, apply or health.
It is not a qualification PASS; no v16 deployment or health claim is made.
None of these attempts is a qualification PASS.

Development v17 completed the normal rehearsal in
`/home/bb/scratch/dedup-dry-development-v17`. The real offline `r000089` gate
closed `applied` with 2,000 tasks and no failed lease. New code/params applied;
public checks passed for `r000090`; its health was `duplicate=1`,
`null_update=1`, admission `min_rollouts=1` and `null_update=1`, with four of
six selected batches surviving. `r000091` opened in that same loop, whose real
stop-after-round path then closed it and exited. The harness printed only
`DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no final PASS`.
This is not a committed-kit PASS and does not arm production.

Development v18 repeated the complete normal rehearsal with exit 0 in
`/home/bb/scratch/dedup-dry-development-v18`. Its offline `r000089` gate closed
`applied` with 2,000 tasks and no failed lease. Apply reached the frozen new
HEAD and params; public/open checks passed. First-round health again recorded
`duplicate=1`, `null_update=1`, admission `min_rollouts=1` and `null_update=1`,
and a 1,234-second evaluation wall. `r000091` opened in the same loop and the
real stop-after-round request was obeyed and removed. Generator/reserve bytes
were preserved and `pool/state` was absent from the public projection. Its only
top-level completion line was `DRY DEVELOPMENT COMPLETE: real run/gate/apply/public/health; no final PASS`.

Abort development v1 genuinely left an undecided open `r000090` after its real
worker started. Its crash marker then persisted into recovery, causing the
replacement worker to crash again before judging; `r000091` was opened on a
later pass. This was a failed fixture, not a recovery PASS or a controller
stop-after-round failure. The isolated run was stopped before old-code
checkout. The injection is now one-shot, and recovery fails immediately on a
successor-open attempt. Abort v2 then finished `r000090` as `applied` with no
successor-open marker, but its recovery run did not remove the stop request;
rollback correctly stayed blocked and the watchdog stayed disarmed. It is not a
recovery PASS.

Abort v3 stopped during helper generation on a stray heredoc delimiter
(`line 215: PY: command not found`); no fixture round ran. The separate
`/home/bb/scratch/dedup-dry-development-abort-v4` attempt stopped during its
predecessor setup: `KeyError: 'SN120_WORK_DIR'` in the real stop path. A later
retry using the same intended root failed immediately on its nonempty `D`
guard, with the competing-process refusal for that previous attempt's
`genfeed-deploy/autodeploy.sh`; it did not run the gate or recovery. The
one-shot injection, explicit `SN120_WORK_DIR` propagation and fresh-directory
checks address these setup faults. None of these attempts is a PASS.

The dry-run fixture uses real controller, admission, worker/result, journal,
CLI and stop-after-round paths, with deterministic local trainer/evaluator
seams. It is not a production-GPU or production-throughput qualification.
Normal close removes pending objects; independently archived sealed evidence
is used where the rollback fixture needs to inspect it.

Abort development v5 completed with exit 0 in
`/home/bb/scratch/dedup-dry-development-abort-v5`. A real worker crash left
`r000090` open and undecided. The recovery run remained on new HEAD, closed
that round `applied`, emitted `stop after the round: obeyed.` and removed its
request; no successor-open marker exists. Only afterward did the fixture check
out `c5252865`. Compatibility replay reported IDLE, old params `bba7a409…`,
`batch_cap=6` and the retained old worker recipe digest. One old-code loop was
requested, generator/reserve bytes were preserved, and the watchdog was
re-armed only after recovery. Its only top-level completion line was
`DRY DEVELOPMENT COMPLETE: bounded new-code finish and shimmed old rollback`;
it is not a committed-kit PASS.

Sealed-result development v1 completed with exit 0 in
`/home/bb/scratch/dedup-dry-development-sealed-v1`. The one-shot injector
crashed only after a durable ticket-bound result was stored on `r000090-l0`.
`sealed-result-proof.json` recorded `decided=false`; recovery then judged the
stored result under new code rather than copying or discarding it. The round
closed `applied`, emitted the real stop-obeyed line, removed the request and
opened no successor. Only then did the fixture check out `c5252865`, re-pin the
old params through the compatibility loader, and request exactly one old-code
loop. Generator/reserve bytes were preserved and the watchdog was rearmed after
verified recovery. Its only top-level completion line was
`DRY DEVELOPMENT COMPLETE: bounded new-code finish and shimmed old rollback`;
this is not a committed-kit PASS.

Working-kit rollback requalification v24 completed with exit 0 on 2026-11-21.
Its log is `/home/bb/scratch/dedup-rollback-qualification-v24.log`; the same six
PASS checks quoted above were observed, and the generated journal SHA-256 was
again `e692ddcd946c64a19f21564be3ff0a3ba89414adfafae54905ec64177853d21f`.
Working-kit v25 repeated all six checks with exit 0 in
`/home/bb/scratch/dedup-rollback-qualification-v25`, with the same fixture
journal digest. Working-kit v26 again completed all six checks with exit 0 in
`/home/bb/scratch/dedup-rollback-qualification-v26`, with the same digest.
These are working-kit results, not final committed-kit runs.

The final normal and post-open recovery results must be copied verbatim from
clean committed-kit runs. No inferred PASS line belongs in any report.

Decided-open development v1 did not reach apply or the intended crash. Its
`autodeploy` correctly refused with exit 3 because the harness had not yet
created the disposable `go-confirmed` marker; no stop request or state change
was made. The harness no longer treats a fixed two-second sleep as proof that
the driver has passed its marker checks: it creates both offline markers before
releasing the gate and waits for the driver's armed log line. A fresh
`dedup-dry-development-decided-v2` rehearsal is required; v1 is not a PASS.

Decided-open development v2 completed with exit 0 in
`/home/bb/scratch/dedup-dry-development-decided-v2`. Its real journal boundary
appended `r000090` `round_settled` at sequence 593 with a non-null decision,
then crashed before `round_closed`. `decided-open-proof.json` recorded
`decided=true` and the durable sealed result on `r000090-l0`; the proof is
against the isolated store, not production. The bounded new-code recovery
closed that same round `applied` at sequence 597, emitted the real stop-obeyed
line and removed its request. The recovery successor-open marker is absent.
Only afterward did the fixture check out `c5252865` and report IDLE with the
complete old params pin `bba7a409…`, `batch_cap=6` and the retained old worker
recipe digest. Exactly one old-code loop was requested; generator/reserve bytes
were preserved and the watchdog was rearmed after verified recovery. Its only
top-level completion line was `DRY DEVELOPMENT COMPLETE: bounded new-code
finish and shimmed old rollback`; this is not a committed-kit PASS.

## Final committed-kit qualification

Pending at this source revision: complete normal and abort-after-open
rehearsals, final rollback qualification, broad tests, clean-tree checks, exact
kit HEAD, and SHA-256 for every kit file. Final rehearsal must compare the
complete recursive kit and must run without `DEDUP_DRY_DEVELOPMENT=1`.
