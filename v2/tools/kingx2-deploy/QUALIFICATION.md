# Offline qualification — DEPLOY KIT (do not run against prod)

This note records offline evidence, not deployment authorization or production
health. The deployed source base is
`d368b2601f06458349cfe3e84adfbf691994fcb3`; the selected implementation is
`c525286580c84d795b3ca7cbdf2302e96fce31ba`, its direct child. PR 307's kit head
is `8716b7da`.

No production service, chain write, R2 operation or secret access is part of
these checks. All deployment tests use local fixtures only.

## Implementation identities

- GPU code: `64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9`
  -> `5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47`.
- Params: `94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c`
  -> `bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a`.

```json
{
  "irt_draw_rule": {"before": null, "after": "sqrt_fisher"},
  "null_samples_per_task": {"before": null, "after": 2},
  "recipe": {"code_digest": {
    "before": "64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9",
    "after": "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
  }}
}
```

The `null` before values mean omitted fields, not stored JSON nulls: the old
rule was Fisher and the king inherited the LoRA sample count. LoRA
`samples_per_task=1`, 2,000 tasks,
six-LoRA batch cap and 10,000-task rotation are unchanged.

The complete genfeed consumer, `reserve/generator.py` and ingestd script are
byte-identical between d368b260 and the selected implementation. The shared
controller retains `open_and_publish`; the new `set_params` rebuild retains
the stored IRT snapshot and live pool params. The `genfeed-deploy` directory is
identical to PR 307 head 8716b7da on the final rebased branch.
The rebased branch additionally includes the five files changed by PR 307 kit
head 8716b7da exactly: `test_generator_deploy.py` and the four updated
`genfeed-deploy` files. Those kit/test-only additions do not alter the GPU
manifest or selected params.

## Generator live-v2 verification

Fresh isolated report: `/home/bb/scratch/kingx2-genfeed-rebase/genfeed-four-epochs-final.log`.
The rebased consumer ingested the supplied live-v2 `.done` inputs offline:

| Epoch | Rows | Kept | Schema rejects |
| --- | ---: | ---: | ---: |
| 20261120T0400Z | 885 | 318 | 0 |
| 20261120T0800Z | 1,231 | 554 | 0 |
| 20261120T1200Z | 679 | 408 | 0 |
| 20261120T1600Z | 533 | 430 | 0 |
| **Total supplied** | **3,328** | **1,710** | **0** |

King revisions were `56bae61bb4fde8dc786567e1e833ff5d53f5203f537a32ff2d2ee1fbdfc2e810`
for 0400Z/0800Z and `1433153044e242bf28583fe3d30c8e6ca8752ef19fa808b85e8bed4a4c4373e9`
for 1200Z/1600Z. The input files' supplied SHA-256 values were checked before
use; the new isolated reserve manifests have different byte identities from
the other thread's run and are reported as this run's outputs, not claimed to
be byte-identical to that run.

**2000Z is not verified.** The other thread searched this host and could not
supply a real 2000Z accepted JSONL plus `.done`/results artifact or substantiate
the proposed `194` count. No expected-194 assertion is used or claimed. This is
an explicit coverage limitation of the genfeed verification, not a fabricated
pass.

## Replay and synthetic evidence

- Historical report: `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/history-final/history.json`
  (log: `history-final.log`; report SHA-256
  `747ffb01f47fb6b60f9cec05260e092248545d15d904807789e5f104dcd69329`).
  The old r000001-r000081 snapshot passed the available-input identity checker
  against its original historical round artifacts and params (the report's
  `base_commit` labels that old snapshot, not this rebase's implementation
  parent): all 81 public draws, public entry fields, public score fields and
  round metadata matched, along with 33 public winner decisions, 14 rotations
  and the fully reconstructible r027 IRT update.
  Sixty-nine rounds lack private scored-LoRA outcomes needed for complete IRT
  recomputation. No standalone score artifacts exist; score fields in
  `round.json` were compared instead. This is not complete private decision/IRT
  recomputation.
- Synthetic round: `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/synthetic-run/synthetic.json`.
  Two king strings and one LoRA string give delta 0.75, SE
  0.14433756729740643, cap denominators 8/4 and an SE ratio 0.8628490455699164,
  near sqrt(0.75) = 0.8660254037844386 for comparable independent noise.
- Next open: `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/next-open-run/next-open.json`.
  The old/new opens have the same 10,000-task active set and rotation, the new
  params hash and sqrt-Fisher snapshot, and different intended draws.
- Timing projection: `/home/bb/scratch/kingx2-timing-final.json`. Adding one
  king sample changes `1+L` rollouts to `2+L`: +14.2857% at L=6. Projected
  totals are r074 8,138s, r080 9,640.86s and r081 7,300.43s, below the 10,200s
  scoring-before-reserve limit. This is rollout-count scaling, **not measured
  GPU throughput**. Older maxima r049 (9,455s evaluation) and r055 (10,212s
  total) used other configurations; live health remains an explicit gate.

For r081, the journal-authenticated rotation is
`289a2184869bda8a2d97cef809c3bf91c64c3db357e3108fea6ff11164e47dcf`
and round SHA is `3f49be1b91f866bd280f956be3add0b9add27d751ae6be3d65a5762cf8e365b0`.
The older cached ranking `6306d36d920380f76e02027ca5757a9f64ef991c80dffc000927bb84f9bc4b7d`
and rotation `449af9ac3fa9f2257296f507f63f5f6238265062b64e31ed9fcbb2b52ab2a7bc`
do not match that journal generation. They remain explicitly reported, not
waived; the authenticated current generation was used for replay.

## Test and rehearsal logs

- Exact-base full suite: `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/full-base.log`.
- Selected-implementation full suite:
  `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/full-impl.log`.
  Implementation: **29 failed, 1559 passed, 1 skipped, 2 errors** (1,591
  collected; 2,193.37s). Exact base: **29 failed, 1538 passed, 1 skipped,
  2 errors** (1570 collected; 2,192.44s). Both suites exited nonzero. The
  completed comparator report is
  `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/full-suite-comparison.json`.
  It captures all 29 named failures and two named setup errors on each side:
  **zero implementation-only failures, zero base-only failures and zero
  differing named-failure reasons**. The implementation collects one additional
  restored genfeed-deploy test file; that file separately passes below.

  The shared failures are one `test_grader_lifecycle` lazy-pool/round-close
  failure, one `test_public_queue_files` snapshot-key byte-identity failure,
  seven `test_pull_chain` historical-chain/link failures, eleven
  `test_site_build` site/frozen-history failures and nine `test_site_pull`
  layout/store/frozen-history failures. Their exact parametrized test IDs and
  summary reasons are enumerated in the JSON comparator. The two shared setup
  errors are
  `test_adam_eps_reaches_both_training_paths_and_default_matches_the_old_optimizer[1]`
  and `[3]`: both test fixtures attempt a historical `git show` outside a Git
  worktree. This is a shared isolated-test-environment limitation, not evidence
  that those trainer tests passed.

  Targeted reruns resolved the local exported-tree harness issues identically on
  both trees. With each selected tree's absolute `v2` on `PYTHONPATH` for child
  processes, the pull-chain/site files containing the other 27 failures gave
  **33 passed in 89.15s** on implementation and **33 passed in 88.75s** on base
  (`env-repair-site-impl.log`, `env-repair-site-base.log`). Supplying the local
  Git object database to the archive-only trainer fixtures gave **72 passed in
  35.63s** on implementation and **72 passed in 33.82s** on base
  (`env-repair-trainer-impl.log`, `env-repair-trainer-base.log`). Only these two
  genuine assertions then remained, and their isolated reruns still fail on both
  trees: the synthetic `VllmEvaluator` lacks `capture` in its `close()` path,
  and the historical snapshot test expects SHA `a5ffb03f...` but observes
  `7b07d90a...` (`isolated-shared-impl.log`, `isolated-shared-base.log`). The
  raw 29-plus-two comparison remains the actual full-suite result; the targeted
  runs explain its environment-dependent subset rather than replacing it.
- Focused feature/import tests: `/home/bb/scratch/kingx2-genfeed-rebase/tests-agent/focused.log`
  (**175 passed in 150.98s**).
- Restored PR 307 kit plus consumer regression tests:
  `/home/bb/scratch/kingx2-genfeed-rebase/genfeed-focused.log`
  (`tests/test_generator_deploy.py` and `tests/test_generator_feed.py`, exit 0,
  fifteen passing tests).
- Genfeed-aware deployment development fixture:
  `/home/bb/scratch/kingx2-r086-genfeed/dev-normal-v3/dry-run.log`.
  This uses a real detached ingestd from the selected copied checkout, a valid
  external reserve and a real competing genfeed-named process. This development run reached the real autodeploy's r000087/r000088 health
  PASS, then stopped at its strict reserve-byte comparison because the initial
  read-only shared-lock verification had created the normal empty
  `.ingest.lock` coordination file. No manifest, shard or generator state bytes
  changed. The final harness takes its baseline immediately after that initial
  verification and requires the empty lock and every reserve byte to remain
  unchanged thereafter. The development run is not a complete qualification
  PASS. Final PASS lines below come only from later committed-kit runs.

The corrected entry point is `kx2_operator.py`, invoked with `python3 -I` to
avoid stdlib `operator` shadowing and ambient import paths. `autodeploy` alone
constructs that command; the rehearsal has no `OP` override. The gate default is
read from `expect.json` and is `r000086`. The committed-kit rehearsal uses
implementation `c525286580c84d795b3ca7cbdf2302e96fce31ba` directly: no scratch
commit, working-tree overlay or `new_commit` substitution. Helper modules live
outside the deployment checkout and use explicit isolated imports.

The rehearsal copies journal/public objects and simulates only the loop and its
outcomes. It advances the copied r000082-open journal through the r000086 gate
with synthetic old-code closes, then checks the new-code r000087 close and
r000088 open. The live reserve is an independent, valid local prefix selected by
`reserve-live.path`; checkout never copies or reseeds it. The disposable ingestd
uses genfeed's exact detached launch construction and the selected checkout's
working directory. PID, full command and proc start ticks are checked through
apply and rollback. The initial shared-lock read creates the normal empty
`.ingest.lock`; the fixture fingerprints it before apply and requires it to
survive. Generator selector, state, prior index, pending inbox and reserve
files are fingerprinted independently of mutable future ingest appends.
Compatible-schema synthetic follow-ups use the pinned implementation to read
historical new-schema journal records after old params/code are restored. No
GPU workload, production connection or production health claim is involved.

Final committed-kit rehearsal destinations:

- Normal: `/home/bb/scratch/kingx2-r086-genfeed/final-normal/dry-run.log`.
- Abort after open: `/home/bb/scratch/kingx2-r086-genfeed/final-abort/dry-run.log`.

Each run clones the exact final branch head and compares all eight kit files to
its Git archive before proceeding. The `qualified branch head ...` line in each
fresh log identifies that exact commit; the required PASS lines are reported
with the kit-head result after those runs finish. Earlier logs above are
historical baseline evidence, not qualification of the corrected final kit.
