# Re-pin a final stacked release

Run from the final stacked **development checkout**, after the crown-gate and
AFF-6 commits are present:

```bash
/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python -I \
  v2/tools/dedup-deploy/repin.py FINAL_COMPLETE_40_HEX_SHA
```

The checkout's `HEAD` must equal that SHA. The tool does not fetch, checkout,
commit, push, deploy, access R2 or production, or write chain state. Its default
rounds are gate `r000093` and first health `r000094`; override them together:

```bash
python -I v2/tools/dedup-deploy/repin.py FINAL_SHA \
  --gate-round r000101 --first-health-round r000102
```

The health round must be the gate's successor. `--python` selects the reviewed
test interpreter. `--accept-new-pin PARAMS_HASH:GPU_DIGEST` retains an additional
actually applied/reviewed pair and is repeatable. `--skip-frozen-inputs` is an
explicit incomplete run, not final kit validation.

The command recomputes the GPU source digest, fills newly introduced params
fields from their committed defaults, preserves the approved dedup release
settings, updates `new-params.json` and `expect.json`, retains all accepted
historical and repaired params/code pairs, and rewrites the generated rollback
pins and compatibility manifest. It also checks the exact raw incident pin
`d5979c30…`, its incident GPU digest, old-code translation to `bba7a409…`, and
the historical decoder hashes. New fields are removed for `c5252865` replay only
after the full new params hash and paired code digest match an accepted release.
Older accepted records that omit later fields remain accepted. Unknown or
changed params fail closed.

The tool compiles the kit, shell-checks deployment entry points, runs
`git diff --check`, verifies the rollback manifest, and runs all **14 offline
frozen-input tests**. A successful run ends with
`REPIN STATIC/FROZEN-INPUT CHECKS PASS`.

**This is not deployment qualification.** Commit the resulting kit, then rerun
fresh committed-kit normal/abort rehearsals, rollback and validator-restore
qualification, and the applicable production-code tests. Physical-GPU behavior
is not established by this command. Do not deploy an uncommitted or modified
working-tree kit.
