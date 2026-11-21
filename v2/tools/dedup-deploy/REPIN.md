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

`new_commit` identifies the production source to install; `kit_commit` identifies
the exact deploy-kit checkout. They are normally the final stacked SHA. To re-pin
from a later kit-only commit, keep its source and rollback overlay byte-identical
to the production-code commit and pass both identities explicitly:

```bash
python -I v2/tools/dedup-deploy/repin.py KIT_ONLY_SHA --code-sha PRODUCTION_CODE_SHA
```

The tool compares both commits and the working tree against the production-code
commit. It refuses a kit-only commit that changes production source or the
rollback overlay. Deployment still requires the separately reviewed workflow
that installs `new_commit` from the `kit_commit` checkout.

The production lineage must include the incident `50afaca3`, exact repair
`14ae80f5`, and full-fold child-process isolation `71d287f6`. The tool explicitly
checks the crowned-install `in_children` call. The divergent `6fa59227` lineage
omits the full-fold fix and is refused; do not substitute its pins for this
integrated stack.

The command recomputes the GPU source digest, fills newly introduced params
fields from their committed defaults, preserves the approved dedup release
settings, updates `new-params.json` and `expect.json`, retains all accepted
historical and repaired params/code pairs, and rewrites the generated rollback
pins and compatibility manifest. It reads the preserved raw records for incident
pin `d5979c30…` and first-repaired pin `8caddd9c…` and also validates the combined
pin `4c83d52f…`; it never guesses historical values by adding or removing fields.
Both raw and decoded historical canonical hashes must remain unchanged.

For `c5252865` replay only, the rollback overlay removes `DEDUP_FIELDS`,
`NEW_PARAMS_ONLY_FIELDS` (including `crown_gate` and `z_field_bar` in this stack),
and `z_bar`. This happens only after the complete new params hash and paired code
digest match an accepted release. The pre-field-gate releases carry `z_bar=1.5`;
the integrated field-gate release carries `z_bar=0.0`. All translate to the
reviewed old params pin `bba7a409…` with `z_bar=1.0`. Older accepted records that
omit later fields remain accepted. Unknown or changed params fail closed.

The tool compiles the kit, shell-checks deployment entry points, runs
`git diff --check`, verifies the rollback manifest, and runs all **14 offline
frozen-input tests**. A successful run ends with
`REPIN STATIC/FROZEN-INPUT CHECKS PASS`.

**This is not deployment qualification.** Commit the resulting kit, then rerun
fresh committed-kit normal/abort rehearsals, rollback and validator-restore
qualification, and the applicable production-code tests. Physical-GPU behavior
is not established by this command. Do not deploy an uncommitted or modified
working-tree kit.
