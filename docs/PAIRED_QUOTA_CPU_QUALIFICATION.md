# Prospective paired-quota CPU controls

`ops/paired_quota_qualification.py` is a CPU research helper, with **no production
call sites or activation**. It supports the controlled experiment in
[PAIRED_SAMPLE_STABILITY_PLAN.md](PAIRED_SAMPLE_STABILITY_PLAN.md). Default quota
is one success and one failure. K2/L2 is only an explicit helper argument.

An authenticated adapter must supply the pinned task, checkpoint, taskset,
harness, sampling context and approved prescribed attempt IDs. The helper
recomputes execution and content identities; submitted IDs are never trusted.
Execution IDs distinguish approved attempts, while content IDs exclude miner,
attempt, classification, reward, filenames, proof wrappers and upload time.
Content includes all ordered prompt/output tokens and actions/observations.
The epoch belongs to the contribution slot; the approved sampling-context digest
belongs to execution. Checkpoint/taskset/task/harness bind both identities.
Canonical identities are namespaced SHA256 hashes of strict JSON metadata.

The whole cumulative task-slot revision is checked before selecting pairs.
Exact redelivery adds no member. A repeated execution with changed content or
label is refused; repeated content from a different attempt is a duplicate and
cannot fill more quota, **not a fraud verdict**. Conflicting classifications
cannot turn one content into both success and failure. Pairing is deterministic,
independent of upload order, and uses each selected member once. K2 produces two
nonoverlapping pairs, each with half the task's weight. Reward contribution
remains one task unit. Cross-UID copies preserve content identity while the
miner-specific task slots differ; the helper does not itself assign cross-miner
zero scores.

These are cheap structural controls, not proofs of authentic sampling or native
outcomes. The actual prescribed draw and inference verifier still has to check
every audited trajectory; grading still authenticates outcomes. Supplying four
synthetic traces with plausible labels to this helper alone is not a qualified
sample-generation test.

## Remaining integration and research evidence

- The production rollout schema uses harness-specific fields. Write and qualify
  an adapter that extracts complete canonical actions/observations and verifies
  approved context/attempt bindings from actual manifests. Do not just trust
  unverified caller metadata or drop tool turns.
- Recompute identities across the entire cumulative commitment, bind selected
  revision/content IDs into signed training inputs, and freeze that revision.
  Canonicalization must remain consistent across miner, eligibility and audit.
- Persist a selected task-slot and training-job association with transactional
  uniqueness, preserve completed receipts on recovery, and prove actual trainer
  restart/redelivery cannot apply the selected revision twice. The returned
  stable `revision_id` is **not** a persistent ledger and does not implement
  optimizer idempotence. These tests establish only repeat-selection identity.
- Preserve existing cross-miner same-task duplicate-zero scoring and audit
  evidence deduplication. Neither is implemented by this helper.
- Qualify the existing producer on real K2/L2 attempts, then run matched
  same-parent/Adam K1/L1 and K2/L2 branches. The CPU tests say nothing about
  held-out gains, GPU/runtime compatibility or achieved task coverage.
- Activate only through a reviewed future contract, published miner guidance,
  adjusted budgets and a real end-to-end epoch. No live configuration changes
  are made here.

CPU controls cover four distinct members, task normalization, repacking,
reordering, label conflicts, repeated attempts, repeated content, cross-UID
copies, old checkpoint/epoch refusal, exact redelivery, complete trace bindings
and schema/budget refusals. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -X pycache_prefix=/tmp/paired-quota-fresh-review \
  -m unittest discover -s tests -p test_paired_quota_qualification.py -v
```

Use a fresh cache prefix for each qualification run. This adds neither a live
proof exemption nor a deployment authorization.
