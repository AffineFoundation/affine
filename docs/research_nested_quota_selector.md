# First-prescribed-attempt nested quota selector (default off)

`ops.paired_quota_nested_selector.select_nested` is a research-only function with
no live import. Calls require explicit `enabled=True`; default calls refuse. It follows frozen ascending approved attempts, rather than the
existing research `select_pairs` content-SHA ranking. Neither selector changes
production wire, client, quota, miner, scoring, admission or continuous audits.

Inputs: an `ApprovedTask` constructed from an independently authenticated source/
manifest/native reset, registered miner public key, EOS token IDs from the pinned
authenticated tokenizer, and frozen `{attempt,row,evidence}` records. `row` is an
original normalized admitted trace; None represents a captured admission or
infrastructure failure. `authenticate_admitted_native(record, expected_binding)`
is a TRUSTED integration callback, not implemented admission or grading here.
It independently authenticates original captured input provenance and native
grader evidence, checks expected task/checkpoint/epoch/harness/draw/EOS/owner
scope, and returns the exact grade receipt. Constructing the receipt by copying
expected fields is not authentication. A future bridge must supply actual checks;
there is no inference proof or newly graded result produced by this function.

Receipt schema: exact `version=authenticated-admitted-native-grade-research-v1`,
`attempt`, `selection_scope_sha256`, `row_sha256`, recomputed `execution_id` and
`content_id`, plus `status`, `classification`, `reward`, `native_done`. Status is
native-graded, indeterminate, admission-rejected or infrastructure. Native-graded
requires original claimed class to agree with actual binary native class/reward,
and a boolean actual native terminal result. Other statuses require those three
outcome fields to be None. Authentication refusal/conflict rejects the whole
selection, so callers must preserve that original refusal record rather than
silently skipping it or replacing the attempt.

Member IDs are recomputed with existing identities; token dedup uses the shared
prompt/output digest. Repeated same execution is a no-op; changed content/outcome
under the same attempt refuses. Equal content/token traces across attempts do not
add diversity, and conflicting native labels or changed observations for the same
token trace refuse. Eligibility requires native_done=True and a real final output
token matching a pinned EOS ID; claimed done/eos/stop_reason metadata is ignored.
Missing and indeterminate/incomplete/non-EOS attempts remain explicit in `supply`.
Completion requires an authenticated captured status for EVERY attempt in that
arm's prefix. A missing lower attempt blocks its revision and prefix-completion
claim even if later observed rows could fill quota; a gap between K1/K2 blocks
only K2. Explicit authenticated failed/indeterminate status counts as observed.
Missing attempts after an established K2 prefix can represent stopped collection.

Output schema version `first-prescribed-native-EOS-nested-quota-research-v1`:
selection scope SHA, slot ID, complete_K1/complete_K2, nullable K1L1/K2L2 revisions,
first_K1/first_K2_prefix_length (length of the frozen attempt prefix, not seconds),
per-arm completion-prefix gaps, per-attempt supply status, observed unique counts, duplicate counts, selected member
IDs/attempts/rowSHA, and per-arm task-contribution units. Not-observed is not a
claim that generation failed; parent population reporting must retain every task,
including tasks with empty/incomplete observations. No walltime is inferred.

K1 uses p1/n1; K2 uses p1/n1 and p2/n2. There are exactly two disjoint pairs,
never four Cartesian pairs. Revision schema remains selected-task-revision-v1
with quota1/2 and within-task pair weight1/0.5; existing ResearchTrainingLedger's
validate_revision accepts it. Average across the SAME T tasks to give each task
1/T weight and each K2 pair1/(2T). A nullable revision never authorizes training.
Caller freezes the common completed task population and no incomplete tasks are
silently omitted by this module.

Durable integration remains prospective: authenticate the captured original
boundary, append its originals/evidence to ResearchRevisionJournal, then recompute
this selection from the frozen admitted/native-graded attempt record. Bind selected
revision and immutable plan/settings hashes to research job reservation and actual
parent optimizer lineage. SQLite revision append and at-most-once job claims do
not make optimizer/publication application exactly once. Native grading does not
prove checkpoint sampler execution; expensive audits remain independent.

Tests include real Ed25519 authentication of synthetic native records, original
row/scope binding, upload permutation, a lower content hash deliberately arriving
later, nested selections/ledger compatibility/task weights, duplicate attempts/
content, conflicting labels/observations, indeterminate/capped/incomplete/missing
supply, nonbinary reward inconsistency, bogus member IDs, fake EOS metadata,
unapproved/wrong task and reordered attempt stream, and input immutability.
All fixtures are CPU synthetic, not real model/grader qualification.
