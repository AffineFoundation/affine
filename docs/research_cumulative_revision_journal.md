# Default-off cumulative research revision journal

`ops.paired_quota_revision_journal.ResearchRevisionJournal` persists the existing
research cumulative-slot checks. No production module imports it. It does not
change wire formats, K1/L1, max attempts, admission or independent audit progress.
It performs no model execution, grading, proof verification, optimizer update,
checkpoint publication or application exactly-once claim.

The future integration boundary is the already authenticated captured learner
input. The caller first authenticates the original manifest, miner public-key
commitment, ROOT cheap-admission envelope, original child slot, captured document
full SHA/size and checkpoint/source. It derives the task hash from native CPU
reset and supplies the pinned tokenizer decode when constructing
`CurrentBatchAdapter.from_manifest`. UID is never a slot identity. These checks
retain the existing unaudited assurance; they do not wait for inference audit.

The journal takes the immutable ORIGINAL canonical token-document bytes, not
caller-supplied member IDs or a replacement batch. `authenticate_original(bytes,
evidence, proposed_scope)` is a trusted integration callback. It must verify the
above original evidence and independently compare `proposed_scope` with the
approved manifest/task/public-key context, then return exactly
`boundary_receipt(original_sha256, scope_sha256)`. Constructing that receipt alone
is not authentication. The journal cannot assess whether an arbitrary caller's
callback tells the truth; a future live bridge must supply the actual signature,
child/digest and source binding validation. Self-asserted booleans are refused.
Neither claimed classification nor public draw receipt establishes actual sampler
execution or native grading. No caller is installed by this change.

`ResearchRevisionJournal.create(path, enabled=True)` creates a new private SQLite
file exclusively. Reopen with `ResearchRevisionJournal(path, enabled=True)`.
`append_authenticated(adapter, miner_public_key, manifest_sha256, original_bytes,
evidence, authenticate_original)` requires exact pubkey-shaped hex, consistent
adapter task/harness/draw binding and a frozen scope. It normalizes the original
batch and recomputes all execution/content and shared token-trace digests. A
`BEGIN IMMEDIATE` transaction loads the prior head, runs `CumulativeTaskSlot`
checks, stores original bytes plus bounded canonical authentication evidence, authenticated-boundary receipt and immutable
revision, then advances the head. FULL synchronous WAL commits preserve this
single-database operation; independent connections serialize writers.

An identical original replay is a complete journal no-op, including replay of an
older original after the head advanced. It reports both historical revision and
current head. A different byte representation with the same current semantic
revision records its distinct original without making a second logical revision.
An unseen revision removing or changing any accepted attempt refuses atomically.
When concurrent cumulative extensions omit one another, one commits and the
other refuses; the caller may later present a separately authenticated superset.
This is revision admission, never a retry of optimizer work. Different manifest
SHA or task/harness/draw configuration cannot rebind an existing slot. SQL triggers
protect originals, revisions, observations and slot bindings from UPDATE/DELETE
through this database schema. This does not claim protection against an actor
replacing the file or altering the schema outside the library.

Do not use revision history as an optimizer application ledger. The separate
research training ledger's reservation/recovery and genuine original optimizer
state/publication evidence remain separate. Activation requires a versioned live
bridge and failure review; no contract activation is included here.

Controls exercise restart monotonicity, old redelivery, token/class rewrites,
concurrent repeated originals and incompatible extensions, actual subprocess
death after SQL inserts before commit, exact original preservation, immutable
record triggers, fabricated member IDs, scope rebinding, mutation of adapter
configuration, opt-in and real Ed25519 verification at a test boundary. The crash
control uses a synthetic trusted boundary callback to isolate database recovery;
it never represents a live signed commitment or native proof.
