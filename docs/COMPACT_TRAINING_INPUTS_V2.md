# Prospective authenticated compact training inputs v2

Status: prospective implementation integrated in this checkout, not sealed or
deployed. E9 is actually running
`authenticated-verifier-receipts-v1` against sealed source `94ff` and original
job `nonpayable-live-reward-math-v1--1791128338-9-train-b4b84b02`.
This candidate does not amend that job, its requests, signed manifests, sources,
services, thresholds, authority, or download behavior. The future coordinator/router/backend path is implemented locally. No operational
publication, signing, deployment, queue mutation or network operation was performed
here; local controls use synthetic keys and an in-memory bucket.

`subnet/compact_training_inputs.py` defines the explicit prospective policy
`authenticated-verifier-compact-inputs-v2` and canonical JSON artifact version
`canonical-accepted-batch-documents-v2`. Existing v1 code is untouched.

## Security boundary and artifact

The coordinator reads its own authoritative SQLite `COMPLETE` verify row. It
checks the stored job/report digests and registered worker/report-request binding,
then reuses the original authenticated job/report admission checks in v1. A signed
worker request by itself cannot supply completed queue authority. The low-level
row builder assumes its row comes from that trusted database; use the read-only
`prepare_from_queue` adapter for integration rather than accepting a remote row.

The builder serializes only original accepted batch documents from the original
signed worker report, ordered by their original batch numbers. It retains every
canonical document field, including rollout tokens, contexts, rewards, sampler
receipts, proofs embedded in documents, task metadata and other verifier-bound
metadata. It neither projects fields nor regenerates documents. Separate NumPy
full-vocabulary probability arrays in the original ZIP are absent. Any proofs
already embedded in batch documents remain as part of their exact original hash.

The proposed operator-signed v2 receipt nests the unchanged signed v1 receipt
and commits the exact compact artifact SHA256, length and artifact version.
The nested v1 signature retains original frozen ZIP digest, length and key;
original signed manifest, verify job, worker request and report hashes; registered
verifier, source/runtime pins, miner, computation binding, completion times;
and exact accepted-batch and positive/negative rollout digests. Two small
signatures deliberately reuse existing v1 validation without altering its format.
The artifact also commits the original receipt hash, ZIP hash, epoch and checkpoint.

The trainer authenticates both signatures, the original frozen-manifest binding,
the explicit prospective policy and the compact object commitment. Reads are
bounded to the signed size plus one byte, with a fixed 2,000,000-byte document
ceiling. Canonical JSON must match byte for byte; duplicate keys, nonfinite
numbers, alternate serialization, excessive nesting, extra documents, missing
documents and substituted contexts fail. The ceiling is a separate prospective
transport bound based on the original ZIP manifest's existing 2 MB ceiling; no
accepted pair is silently dropped when it is exceeded. Oversized candidates fail
and need a separately reviewed prospective policy before use.

Every document must match its original accepted batch hash and slot. Existing
epoch/checkpoint/environment/index/version/quota checks remain. Every selected
positive and negative rollout must match the original full canonical rollout hash.
The resulting `(environment definition, positive rollout, negative rollout)` pairs
are the same data consumed by v1 training. The trainer does not open the original
ZIP or compute inference verification, TOPLOC verification, sampling replay or
environment replay. Reference log probabilities, training forward/backward and
optimizer work remain part of the existing objective.

This preserves the same trust in operator admission and registered verifier
results as v1. A compromised authority that also forges the nested v1 receipt
can attest false evidence; compact transport does not solve that existing trust
boundary. Canonical hashes prove consistency with the signed original admission,
not independent correctness of verifier computations. The original frozen ZIP,
original signed requests/reports and source archive must remain durably available
for independent audit; compact objects are derived training transport, not their
replacement audit record.

## Metadata and future integration

`admitted_submission` retains `submission_sha256` and `submission_size` as the
**original ZIP** provenance and adds separate `compact_sha256`/`compact_size`.
It retains accepted batch documents and original report/job identifiers, adds
both receipt hashes, and explicitly reports that training performed no verifier
work. Consumers must not reinterpret the original ZIP hash as the compact hash.
`validate_report` binds those exact fields, documents and truthful training flags;
it refuses fresh trainer audit claims. `validate_job` requires the new module
and unchanged v1 module pins, explicit v2 policy, bounded submission inventory,
and one original miner/artifact admission each. Existing v1 execution amendments
are explicitly refused for prospective v2 jobs. They cannot authorize this new
policy through an implicit transport change.

Minimal future integration, subject to review and a separately sealed/admitted
source and a future signed epoch/job contract. Steps 1–4 are now implemented in this
checkout for the explicit v2 path; source admission, external audit readers and
activation remain root review/qualification gates:

1. Issue the existing v1 receipt from original COMPLETE evidence, then call
   `prepare_from_queue` and sign the returned v2 payload using the operator's
   existing authorized signing path. Store/read back the compact bytes durably
   before including their hash, size, URL and v2 receipt in the training request.
   The low-level preparation helper returns unsigned evidence. The coordinator integration signs and publishes only when an explicitly selected prospective v2 epoch invokes it; no operational v2 epoch has been activated.
2. Add an explicit v2 job path at coordinator/router/backend dispatch and worker
   download/admission. A v2 submission object's transport `sha256`/`size` identify
   the compact artifact; nested v1 receipt fields identify the original ZIP.
   Keep v1 branch and schema unchanged. Use the validators here only on v2 jobs.
3. Calculate disk admission and retirement using compact transport sizes while
   preserving original ZIP provenance for source-bound audit/reward contracts.
   Retain the existing weights/export/workspace/safety reserves and exact training
   objective, coverage/task-normalization and persistent optimizer semantics.
4. Dispatch resulting unchanged pairs through the existing objective and report
   v2 truthful admission metadata. Extend independent audit readers explicitly
   for v2 receipt nesting and original ZIP provenance. Do not rewrite historical
   v1 job/report records or teach v1 amendments to silently opt into v2.
5. Qualify actual prospective canonical population, durable readback, source pins,
   worker admission/reporting, capacity, retention and independent source-bound
   audit before activation. Running E9 continues its original signed v1 path.

Root independently measured E9's original accepted documents: 186 canonical
batch documents total 8,245,308 bytes, versus 78,664,386,387 original frozen ZIP
bytes across the selected submissions (approximately 9,540 times smaller for
batch documents alone). This is a measured prospective payload comparison,
not a deployed download reduction. Root also independently authenticated all 121 COMPLETE original v1 receipts and
full report documents in a read-only comparison: estimated v2-framed artifact
bytes total 8,297,398, with largest per-submission artifact 149,795 bytes, below
the 2 MB ceiling. This framing estimate is neither a signed v2 authorization nor
an uploaded compact object. The actual historical E9 preparation call refused
the missing prospective original signed marker. Evidence remains private at
`state/root-audits/compact-e9-readonly-payload-inventory-v1/read-only-comparison.private.json`.
Final future-epoch source admission and durable compact readback remain required.

## Local controls

Run from the checkout:

```sh
PYTHONPATH=tests .venv/bin/python -m unittest test_compact_training_inputs test_compact_training_integration test_compact_old_source_compatibility test_compact_training_audit_reader test_training_verifier_receipts -q
```

64 compact/v1 controls pass: 40 compact document controls, 15 prospective
integration controls and nine existing v1 controls. A broader targeted run of
163 backend, covered/persistent, security, capacity, original-source compatibility
and independent-reader controls also passes. Synthetic test signing keys and claimed verifier results are local CPU
fixtures, never operational admission evidence. Controls cover exact unchanged
pairs without ZIP access or verification; queue completion/digest/role/worker
binding; local-audit replacement and original-report mismatch; forged outer and
nested signatures; signed compact token/reward/sampler/task/context/slot/population
mutations; independently altered rollout commitments; canonical framing,
duplicate keys, nonfinite numbers, nesting, size bounds and symlinks; old v1 ZIP
compatibility; explicit policy/source admission; duplicate jobs; old amendment
refusal; and source-bound report/provenance/truthfulness mutations.


## Concrete implemented integration

The changed runtime modules are `compact_training_inputs.py`, `gpu_service.py`,
`remote_backend.py`, `role_router.py`, `backend_jobs.py`, and
`persistent_training_controller.py`, `persistent_training_protocol.py`,
`persistent_training_evidence.py`, `persistent_training_worker.py`.
`training_receipts.py`, `training_policy.py` and both objectives remain unchanged.
No global expansion of v1 `SOURCE_FILES`/v4 `EXECUTION_FILES` is required; every
role under an explicit v2 manifest must additionally pin the compact module and
v1 receipt module, and fresh-source loading purges/reloads the admission modules.
Actual metadata inventories already enumerate all `subnet/*.py` files.

`gpu_service.contract` and `RemoteController.open` carry an explicit future config
marker into the first signed public epoch manifest. Compact preparation also
requires that marker in the ORIGINAL authenticated verify job's signed manifest.
A historical v1 report, including E9, cannot authorize v2 by being relabelled.
The selected policy supports covered v3 and persistent v4 objectives only.
No configured policy marker means the previous v1 behavior continues unchanged.

`prepare_submissions` reuses original v1 issue, derives from authoritative COMPLETE
queue records, publishes a content-addressed private compact object, performs a
bounded full durable readback, and only then uses the coordinator's existing
operator signing API for the v2 receipt. It returns no submission when the byte
readback fails. This production-capable function was exercised only against local
synthetic test authority and in-memory transport, never against live credentials.

Backend admission authenticates compact receipt nesting before passing ORIGINAL
ZIP identities into the unchanged coverage validator. Training downloads `.json`
with its exact signed size as the GET ceiling. Verify roles still download and
audit original ZIPs. Both training objectives receive unchanged fully authenticated
pairs. Training reports contain admissions and no fresh audits; v4 report/evidence,
state export/readback, authority commitment, counters and cached same-job recovery
all preserve their existing scientific semantics. Resume binds both compact and
original receipt inventories; cached requests cannot substitute another transport
or original report.

Covered capacity reserves retained compact download bytes (at least one 2 MB
artifact reserve), one 2 MB serialized-document working buffer, the unchanged
checkpoint/final/temp-export/missing-input reserves and 2 GiB safety reserve.
Persistent capacity conservatively reserves all 256 possible compact objects
(512 MB), the 2 MB serialized working buffer, and its unchanged model/state/export
reserve. It adds a separate 128 MB CPU materialization reserve for compact JSON
containers, rather than treating 2 MB of serialized bytes as a RAM bound.
Persistent input retirement unlinks only after exact authenticated admission;
covered inputs follow the existing completed-job retention lifecycle.

The 15 integration controls exercise real coordinator queue derivation and durable
readback before v2 signing; fail-closed readback; historical-v1 relabelling refusal;
backend source/coverage admission; all-role pins; actual backend compact download
and unchanged covered objective dispatch with ZIP/audit functions forbidden; first
public manifest policy binding; authorized remote job creation; compact coordinator
capacity/accounting; router export/safety reserves; authenticated persistent input
retirement; receipt-bound remote resume; persistent v4 output/optimizer provenance;
and original completed v4 controller/state recovery without another update.
They use CPU synthetic fixtures and mocked GPU optimizer execution. They do not
establish actual full-model GPU qualification, durable production readback,
performance, public-epoch completion or held-out gains.


## Original v1 compatibility and independent reader authentication

A root review found that the first integration imported the compact selector even
on v1 paths. That dependency is removed: all existing callers decide policy using
literal signed values. V1/unspecified policy never imports the compact module, and
v1 source loading never purges or requires it. Explicit v2 backend admission
requires its source pins before any compact import; original request/report
recovery also checks the compact job pin before importing its validator.

`test_compact_old_source_compatibility.py` uses the actual local immutable archive
`94ff74eb335e24d4702da2ec10cc0aee068076b003e6c0b13b0e81f5090bc79c`, copies its
exact 141 direct runtime modules into an isolated temporary package, and overlays
only the eight prospective caller modules. The compact module is absent. An
import finder raises on any attempted compact import. Two fresh subprocesses pass:
real v1 receipt admission through `install_source_loader`, and backend execution
through the real loader with a toy CPU model/mocked covered objective. Source
hash checks and loader execution remain real. A v2 missing-pin control refuses
before attempting any compact import. Synthetic signatures authorize the isolated
fixture; this neither alters nor reauthorizes the original sealed archive. The
local historical-archive control explicitly skips when that retained private
archive is unavailable; all other controls remain portable.

`ops/compact_training_audit_reader.py` now provides the tracked, read-only adapter
for independent ledger tooling. It authenticates the signed training job/manifest,
requires an independently admitted exact runtime source inventory, explicitly
validates v1 or v2 report semantics, and reconstructs each original admission from
trusted original COMPLETE verifier rows and registered signed worker requests.
Its output identifies ORIGINAL ZIP SHA/length/key/miner/job/manifest/report/request
lineage separately from compact transport SHA/length. V1 reader paths never import
the compact module. Existing private historical readers remain unchanged.

Report consistency is distinct from report authenticity. A raw trainer report
claiming a matching signed job SHA is refused. The reader requires either an
exact authority-signed report envelope or an authority-signed
`training-report-audit-attestation-v1` containing exactly `version`,
`training_job_sha256` and `training_report_sha256` for the original payloads.
No reader API signs or obtains that attestation. A caller's existing authenticated
ledger/report acquisition must supply it. Source admission, worker admission,
scientific/economic ledger checks and actual remote report collection remain
explicit caller responsibilities. Six local controls cover original/transport
projection, original COMPLETE/roster binding, nested provenance tampering,
independent source inventory, old v1 support, and refusal of unsigned raw reports
and mismatched authority attestations.
