# Bounded training selection from full eligible contributions

Cheap eligibility and duplicate exclusion apply to the entire frozen population.
Every eligible pair remains in the signed learner inventory and continuous audit
handoff. Rewards and audit sampling retain that full population; training may
consume at most 256 documents in one job. Unselected documents are not invalid.

The coordinator persists a fresh postfreeze seed before selecting. Its immutable
journal binds the original computation, captured receipt digest, full eligible
inventory digest and cap. Hash ranks select up to 256 entries; the chosen entries
retain original deterministic order. The training coverage uses that exact seed
and selected inventory. Retry cannot redraw or rebind the persisted seed to a
changed population. Publication reports distinguish eligible_count,
training_count, unselected_count and selected/full inventory hashes.

Existing already-published learner journals remain unchanged on recovery. This
controller change does not alter miner generation/sampling, original source
contract, proof objects or GPU optimizer/training semantics. Current original
GPU jobs already support the <=256 selected document limit. A controller-only
operational overlay for an existing epoch requires an explicit authority scope
binding the original manifest and immutable captured receipts; source publication
for future epochs must contain the permanent implementation.

Prospective cheap admission reads use four concurrent object GETs, with original
sorted miner/slot processing preserved. Transport allocation is capped at the
signed small-document byte size (maximum 2 MB), pending raw results total at most
8 MB, and JSON decoding remains serial under the existing 128 MB working-budget
profile. This does not parallelize model execution or certify unaudited samples.
Transport and immutable-object faults still abort collection rather than count
as fraudulent submissions. A diagnostic sidecar records capture, metadata,
read/decode, selection and publication timings; its publication does not gate
training and it is separate from computation/scoring bindings.
