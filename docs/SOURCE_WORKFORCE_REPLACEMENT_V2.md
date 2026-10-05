# Prospective source workforce replacement

`live-compute-source-approval-v1` remains unchanged: an ordered chain adds new source snapshots and monotonically preserves all prior verifier identities, with four to six identities per source. Existing v1 approvals, cutover documents, source-specific grants, anchors, runtime pins, original job signatures and epoch timing continue to verify as before.

For a new source only, `live-compute-source-approval-v2` permits a different active roster of four to six distinct verifier identities. This resolves the case where a five-identity source grant includes a deleted node whose signing key is no longer available: its successor source can retain four live identities and add two newly qualified identities. The old source still authenticates its original five identities and original reports. This is not a global revocation or retroactive rewrite.

V2 has all v1 fields plus:

- `previous_authorization_sha256`: SHA256 of the complete immediately preceding signed source approval envelope; for a first source approval, SHA256 of the original signed cutover.
- `retirements`: root-signed `live-compute-verifier-retirement-v1` envelopes, exactly one for each identity removed from the immediately preceding active roster. Empty when no identities are removed.

Each retirement payload has exactly these fields:

| Field | Required binding |
| --- | --- |
| `version` | `live-compute-verifier-retirement-v1` |
| `original_cutover_sha256` | Original signed cutover envelope |
| `previous_anchor_sha256` | Immediately preceding signed anchor |
| `previous_authorization_sha256` | Immediately preceding signed approval, or cutover for first approval |
| `previous_source_sha256` | Last source in the immediately preceding signed anchor |
| `verifier_identity` | One identity actually removed from the previous active roster |
| `effective_at` | Exactly the new source approval's finite prospective effective time |
| `reason` | `node_terminated`, `signing_key_unavailable`, `key_compromised`, or `operator_retired` |
| `evidence_path` | Existing canonical absolute regular file, without symlinks |
| `evidence_sha256` | SHA256 of every byte of that file |

The operator must review the actual retirement evidence before signing. The reader proves that the root authority approved these exact evidence bytes and bindings; it does not independently prove a remote host was physically destroyed or that all copies of a key were deleted. Infrastructure timeouts alone do not create a retirement or miner penalty. Evidence records must remain available unchanged for historical chain validation.

New approval times cannot precede the previous approval time. Missing or duplicate retirement records, surviving identities marked retired, unrelated identity records, mismatched source/anchor/approval/time, altered evidence, unsigned records and other authorities are rejected. An identity retired earlier in this chain cannot silently return in a later active roster. Reinstatement requires a separately reviewed future protocol, not an implicit v2 roster addition.

After entering v2, subsequent approvals must remain v2; v1 cannot express retirement semantics. A v1-only historical chain preserves exactly its prior behavior. Source membership remains additive: v2 cannot reuse an existing digest to overwrite a previous grant. Every new source still requires a signed byte-exact admitted archive, complete runtime inventory, unchanged package pins and original cutover identity. Every authorized job remains tied to the specific source, prospective opening time and original namespace.

The operator consumer is outside the scientific model/sampler/TOPLOC bundle. Deploying reviewed reader support does not itself sign an approval, activate a new source, change current E10 authorization, restart any GPU node or set weights. Root must prepare qualified replacement nodes, obtain their actual public identities, collect actual retirement evidence, publish/review the new source, sign the ordered v2 documents and activate only at a safe epoch boundary.
