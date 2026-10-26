# Prospective complete-group admission

Preparation only: this package is not imported by production, is not a deployed
wire format, and does not change the running fixed-eight GRPO experiment. A live
cutover remains conditional on qualified learning results and full integration.

`contract.py` provides pure, CPU-only validation for a proposed versioned
single-turn math contract. The group size N is independent of minimum success and
completed-failure counts. Every canonical nonce `0..N-1` must appear exactly once
and in serialized order. N is immutable in the caller-authenticated epoch
contract. Parallel physical generation is allowed; sequence order in a document
does not establish historical execution order.

## Trust boundary

The caller must independently authenticate:

1. The epoch manifest and its immutable checkpoint, runtime, sampler and group
   contract. This module accepts a strict projection, not an arbitrary full
   legacy manifest.
2. The submitting miner's registered Ed25519 public identity (hex64), and the
   environment's trusted task index/version/hash and exact prompt-token hash.
3. Each verifier's signed report, registered worker/job/source identity, exact
   checkpoint/profile, challenge binding and frozen artifact digest.

`validate_batch` checks metadata and token framing only. Its `Admission` always
has `sampling_verified=False`; claimed successes/failures remain untrusted.
The separate `audited_verdict` accepts caller-owned `VerifierAssessment` instances
for every original attempt. Miner-provided JSON dictionaries cannot stand in for
these records. A Python dataclass by itself is not an authentication mechanism;
the production adapter must perform the signature and evidence checks above.
Missing/unchecked/indeterminate verification remains indeterminate. A known
invalid sampling or terminal result rejects the group. Grading mismatches are
checked per nonce, so exchanging positive and negative labels cannot preserve
acceptance merely by keeping aggregate counts equal. No function here runs a
model, validates TOPLOC or claims to prove historical physical execution.

## Future schema and existing-code adapters

The reviewed production schemas were `forced_sampling.py`,
`sampling_uniqueness.py`, `miner.py`, `verifier.py`, `protocol.py`,
`math_completion.py` and `training_receipts.py` in the pinned production-insertion
source tree. This prototype deliberately retains familiar names:

| Existing field | Prospective interpretation |
|---|---|
| Batch `epoch`, `checkpoint`, `env_id`, `environment_version`, `index`, `sample_index` | Same trusted epoch/environment/task binding |
| Rollout `seed` | Canonical attempt nonce, exactly one of `0..N-1` |
| `sampling.version`, `binding_sha256`, `attempt` | Receipt for the entire fixed-group contract, miner and trusted task |
| `classification` positive / negative / neutral | Correct / completed incorrect / unresolved |
| `reward` 1 / 0 / null | Explicit class consistency; neutral is not a completed failure |
| `turns[].prompt`, `output` | Exact trusted prompt and supported generated tokens |
| New `turns[].stop_reason` | Explicit `eos` or `budget`, independently checked against actual tokens |

The exact new `schema` and contract `version` reject historical v5 inputs. Legacy
manifests must continue through their original validators. Do not silently map
K=L=1 into an eight-response group, or silently strip extra submitted fields.
Before adopting this format, define the full canonical wire artifact and preserve
its original digest through any projection. Production proofs/transport may
contain additional fields; their authenticated original hashes and explicit
projection must be bound by the adapter, not discarded to make this parser pass.

Challenge derivation binds epoch randomness, checkpoint, runtime, sampler, group
rules, miner, environment, task index/hash, exact prompt hash, nonce and token
position. Extra salts, group numbers, seed offsets and prompt prefixes are
rejected. This specifies accepted attempts; it cannot prohibit private research
computation or prove that the miner executed a particular historical program.

Identical token text from distinct independently verified attempts remains valid.
Reusing a nonce is invalid. Budget-capped responses are neutral even if the text
contains an earlier box. EOS alone does not prove a completed answer; the trusted
native grader must distinguish a completed wrong answer from an unresolved one.
The initial scope is deliberately single-turn math. Multi-turn environments need
their own explicit termination and observation binding before admission here.

`credit_id` identifies the canonical miner/epoch/checkpoint/task/contract group
independently of upload slot and token contents. The network admission path is
`validate_batch` followed by `register_eligible_once`: a structurally eligible
group with claimed outcome minima adds one **provisional** unique-group count
without waiting for any audit. Its record explicitly remains unaudited and
unpaid. Existing training policy can consume these eligible unaudited inputs;
this prototype does not add an audit barrier to training or contribution counts.

`record_audit_once` separately retains authenticated accepted, invalid and
indeterminate evidence, including multiple distinct reports for the same group.
It does not overwrite the provisional ledger or assign a reward. Existing
continuous validity estimates and penalty policy consume this evidence; a new
report is not automatically a last-report-wins decision. `audited_credit_once`
is only a diagnostic helper for tracking the confirmed subset, never the path
for admitting or counting the whole network.

Each helper is a pure ledger transition. Identical retries return zero new
entries; conflicting bytes under a registered group or evidence ID are rejected.
The caller must authenticate the original submission and reports, then persist
each transition atomically with a unique database constraint. These helpers do
not replace cross-miner task-collision rules, audit probabilities, penalties,
hourly moving averages or current UID mapping, and do not accept economic
instructions from miners. A valid nonqualifying group is not fraud and earns no
group point.

## Validation

```
.venv/bin/python -B -m unittest research.group_contract.test_contract -v
```

The 27 CPU controls exercise generic N, independent minima, historical rejection,
cherry-picked/missing/duplicated/reordered nonces, Boolean aliases, extra salts,
stale checkpoint/epoch/identity/runtime/task/prompt, legitimate repeated text,
caps, unsupported terminals/tokens, unchecked and invalid proofs, swapped outcome
labels, stale assessment digests, duplicate credit and strict JSON decoding.
They also verify unaudited provisional admission, independent valid/invalid/
uncertain evidence, idempotent retries and conflicting-evidence rejection.
They verify this pure contract boundary, not numerical model verification or
economic exploit resistance of the entire network.
