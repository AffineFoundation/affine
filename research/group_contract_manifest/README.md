# Prospective manifest resolver

This is a CPU-only integration candidate. It is not imported by any live miner,
verifier, trainer or controller. It does not activate complete-group mining or
change the running private fixed-eight experiment. Deployment remains conditional
on the learning qualification and a separately reviewed production composition.

`for_manifest()` returns a frozen `LegacyQuotaContract` or
`CompleteGroupContract`. It resolves batch semantics before model download or
capability use; caller authentication must already have verified the signed
manifest and selected its approved source. It neither verifies signatures nor
checks calibration evidence, checkpoint/harness/source bindings or sampling.
Those existing checks must remain in the eventual integration.

The legacy branch calls the supplied original `configured_quotas`,
`controller.class_quotas`, and `forced_sampling.validate` functions, preserving
their quota and sampler rules. Supply them from the authenticated historical
source registry, not a source selected by an untrusted miner. Current v5 remains
K4/L4, eight retained class-selected responses, an allowed search range0..999,
and max_batches9. `samples_per_batch` is absent from the actual signed epoch171;
it is an optional consistency/config projection in the existing code, not a new
required field. Older unequal quotas and absent forced samplers remain separate
legacy cases. The resolver does not rewrite any historical contract or draws.

The complete-group branch proposes these explicit wire additions/changes:

```json
{
  "group_contract": {
    "version": "complete-canonical-group-v1",
    "group_size": 8,
    "min_success": 1,
    "min_completed_failure": 1,
    "outcome_policy": "completed-eos-math-unresolved-masked-v1"
  },
  "required_client_features": [
    "canonical-group-nonces-v1",
    "complete-canonical-groups-v1",
    "retained-unresolved-attempts-v1",
    "sampler-selected-behavior-logprobs-v1"
  ],
  "max_batches": 9
}
```

`sampling_contract` stays in its existing root location and retains `randomness`,
`verification`, `generation`, `calibration`, and `support_adjudication`. Its new
proposed version is `forced-inverse-cdf-complete-group-v1`; it has **no**
`max_attempts`. The selected calibration validator checks the existing numeric
bounds and shape. The caller still authenticates the calibration report and
binds it to the checkpoint, runtime and harness. These proposed names are not
supported by the current live v5 code. Existing v5 validation intentionally
rejects them; adding this file alone does not make clients compatible.

One value, `group_contract.group_size`, determines the required response count
and exact ordered nonce set0..N-1. Independent minima are not balanced quotas:
N=16 with minima1/1 still requires all16 original attempts. Keep unresolved
attempts in the group; they never meet the completed-failure minimum. Whether
the recorded outcomes are authentic is a separate native grading/audit step.
The batch cap remains separate from N. Initial comparison sizes are4/8/16;
the structural size bound is2..128, not a qualification claim for every size.

New manifests reject K, L, samples_per_batch, duplicate size/nonce aliases and
any sampling attempt limit, even if redundant values happen to agree. All group
and sampler fields are exact. Required features must include every listed
feature; a client must explicitly supply its actual `supported_features` set.
The default empty set fails closed for new contracts. A feature declaration is
not an implementation: the actual CLI, sampler, behavior probability witness,
transport, admission and trainer still need those capabilities before release.

For the eventual source composition:

```python
rules = for_manifest(
    authenticated_payload,
    validators=ValidationHooks(
        configured_quotas, class_quotas, validate_sampling, validate_calibration,
    ),
    supported_features=ACTUALLY_IMPLEMENTED_CLIENT_FEATURES,
)
```

This resolver does not project a complete runtime challenge. The future adapter
must combine the checked group fields with the authenticated epoch, checkpoint,
runtime, sampler/harness and canonical task identity, then bind N and the whole
contract. `research/group_contract.uniform` already binds its complete contract,
including N. In contrast, the original frozen private study's
`research/fixed_group_grpo/core.py:attempt_uniform` hashes its run/checkpoint/
update/runtime/policy stamp, task, attempt and position without an explicit N
field. Its N=8 is fixed by the signed scope. Leave that running source untouched.
A later variable-N study must add explicit N/protocol binding or a prospectively
declared distinct sampling_run_id; do not treat prefixes of the original groups
as a qualified alternate-N experiment.

No new audit barrier is implied. Structural eligible registration and unaudited
training remain separate from authenticated audit evidence. This slice neither
adds rewards nor changes the hourly best-assessment weight writer.

Validation uses a sanitized literal projection of the locally authenticated
epoch171 envelope SHA256
`807c6c00a675ef2b3360013978cae0dbcb5ee99f498ba8495be0ff2081c52f98`.
Fixtures contain no URLs, credentials, capability tokens, real submissions or
model data. Their provenance records source hashes. The compatibility tests
extract only the exact hash-pinned pure legacy validator functions from the
referenced local M composition; they do not import its controller/model runtime.
The archived reference sources must be present to rerun these local comparison
tests. A future portable package will need a reviewed legacy dependency path,
rather than dropping the hash checks or making a permissive fallback.

Run from the repository root:

```sh
.venv/bin/python -B -m unittest research.group_contract_manifest.tests.test_resolver -v
```

No cloud calls, training, live file changes, website edits or original-study
artifact mutations were performed for this candidate.
