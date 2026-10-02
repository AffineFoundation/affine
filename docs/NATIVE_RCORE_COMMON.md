# Original RCore through the common runtime

The `prime-native-rcore-terminal-v4` adapter connects `create_session` to a
separate trusted CPU grader. Its public spec binds the original 64 task hashes,
source, snapshot, resource profile, authority, guard and worker. It exposes the
original messages and text-only actions; private grader paths and caches stay
in an operator-owned descriptor supplied outside the submission.

The common role wrapper authenticates the signed job and manifest, exact source
pins, role audience and disjoint mining/heldout indices before invoking the
existing backend executor. It supports mine, verify, train and evaluate. Use one
job per process: the scoped environment binding is not a concurrent-thread API.

```sh
PYTHONPATH=.:tests .venv/bin/python -B -m unittest test_rcore_common_admission
.venv/bin/python -I -B ops/run_rcore_common_role.py \
  --job "$SIGNED_JOB" --authority "$AUTHORITY" \
  --operator-descriptor "$PRIVATE_DESCRIPTOR" --workspace "$WORKSPACE" \
  --cpu-admission-only
```

Admission-only mode loads no model, allocates no GPU and creates no model
workspace. Removing that flag invokes the backend; it requires an independently
qualified signed job, runtime and checkpoint. The portable tests use ephemeral
signed metadata and mocked terminal DTOs, not model or native-grader evidence.

The operator-native check exercised five fresh guarded workers through the
promoted common adapter. Original task zero and fresh replays returned positive
reward `1.0` and negative reward `0.001272633801339809`. These are CPU native
controls; common GPU proof generation and optimizer consumption remain pending.

## Preparing the private terminal package

The published templates are `ops/run_rcore_terminal_resources.py` and
`ops/rcore_terminal_worker.py`. Install them as `preimport-terminal-worker.py`
beside the signed profile and `package/worker.py`, respectively. The package
also contains approved `source`, `dependencies`, `resources` and private
`operator` fixtures. No private fixtures or signed operator profiles are shipped
in this repository.

The resource authority must sign exact file membership, sizes and SHA-256 hashes,
absolute package location, original environment/snapshot identities, controlled
resource environment and guard hash. The terminal profile uses revision
`original-rcore-trusted-terminal-resources-v1`, role
`trusted-terminal-grader`, CPU-only scope and no model/optimizer/chain execution.
The guard checks these bytes before importing the provider in a fresh `-I -B`
process. A role-local descriptor binds the package and an audience of miner,
verifier, trainer or evaluator; its digest is also bound in the signed job.

Moving a package to another machine requires a new machine-specific signed
profile, descriptor, public binding and environment/source identity, followed by
actual remote byte equality and original native replay. The existing operator
profile is not portable. The provider namespace is controlled; full transitive
dependency closure and isolation from a hostile host or Python reflection are
not established. Terminal errors refuse verification rather than becoming
negative training samples.

Remaining admission gates are fresh model generation, full probabilities and
TOPLOC verification, disjoint heldout evaluation, and actual common optimizer
updates. This integration does not change deployed controllers or coverage flags.
