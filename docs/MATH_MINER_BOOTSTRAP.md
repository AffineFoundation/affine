# Direct-R2 MATH miner source bootstrap

A clean public checkout lacks the generated original task snapshot. Before
mining, explicitly supply the known operator's Ed25519 public authority, a
signed direct-R2 discovery URL and an existing identity key file. Example:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m subnet.source_bootstrap \
  --authority "$AFFINE_OPERATOR_AUTHORITY" \
  --current-url "$AFFINE_SIGNED_CURRENT_URL" \
  --source-cache /absolute/private/source-cache \
  --key /absolute/private/miner.key \
  --state /absolute/private/miner-state --once
```

The discovery and matching epoch manifest must both verify under the supplied
known authority. The source descriptor's HTTPS R2 URL, exact size and SHA256 are
bound by that manifest. Redirects, transport downgrades, expired epochs and
mismatched discovery/manifest epochs refuse. Source downloads are bounded to
32 MiB compressed; complete decompression is bounded to 256 MiB. All bytes and
archive membership are checked before any source writes. No private key is read
by the bootstrap; the admitted miner receives the same explicit key path.

The accepted archive consists of regular public code/documentation files in
subnet, ops, tests, docs, dashboard, examples, prototype and systemd, named public
root documentation/build files, and the explicit generated
`assets/original-math7496.tasks.json`. Unknown generated assets, private state,
key files, environment files, symlinks, hardlinks, duplicates and path traversal
refuse. This path policy is not an assertion that arbitrary signed code is free
of embedded secrets: the operator must review the source they sign.

Extraction uses a new SHA256-addressed directory and read-only source files.
Every reuse checks exact membership and every file's bytes, rather than trusting
a marker. An existing tampered cache refuses; host checkout files are never
replaced. The fresh interpreter uses `-I -B`, excludes host PYTHONPATH and cwd,
and inserts only the admitted source as its application import root.

The admitted CLI is additionally pinned to that source-bundle SHA256. A later
epoch/checkpoint with the same source can continue; a change in source refuses
before runtime/checkpoint/model admission. Rerun the bootstrap to approve the
new signed epoch source. With `--once`, use an external supervisor to repeat
bootstrap across epochs. Automatic cross-source hot reload is deliberately not
implemented. The signed archive must include the CLI's source pin argument;
older archives without it fail argument parsing rather than silently run.

The source is signed executable code, not sandboxed untrusted code. Cache roots
must be controlled by this operator account; writable hostile parent directories
and a concurrent hostile same-user process are outside the admission scope.
Pinned model/runtime packages still need installing and qualifying on the miner
hardware. This bootstrap does not install dependencies, claim GPU compatibility,
prove inference, submit chain weights or create/read a wallet.

The initial GPU profile requires CUDA compute capability 8.6, BF16, eager
attention and deterministic CUBLAS with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set
before inference. The retained RTX 3090 has that architecture. The pilot's actually qualified
runtime reports torch 2.14.0, transformers 5.14.1 and toploc 0.1.6. These are
observed operator-runtime versions, not a promise that a clean installation or
another GPU will reproduce them. Validate installed runtime admission before
mining; compatibility with other hardware is not established by the bootstrap.

Miners may choose a bounded search on particular authorized tasks by adding
`--env-id affine_math --indices 1553 --search-budget 32 --max-batches 1`.
The CLI checks that selected indices are unique members of the signed epoch's
training pool before downloading a checkpoint or creating a model. An index
reserved for evaluation is rejected. The search budget must be 1–128 attempts
per task; the default remains 50. Omitting task selectors searches the full
authorized pool. These local preferences do not change the public challenge or
its scoring rules.

Portable reproduction:

```sh
PYTHONPATH=.:tests python -B -m unittest discover -s tests -p test_source_bootstrap.py -v
```

The honest subprocess fixture tests real admitted source execution in a fresh
isolated interpreter, not a model, TOPLOC, native math grading or GPU rollout.
