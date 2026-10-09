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

For an operator-delegated test miner, replace `--key` with
`--cap-file /absolute/private/epoch-capability.json`. These arguments are
mutually exclusive. The operator decrypts the upload capability locally and
transfers only its epoch-scoped capability file to the GPU miner; the registered
hotkey stays on the operator machine. The bootstrap forwards the explicit file
path without reading or rewriting its contents. The admitted CLI checks the
capability against the signed epoch and performs the actual mining/upload.
This is a delegated public-client test, not a new independently signed network
registration. A fresh epoch requires its own matching capability.

The discovery and matching epoch manifest must both verify under the supplied
known authority. The source descriptor's HTTPS R2 URL, exact size and SHA256 are
bound by that manifest. Redirects, transport downgrades, expired epochs and
mismatched discovery/manifest epochs refuse. Source downloads are bounded to
32 MiB compressed; complete decompression is bounded to 256 MiB. All bytes and
archive membership are checked before any source writes. No private key is read
by the bootstrap; the admitted miner receives the same explicit key path.

The frozen bootstrap reads the download address from `source_bundle.url`.
Operator descriptors that also expose `read_url` must supply the same signed
address in both fields. A `read_url`-only descriptor does not satisfy this
bootstrap contract. The October 3 public-client attempt exposed this mismatch
before downloading anything; its manifest and failure remain preserved. The
prospective correction passed a separately signed R2 discovery/manifest/archive
admission control against the exact frozen bootstrap, with all source bytes
unchanged. This control stops before CLI execution and does not establish a
successful rollout or training epoch.

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

The current completed-answer MATH profile is H200/SM90 with CUDA FP32 eager
inference, TF32 disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Core observed
package pins are torch 2.14.0, transformers 5.14.1 and toploc 0.1.6; package
versions alone do not establish equivalent results on another runtime/GPU.
Follow the exact signed manifest's runtime profile and numerical policy.

The restored contract uses `text-tools-long-v2`, temperature 0.8, top-p 1 and
2048 output tokens. The `forced-inverse-cdf-prefill-miner-bound-v5` contract
selects cached eager generation with prescribed public draws and calibrated
prefill verification, with exact cached replay for ambiguous checks. Each batch
contains four distinct completed successes and four distinct completed failures
for one task. Attempt nonces are 0–999 and bind miner identity, epoch, checkpoint,
task, turn and token position. Missing/unfinished boxed answers are unresolved,
not negative training samples. Selected-token logprobs and TOPLOC are uploaded;
the verifier recomputes full distributions. `small-commitment-pairs-v2` names the
compact batch transport; `direct-r2-v1` on the outer discovery pointer describes
the storage access path and does not replace the batch transport contract.
Historical epochs retain their original source, budgets and proof rules.

Miners may choose a bounded search on particular authorized tasks by adding
`--env-id affine_math --indices 1553 --search-budget 32 --max-batches 1`.
The CLI checks that selected indices are unique members of the signed epoch's
training pool before downloading a checkpoint or creating a model. An index
reserved for evaluation is rejected. The search budget is bounded by the signed contract: old versions allow
1–128 as a CLI bound and may impose a smaller manifest ceiling. The pending
v5 contract permits up to 1,000 attempts per task. Follow the source of the
current OPEN manifest. See [the cutover](FOUR_SAMPLE_MINER_BOUND_CUTOVER.md). Omitting task selectors searches the full
authorized pool. These local preferences do not change the public challenge or
its scoring rules.

Portable reproduction:

```sh
PYTHONPATH=.:tests python -B -m unittest discover -s tests -p test_source_bootstrap.py -v
```

The honest subprocess fixture tests real admitted source execution in a fresh
isolated interpreter, not a model, TOPLOC, native math grading or GPU rollout.
