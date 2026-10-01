# Original MRCR public-shell control

The original `affine_mrcr_v1` task puts the full conversation transcript in
`/workspace/context.txt`. The question asks for a particular occurrence of a
response and a public twelve-character prefix. The original grader reads
`/workspace/answer.txt` and applies its prefix-gated SequenceMatcher reward.
This remains a native file/tool task; the full transcript is not shortened into
an LLM prompt.

The new `subnet/native_mrcr_public_policy.py` is a disclosed symbolic retrieval
control. It derives a shell command from the public question. That command reads
only the original public transcript, selects the requested response and writes
the original answer-file path. It never reads task answers, task snapshots or
private grader fields. It prints a bounded acknowledgement, while the original
grader reads the complete answer file. This is curated target-model computation,
not evidence that an LLM independently solved the task or sampled the command.
Model probabilities and TOPLOC verification remain separate integration gates.

A first 32-index inventory passed 64 original sandbox/grader controls, but a
stronger check found that its mining and held-out questions reused the same
underlying conversation. Those results remain conformance controls and are not
an admissible disjoint learning evaluation.

The replacement inventory selects sixteen original questions from each of two
cached, original default data buckets:

| Partition | Bucket | Local indices |
| --- | --- | --- |
| Mining | `4n-32k-64k` | 0–15 |
| Held-out | `8n-32k-64k` | 16–31 |

After removing only the current follow-up question, the conversation-body hashes
have no overlap between partitions. Each partition contains one underlying
conversation with sixteen questions, so it is still a small and correlated
pilot. The benchmark bucket `8n-64k-128k` is excluded.

All sixty-four fresh original Docker/tool/grader controls passed on the
replacement inventory: public retrieval receives reward 1 and writing a wrong
answer file receives reward 0. Four parser/shell controls also pass. None of
these checks performed model inference, generated TOPLOC artifacts, completed a
common epoch or trained a model.

The replacement snapshot has 32 tasks and 10,252,240 bytes, SHA256
`359aabee797b75e1427b5e1a90edc6fbd13d1b4d2e79266fd53fbd0ccc495429`.
The source hash is
`14cfed33c566bcb7b08c023df9ce5760f8c95321099973933689167865c7acac`.
Private evidence is under `state/mrcr-tasksets/root-disjoint-buckets32`, including
materialization, conversation-split and `root-native-shell-controls.json` reports.
The earlier inventory remains under `root-original32` with its overlap finding.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_native_mrcr_public_policy.py'
```

Prospective integration must bind this policy source, original task definitions,
container/runtime and complete tool trajectory; verify full model probabilities
and fingerprints; and retain the separate disjoint evaluation group. Native
conformance alone must not be presented as learned performance.

A prospective common-source preparation helper is now available as
`ops/prepare_mrcr_common_source.py`. It accepts an operator-owned, approved source
package with the qualified harness and GPU-runtime hashes and creates a new
package. It rejects changed pins, existing destinations, symlinks, model-file
extensions, wallet/state directories and oversized inventories. Source files are
recorded by hash. It does not copy the package's top-level state, models or
configuration, publish an archive, migrate a controller or run a model. It is
not a sanitizer for untrusted miner uploads.

The new `public-mrcr-shell-candidates` policy constructs two comparable shell
commands using the public question: one uses its exact prefix, and the other
changes a single prefix character. Both retrieve the same public response and
write the complete answer file. This avoids comparing a long correct command
against a much shorter wrong-file command under summed sequence probabilities.
A separate second-turn candidate override can finish with `Done` or `Finished`.
The policy revision and source SHA256 are required, and the harness source hash
also binds the public policy file. Existing sampling and rendering branches are
preserved. Five preparation controls and the four public-shell controls pass.

```sh
.venv/bin/python ops/prepare_mrcr_common_source.py \
  --source /path/to/approved-extracted-source \
  --destination /path/to/new-prospective-source \
  --policy-file subnet/native_mrcr_public_policy.py
.venv/bin/python -m unittest discover -s tests -p 'test_prepare_mrcr_common_source.py'
```

Operator configuration must also stage the exact disjoint snapshots and bind
its environment hash to the selected approved adapter. A changed code binding
creates a new evaluation cohort; retain historical reports rather than silently
merging those cohorts. Shared-pipeline model/proof qualification and a completed
native epoch remain required before this can be counted as a trained family.


## Actual remote model/proof qualification

A separately pinned remote search used approved checkpoint
`d8e047f13278692e0baa0df213b4d5566318e582bb74e47754a63a91d6125383`
and the disclosed public-shell candidate policy. Original index 0 found one
positive and one negative in four attempts. Index 1 found a positive but no
negative after eight attempts; it is not a qualifying K1/L1 batch.

A separate process reloaded the exact model and verified all three complete
traces against their full log probabilities and TOPLOC fingerprints, then
replayed their tools through the original native environment. Both immutable
ZIPs and the search/verification reports match the operator-signed completion
record. A root check independently authenticated the plan and completion,
checked exact source-file membership against the approved archive, and hashed
all actual artifact bytes. That root check did not itself rerun model inference.

The index-0 ZIP has 45,343,956 bytes; index 1 has 22,678,196 bytes. Evidence is in
`state/mrcr-model-control/1790863561`, with the root lineage check in
`state/root-audits/mrcr-and-tau2-prerequisite-root-evidence.json`. This qualifies
controlled model computation and original environment replay. It is not an
unrestricted model-solving result, a completed common training epoch, or a
performance gain. The next gate is the shared frozen-upload/audit/scoring and
optimizer/checkpoint loop, at a completed controller boundary.
