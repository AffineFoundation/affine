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
