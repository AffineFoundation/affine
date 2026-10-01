# Original Trivia Abstain qualification

The next prospective family uses the original `affine_trivia_abstain_v1` wrapper
and pinned TriviaQA provider. It does not replace questions, infer answers or
change the original grader. Thirty-two original tasks were materialized from the
existing dataset cache with network fetching disabled. Mining indices are 0–15;
held-out indices are 16–31. The 36,983-byte snapshot has SHA256
`c08f2397ce161235e6ccefd78cd2294d2249a5e92c1640344e64b25fc528819e`.
Its environment source hash is
`fe5322f7a28c5732dd0f6d2eca500287ac57832e42b36c964e48165370f34956`.

Fresh original grader execution tested two disclosed controls on every task:

| Control | Original reward | Correct | Abstained | Omniscience |
| --- | ---: | ---: | ---: | ---: |
| `I don't know` | 1 | 0 | 1 | 0 |
| Deliberately wrong nonanswer | 0 | 0 | 0 | -1 |

All sixty-four native grader calls passed. The abstention control is positive
under the original `solved` rule, which accepts correct answers **or** abstention.
It demonstrates legitimate outcome grading, not factual knowledge. Training or
evaluation must preserve these secondary metrics so an increasing success rate
caused by abstention cannot be described as increasing correctness.

This qualification has not generated model probabilities or TOPLOC artifacts,
run a common storage epoch, trained a checkpoint, or submitted chain weights.
Those remain separate prospective integration gates. Task answers and full
snapshots remain operator-private. Evidence is stored under
`state/trivia-abstain-tasksets/root-original32`, including the materialization
report and `root-native-grader-controls.json`.

Reproduce materialization with sufficient local cache storage:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python \
  -m ops.materialize_tasksets --sources affine_trivia_abstain \
  --output state/trivia-abstain-tasksets/reproduction \
  --count 32 --training-count 16 --timeout 300
```

This bounded command fails if the pinned provider is not available offline; it
does not silently switch datasets or fetch a replacement.

A prospective common-pipeline configuration now adds this original taskset to
the thirteen-family MRCR source without changing source code, existing
environment definitions, held-out harnesses or prior training groups. It uses
sixteen mining questions and sixteen disjoint held-out questions. The shared
harness source binding remains unchanged. The new held-out policy is
unrestricted autoregressive sampling with a 256-token budget; the mining policy
is a disclosed abstention-versus-wrong candidate control.

All sixty-four fresh native controls also passed under the staged approved
adapter: abstention gives solved 1, correct 0, abstained 1 and omniscience 0;
the wrong control gives solved 0, correct 0, abstained 0 and omniscience -1.
Full public question/context inputs do not overlap between the prospective
fourteen-family mining and held-out sets. Question-only checks must include
public context files for tasks such as Oolong, where identical questions can
refer to different original contexts.

The prospective adapter source binding is
`5dd2d6a402e66d0b92dcc10f2178669030bd30c0d160a7d1b6b08d65da621956`.
The original snapshot remains unchanged. No remote model/proof qualification,
shared training epoch or learned-performance gain is established by these
preparation and grader checks. Existing historical evaluations are retained,
and reward that includes abstention must not be described as factual accuracy.
