`ops/probe_trivia_abstain_model_search.py` is a narrowly scoped operator probe for
the original TriviaAbstain task snapshot. It allows only mining indices 0 and 1,
at most 16 seeds each, and the exact public candidates `I don't know` and
`definitely_wrong_answer`. The model chooses between those candidates under the
pinned harness; this is curated candidate sampling, not autonomous trivia QA.

An operator-signed plan pins the model files, full declared worker source,
probe bytes, task snapshot and numerical profile. A new private source namespace
runs generation, then a separate process reloads the model to verify all complete
traces using full probability rows, TOPLOC and original environment replay.
Capacity waits end before model loading; no automatic retry or training occurs.
An index needs both positive and negative traces to qualify as a K1/L1 pair.

The original grader rewards abstention, so reward 1 does not mean a correct
factual answer. Native controls already establish abstention reward 1 with
correctness 0, and wrong-answer reward 0 with correctness 0. This model search
checks reachability and computation for those original controls. It does not
establish knowledge gains, a common storage epoch, or a training update.

Four admission controls cover the exact contract, held-out/duplicate/boolean
indices, bounded budgets and changed sampling or snapshot bindings. Actual
remote qualification results must be recorded separately once terminal.

The first actual probe completed in
`state/trivia-abstain-model-control/1790867043` at checkpoint
`46244fc043b1751fa1bdf53248e80f5db4f0d84e5a42c637ca78e54994ede5f9`.
All 32 attempts chose the abstention candidate. The two retained positive traces
passed independent model reload, full probability/TOPLOC verification and original
native replay. Root checked signed lineage, source/snapshot hashes and both
actual ZIPs. There were no negative traces and neither index qualified as K1/L1;
no common training family or epoch was added from this result.

Six additional original-native controls on those two mining tasks confirmed
`I don't care` and `I do know` remain wrong, non-abstaining responses with reward
0 and correctness 0. Those checks do not establish model sampling probability.
A separately signed calibration measures exact tokenizer lengths and current
SUM-logprob candidate scores before any new candidate harness is selected.
The original 32 attempts and original grader remain unchanged.

The first current-model calibration completed and its exact report/probe/plan
bytes were authenticated in `calibration-collected` under the same evidence
directory. Under the unchanged SUM-logprob/temperature-4 sampling rule:

| Wrong public candidate | Tokens | Probability on task 0 | Probability on task 1 |
| --- | ---: | ---: | ---: |
| `definitely_wrong_answer` | 6 | 0.00008614 | 0.00005980 |
| `I don't care` | 4 | 0.01168726 | 0.00844563 |
| `I do know` | 3 | 0.14508286 | 0.18857263 |

The abstention candidate has four tokens. These are two-candidate probabilities
against abstention, not measured rollout success rates. They explain why a
small seed budget produced no original wrong-answer sample. Four additional
native controls confirm `I do know.` and `I do know!` remain wrong and
non-abstaining under the unchanged grader. Their equal-length probability
calibration is separate and pending. No new sampler contract or common epoch
is inferred from calibration alone.
