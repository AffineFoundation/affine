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
