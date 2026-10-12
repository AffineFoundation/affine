# Original suspicious-rollout replay: finalized epoch 153

All nine investigated batches passed the original verifier: 72 rollouts and
35,886 generated token positions. This result does not support a token-forgery
penalty for those submissions. It does not establish honest historical execution,
sound reasoning, or useful training data.

The batches were selected because twelve responses reached the 2,048-token
budget after a complete boxed answer, with unusually long or garbled suffixes.
Two were graded correct and ten incorrect. The historical rule permits a binary
grade after a complete boxed answer; an unfinished response without a complete
answer is a different case. This targeted sample cannot estimate the overall
network's invalid-submission rate.

The replay used the original checkpoint, archived verifier and native grader,
miner-bound randomness, probability artifacts, TOPLOC and numerical calibration.
No submitted tokens or acceptance thresholds were changed. Each complete batch
was checked once; there were no numerical retries. The verifier's existing cached
reference adjudication ran for all twelve capped responses and reproduced their
entire token paths. The other sixty rollouts passed its prefill-based checks.
Two short honest controls also passed on the research H200. This bounded check
does not qualify arbitrary hardware configurations.

The nine batch checks took about 1,000 seconds in total. Because the selection
deliberately includes costly cached fallbacks, this is not a representative
network throughput measurement.

## What this changes about the investigation

Sampling verification and mathematical quality remain separate. A model can
produce invalid reasoning or a poor suffix while following the prescribed
sampler. These artifacts therefore remain relevant to training-data research,
without being evidence of a demonstrated forgery.

Under the actual epoch objective, the twelve post-answer suffixes account for
2.515% of raw training tokens but only 0.249% of total absolute derivative mass
with respect to the selected token log-probabilities. The two correct-response
suffixes account for 0.046%. These are objective coefficients, not measured
parameter-gradient norms or a causal estimate of held-out degradation.

The [task-coverage trial](TASK_COVERAGE_RESULTS_20261012.md) and controlled
learning-method comparisons continue. A later fixed-data comparison could test
masking post-answer suffixes in the loss, with original trajectories and grading
preserved. No live contract, rewards or historical grades changed in this study.
No model-quality improvement is claimed.

## Evidence

The original terminal record, batch artifacts, per-rollout outcomes and execution
sources were preserved. An independent full download of the private archive
matched its local contents before the research checkpoint cache was removed.

| Record | SHA256 |
| --- | --- |
| Original checkpoint | `3fbab8de4476ba623f0090d7cc96211b46635ac7123defd6a65050a91a569311` |
| Original verifier source | `f5287f57bdae344e5a3d073309e5c830073f603838b4b52a189f5ec0a269cb24` |
| Signed execution scope | `e24893a6eea325de50ef89a420a5eeabf165eedc957e9a9053e6e7ee44fd200b` |
| Original result | `8d7f96d115787d4aa9665d57910e8882a90f4e93f4dd741079ae2d249a41ea48` |
| Original capture | `45cf28b96c858b85b5be3eac64baf0e156ffcb77ee79d857235f95973cfc13bf` |
| All 72 outcome rows | `8bf6bde3d12a28e5b08994f69dd479986860fbaa0254badb76dbc0a206947eeb` |
| Independent signed capture review | `69a68c4889f5e65a59ed61adf820727a39935778642364305ede93175ad22aca` |
