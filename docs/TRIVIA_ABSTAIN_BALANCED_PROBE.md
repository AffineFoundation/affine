# Versioned TriviaAbstain candidate qualification

The original 32-task snapshot and native grader remain unchanged. The earlier
16-seed search returned only abstentions and remains preserved as an unsuccessful
K1L1 qualification. A new bounded candidate harness uses `I don't know` and
`I do know.`: both tokenize to four tokens. This tests target-model computation
on curated public-input candidates; it is not autonomous answer discovery.
The original grader gives abstention reward 1 and the incorrect candidate 0;
this metric is abstention behavior, not factual knowledge improvement.

The new immutable experiment `original-trivia-abstain-balanced-model-search-v2`
uses temperature 4, top-p 1 and at most 32 seeds for each original mining index.
It never trains or submits chain weights. On checkpoint
`46244fc043b1751fa1bdf53248e80f5db4f0d84e5a42c637ca78e54994ede5f9`,
indices 0 and 1 obtained K1L1 after 4 and 9 attempts respectively. A separate
process reloaded the exact model and verified all four retained traces using
full probabilities, TOPLOC and original native replay. Both processes exited 0.
Root authenticated the signed plan and completion, original snapshot, source
inventory and actual ZIP bytes. Root did not repeat model computation.

Evidence lives privately under `state/trivia-abstain-model-control/1790869414`.
The source archive SHA256 is
`4ff7f2d848bf361d38456a03dea474ad755bfe3b0d9a622db1d8713bef7821df`.
This qualifies the new candidate configuration only. Admission into a live
multi-environment epoch, training and heldout gains are separate future gates.
