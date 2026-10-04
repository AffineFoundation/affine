# Full-coverage training candidate

The current fixed-reference v2 trainer computes references for every verified
pair, but three optimizer updates select only the first three pairs. A report
of 175 input pairs therefore does not mean 175 pairs contributed gradients.
The latest completed fixed 32-task evaluation fell from 21 to 20 correct; the
separate 128-task comparison also did not establish improvement.

`subnet.covered_epoch_optimizer` provides the prospective
`bf16-full-adamw-covered-fixed-reference-v3` policy. It binds pair ordering to
full pair content and a post-freeze challenge seed. Three optimizer updates
partition 175 pairs into groups of 59, 58 and 58. Each group accumulates the
mean preference-loss gradient one pair at a time, then clips and updates once.
All references are computed from the immutable input checkpoint before updates.
For fewer pairs than updates, the schedule repeats pairs without empty groups.

The optimizer remains full-model BF16 AdamW at the existing learning rate.
This candidate addresses data coverage, not a proven numerical-precision issue.
Every update records its actual pair identities, group mean loss, cumulative
coverage, full-parameter gradient coverage and persistent optimizer counters.
It exports one final checkpoint after complete coverage, reducing intermediate
checkpoint disk writes. It does more backward computation than training on
only three pairs; faster wall-clock training is not claimed.

Seven CPU controls include real gradient accumulation compared with a dense
mean, actual updates to pairs beyond the first three, a 175-pair schedule,
input-order independence and invalid/duplicate/nonfinite refusals. The isolated
GPU admission has three additional controls, including the fact that a device
property query itself initializes CUDA. Check process freshness before that
query, then bind the actual device capability.

The isolated
GPU probe `ops/probe_covered_epoch_optimizer.py` additionally checks five curated
native success/failure pairs on approved mining tasks, full input-model proofs,
three accumulated updates, changed parameter values, a fresh successor model
reload and proof replay, and rejection of a tampered successor proof.

The H100 control completed successfully on the 7,615,616,512-parameter model.
All five pairs contributed in groups of two, two and one; all 339 parameter
tensors received gradients and the persistent optimizer counters reached three.
The fresh successor passed independent model reload and full native/proof replay,
and a tampered proof was rejected. Root separately read every byte of the input
and successor checkpoints, all 1,875 source files and the five original frozen
artifacts, reconstructed fixed references from their full probabilities, and
checked the original successful wait, process absence and idle GPU. Peak GPU
allocation was 78,644,184,064 bytes; larger trajectories and H200 production
workloads still require qualification. Aggregate evidence is in
`docs/data/covered-training-control-20261004.json`.

The GPU control is a numerical/proof qualification, not evidence of genuine
mining, a completed public epoch or held-out learning. The prospective controller
now accepts `training_policy: bf16-full-adamw-covered-fixed-reference-v3` explicitly;
the default remains v2. It constructs a separate signed training context from
the original post-freeze challenge, binding the epoch, input checkpoint, seed
and receipt-set hash. Workers require exact policy agreement and source pins
before artifact reads. Unselected batches remain ineligible for training.
Exact copies of fully audited pairs are collapsed before accumulation so clones
cannot multiply their gradient contribution. Original-request recovery compares
the same signed context and policy, without reapplying a new-job disk reserve.
Unexpected remote context changes are refused before successor publication.

The deployed workers still use their original source and policy. The integrated
signed worker path, production workloads, source admission and a completed-boundary
handoff still require qualification. Older signed
epochs, their training policies and their deadlines remain unchanged.
