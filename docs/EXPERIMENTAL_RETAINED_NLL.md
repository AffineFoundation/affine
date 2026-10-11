# Experimental retained-Adam positive NLL objective

This objective requires a qualified signed operator release. The live release
was activated on October 10, 2026 for epoch 117, covering successful optimizer
counters 26–42. Its first two updates completed and published checkpoints 27
and 28. See
[the live training guide](CONSERVATIVE_FP32_TRAINING.md) for the activation,
measured results and fixed evaluation plan. Learning improvement remains unproven.

## Objective and state

For each positive/negative pair, let `p` and `n` be the mean output-token log
probabilities, and let `r` be the pair margin measured on the immutable BF16 epoch
input. The optional objective is

```text
softplus(-0.1 * ((p - n) - r)) + coefficient * (-p)
```

The coefficient is exactly zero or one. Each task receives equal weight within
an update; pairs within a task share that weight. The positive NLL uses the full
original output, without a tail mask or extra model forward passes. The default
and explicit zero retain the historical preference graph and report shape.

The released horizon permits one update per job at learning rate `5e-7`, with
coefficient one for exactly 16 retained optimizer updates. Its
`first_optimizer_step` is the counter **before** the first covered update. The
coefficient becomes zero when that counter reaches `first_optimizer_step + 16`.
Retransmitting or failing a job does not advance the counter or consume another
update. The FP32 master weights, Adam moments, hyperparameters other than the
authorized learning rate, and optimizer lineage continue without a reset.

## Qualification and activation

An authenticated release must bind the exact execution source and runtime,
qualification evidence for both zero and unit objectives, and a nonzero retained
Adam parent. The private mechanical qualification scope explicitly makes no
claim that its optimizer moments have the same distribution as live training.
Its CPU fixtures use synthetic signing identities and are not GPU evidence.

The prospective live horizon separately binds its initial checkpoint and parent
descriptor, genesis, first eligible round, and counter. Existing manifests and
jobs remain immutable. Reports must bind the original job and source, the
configured coefficient and horizon, and the task-weighted loss components.
Arithmetic consistency is bookkeeping evidence, not independent proof of
physical execution or learning.

Deployment also requires an operator policy that preserves historical report
admission and selects cache-ACK code from each original job's source. An old
in-flight or prepared job must retain its original route. Model publication and
ACK promotion remain separate from local optimizer retention; this objective
does not authorize optimizer uploads, production resets, or changes to miner
sampling, native grading, rewards, or the website.

## Portable CPU checks

From the repository root, with its Python dependencies installed:

```sh
python -m unittest discover -s tests -p 'test_retained_nll_*.py' -v
```

The controls cover closed-form loss derivatives, equal task weighting, default
zero behavior, shared component passes, continuation of real tiny Adam state,
signed release bounds, the exact 16-update horizon, retry idempotence, and report
provenance. They need no private source trees, credentials, network, or GPU.

## Positive-only SFT research

The separate `task-balanced-positive-sft-v1` candidate is **not active in the
live network**. It implements the proposal to retain the miner's positive and
negative sample quotas while learning only from correct responses. It requires
a new, explicitly qualified v6 execution release. Installing this source or
changing an unsigned configuration does not activate it.

The trainer first checks the complete admitted task groups and their content
identities. It selects positive trajectories with an actual terminal EOS and
no interior EOS. A cap-ended positive may remain valid under the original
mining contract but is excluded from this experimental objective. Exclusions
are recorded; no EOS is appended, and no historical miner penalty is inferred.
Negative trajectories receive no forward pass or gradient in this objective.
Native grading and sampling audits retain their separate roles.

For each response, the loss is mean generated-token cross-entropy, including
the actual EOS and excluding prompt tokens. It is then averaged over retained
positive responses within a task, and over tasks in the update group. Each task
is used once across the fixed groups. The implementation retains FP32 gradient
accumulation, FP32 Adam master weights and moments, and BF16 model export.
The release binds the tokenizer EOS, objective, source, runtime, learning rate,
input inventory and optimizer lineage. Its reports record positive selection,
losses and actual update coverage instead of inventing preference margins.
GPU capacity admission releases unused allocator cache after the diagnostic
forward passes, then checks the original memory reserve. It retains live model
tensors and rejects genuinely insufficient memory.

The initial isolated experiment used the original 7B upstream checkpoint, fresh
Adam, 2,046 completed correct miner responses from 512 finalized training tasks,
eight 64-task updates and learning rate `1e-6`. Development accuracy rose from
94/128 to 98/128: four gains and zero losses. The conservative paired 95% effect
interval was -2.67 to +8.56 percentage points, so this is encouraging but does
not establish independent improvement. A separate official TRAIN-answer control
stayed at 94/128. Its response count, style and length differ, so it does not
isolate rationale quality.

The original independent 512-problem comparison is now complete: upstream
404/512 versus SFT 405/512, with 10 gains, nine losses, and no infrastructure
errors. The paired 95% effect interval is -2.72 to +3.10 percentage points;
one-sided exact p = 0.5. This does **not** confirm improvement. Both original
executions were captured, and prompt-token and decoding parity passed. This
candidate is not being promoted based on the development result.

Those inputs came from finalized epoch 133 under a later network checkpoint;
replaying them against the original upstream weights is historical supervised
learning. It is not a claim that the upstream model sampled those responses.
Fresh mining against the SFT checkpoint and sustained online continuation are
separate next experiments. The original epoch and checkpoint bindings remain
part of each input's provenance.

This experiment also changed optimizer initialization and update dose relative
to production; it does not prove that negative gradients alone caused earlier
regression. Final-answer correctness does not establish correct reasoning, and
quota-selected K4/L4 responses are not an unbiased eight-draw policy sample.
The approach is related to [ReST-EM](https://arxiv.org/abs/2312.06585v4) and
[positive-only RAFT](https://arxiv.org/abs/2504.11343v2), with two distinctions
that matter for sustained learning. ReST-EM's experiments fine-tune the original
base model in each iteration using newly generated solutions; continuing from
retained optimizer state is a different recipe. The RAFT study reports fast
early gains but declining policy entropy and eventual loss of exploration.
Requiring miners to collect negatives does not itself preserve policy entropy
when those negatives contribute no training gradient.

Next comparisons should measure held-out accuracy, response diversity, task
coverage and stopping behavior alongside training loss. Base-anchored replay
and bounded regularization are possible follow-ups if continued SFT drifts or
plateaus. A group-relative or leave-one-out policy-gradient comparison needs
unselected sampled groups and behavior probabilities; the current forced K4/L4
submissions cannot simply be treated as that data. These are research options,
not changes to the active mining contract.

Portable candidate checks:

```sh
python -m unittest discover -s tests -p 'test_positive_only*.py' -v
python -m unittest discover -s tests -p 'test_positive_sft*.py' -v
python -m unittest discover -s tests -p 'test_fp32_gradient_accumulation.py' -v
```

These are CPU contracts using throwaway signing keys and explicitly identified
fixtures. A separate tiny H200 test has exercised real fresh-state training,
local state export/restore and continuation. A subsequent full 7B replication
completed the original eight-update recipe and reproduced all ten exported model
files and every BF16 parameter exactly, with checkpoint
`80870311989c83b4f12c97e8a59b596c11c4a54198b1c9263da7f665963b3e56`.
It retained a newly computed Adam state at step eight on the research machine;
this is not recovery of optimizer state from the earlier experiment. Optimizer
state was not uploaded. Training positive NLL again moved from 0.27255 to
0.24771, which measures fit to the training responses, not held-out improvement.

This replication qualifies the executed numerical recipe, not the live API or
full deployment. The completed independent comparison scored 405/512 versus
404/512 for the base model, leaving held-out improvement unestablished.
Sustained online continuation remains untested. Neither CPU fixtures nor
arithmetic-consistent reports establish those outcomes.

### Matched 1,024-task follow-up

A separate, prospective study started on October 11 after the eight-update
candidate failed to establish independent improvement. Both arms start from
the original upstream 7B model with fresh Adam, learning rate `1e-6`, and 32
updates: two identical passes through sixteen groups of 64 tasks. The network
arm uses all 9,994 distinct, EOS-complete, native-correct positive responses
from a frozen cohort of finalized submissions. The reference arm uses one
official TRAIN solution for each of the same 1,024 problems. Prompts, task
order, optimizer settings and update count match. Task weighting remains equal.

The network and reference arms contain 2,524,097 and 267,549 output tokens per
pass respectively. Consequently, the comparison tests the two data recipes;
it does not isolate reasoning quality from response count, wording or exposure.
The full selected input cohort passed native grading. This checks answer
verdicts separately from sampling provenance and does not establish that every
intermediate reasoning step is correct. Negative miner quotas are unchanged.

Both fixed endpoints completed the same new 512-problem evaluation, disjoint
from the training cohort and previous evaluations. Base solved 405, network
positives 268, and official references 389. The primary network-versus-base
comparison regressed by 26.76 percentage points (paired 95% interval −33.33 to
−19.77). All original outputs were retained; no infrastructure errors occurred.
Neither endpoint was promoted. The [aggregate result, uncertainty and stopping
analysis](POSITIVE_SFT32_RESULTS.md) distinguish strong training fit from this
failed transfer result. Production training and the mining contract are unchanged.

### Prospective variable-coverage dispatch

The operator integration in `ops/positive_sft_variable_dose_lifecycle.py` selects
an explicitly released update count from the current epoch's admitted inputs.
Its present capacity is at most 512 tasks and eight updates, targeting at most
64 tasks per update. Smaller populations form balanced groups without repeating
tasks or borrowing old submissions: 65 tasks produce two updates on 33 and 32
tasks. This is a variable dose, not an assertion that every epoch reproduces the
fixed 512-task research experiment.

The integration checks the complete native-admission record and the original
checkpoint, source, optimizer parent and input inventory before issuing work.
Each possible update count has a separately signed release and a job-bound
learning-rate authorization. With no completed positive responses, it publishes
and reads back an explicit no-update receipt, preserves the existing checkpoint
and optimizer counter, and issues no training job. Infrastructure failures do
not become empty data. An already issued job, or a fully committed no-update
receipt, can finish its original recovery path after release expiry; a newly
prepared job still requires current admission.

A tiny installed H200 exercise passed the 65-task/two-update path, local Adam
export and exact restoration, followed by an idempotent no-update case. It used
259 completed positive responses and performed no negative-response forwards.
The native grading and transport interfaces in that exercise were synthetic
fixtures; it does not qualify production API/R2 admission. A subsequent
admission-only expiry-recovery change has CPU controls, without a claim of a new
GPU execution. Existing native grading, sampling audits and the hourly weight
writer remain separate from this trainer dispatch.


## Prospective optimizer-only threading

An optional signed SFT execution policy, `optimizer-only-cpu-threads-v1`, uses
16 CPU threads only during `PersistentCPUAdamW.step`. Forward passes, backward
passes, gradient accumulation and clipping retain the admitted two-thread
profile, and interop threads remain two. The worker restores the steady profile
even after an optimizer error. Reports record the actual before/during/after
settings; an old release cannot silently acquire the override. The exact new
source and policy require their own GPU integration qualification.

The isolated full-7B Adam8 comparison reused identical frozen gradients and
found all 339 parameter tensors, FP32 masters, both moments, BF16 projections and
step counters exactly equal. The optimizer step took 172.13 seconds with two
threads and 54.18 seconds with 16. This measures the optimizer step only, not a
3.18x whole-epoch speedup. It does not activate a production change.

The optional research-parent adoption helper preserves original research
receipts and local optimizer shards. A new signed publication binds one exact
successor request; an ordinary successful successor and ACK then enter the
normal local-state lifecycle. It never fabricates a historical production job,
resets Adam, uploads optimizer state, or grants space by deleting research
artifacts. This path remains prospective; the independent SFT result above did
not meet the criterion for production adoption.

The helper distinguishes the original eight-update replica from the matched
1,024-task, 32-update network study. The latter requires its original signed
scope and completed numerical receipt, full checkpoint inventory, and the
locally retained Adam32 descriptor. A reference-control endpoint without local
optimizer state cannot be adopted through this path. Neither format establishes
held-out improvement or authorizes deployment by itself: the operator must
separately qualify and release the successor. The original Adam8 checks and
ordinary first-job ownership, local restore and ACK rules remain in place.

The adoption tests exercise a real tiny Adam32 restore and step to 33, plus
rejection of changed counters, study bindings, source receipts, checkpoint
files, missing retained state and a second successor job. These are CPU state
and protocol tests, not a claim that the new full-model production handover has
run. The currently running research jobs use their original frozen source.
