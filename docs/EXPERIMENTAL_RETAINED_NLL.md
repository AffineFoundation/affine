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

The initial isolated experiment used the original 7B upstream checkpoint, fresh
Adam, 2,046 completed correct miner responses from 512 finalized training tasks,
eight 64-task updates and learning rate `1e-6`. Development accuracy rose from
94/128 to 98/128: four gains and zero losses. The conservative paired 95% effect
interval was -2.67 to +8.56 percentage points, so this is encouraging but does
not establish independent improvement. A separate official TRAIN-answer control
stayed at 94/128. Its response count, style and length differ, so it does not
isolate rationale quality. Independent 512-problem confirmation is in progress.

This experiment also changed optimizer initialization and update dose relative
to production; it does not prove that negative gradients alone caused earlier
regression. Final-answer correctness does not establish correct reasoning, and
quota-selected K4/L4 responses are not an unbiased eight-draw policy sample.
The approach is related to [ReST-EM](https://arxiv.org/abs/2312.06585) and
[positive-only RAFT](https://arxiv.org/abs/2504.11343).

Portable candidate checks:

```sh
python -m unittest discover -s tests -p 'test_positive_only*.py' -v
python -m unittest discover -s tests -p 'test_positive_sft*.py' -v
```

These are CPU contracts using throwaway signing keys and explicitly identified
fixtures. A separate tiny H200 test has exercised real fresh-state training,
local state export/restore and continuation. Full-size candidate qualification,
independent confirmation and sustained learning are still pending. Neither CPU
fixtures nor arithmetic-consistent reports establish those outcomes.
