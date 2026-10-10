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
