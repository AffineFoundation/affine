# Trainer optimization without a miner protocol change

The miner requirements remain four distinct successes and four distinct failures
per task, up to three task batches per UID, with attempt nonces 0–999. Proofs,
sampling challenges, admission, rewards and duplicate checks do not change.

## What the existing evidence says

The wider, matched 750-task evaluation found 548 correct answers for the base
model, 482 at optimizer step 24, and 496 at step 32. The last comparison is a
6.93 percentage point decline from the base model, with paired bootstrap 95%
interval −10.13 to −3.60 points. Training margins improved and gradients remained
finite. A repeatedly inspected 32-task diagnostic is insufficient to establish
recovery or convergence. These are older checkpoints, not a measurement of the
latest checkpoint.

Each ordinary epoch currently averages pair losses within tasks, averages tasks
within the update, and makes one Adam update. Thus an epoch with 225 tasks and
four disjoint pairs per task has 900 pair contributions in one optimizer update;
the effective optimizer batch is not just eight rollouts.

The current objective resets the preference reference to the incoming checkpoint
each epoch. Before the first update, every pair's loss is log(2), regardless of
its absolute margin. Improved preference margins alone do not establish improved
problem-solving. The learning rate is 1e-5 with persistent FP32 Adam moments.

An earlier small matched experiment scored 85/128 with completed wrong answers
versus 77/128 with capped wrong answers. Termination, length and content changed
together; this is evidence to investigate negative-example weighting, not an
isolated causal result or a demonstrated production improvement.

## Controlled work

Research arms restore the same CP33 checkpoint and Adam33 state and use the same
256 archived tasks and task order. This archived population has one positive and
one negative per task, so improvements require confirmation on fresh eight-rollout
production batches. The existing baseline and positive-only arms are
retained. Additional trainer-only arms test learning rates 1e-6 and 5e-7 and
FP32 accumulation of microbatch gradients. None alters miner sampling or quotas.
Research optimizer moments are discarded and cannot be returned as a resumable
production optimizer.

The [SimPO implementation guidance](https://github.com/princeton-nlp/SimPO#hyperparameter-tuning)
specifically warns that 1e-5 can degrade preference-optimization performance and
suggests trying smaller rates, including 5e-7 for math. This motivates the
learning-rate sweep; it does not establish the best rate for Affine.

The initial screen uses the same 128 held-out tasks, seeds, prompts, native
grader, BF16 runtime and fixed evaluation batch size for every arm, with repeated
native controls. This new evaluation cohort is not numerically paired with the
older FP32 evaluation. A promising arm still requires wider confirmation and
fresh checkpoint-bound training data across multiple epochs; a one-update
comparison cannot establish convergence.

## Gradient accumulation precision

Persistent FP32 Adam state does not make parameter gradients FP32. Production
currently adds many microbatch gradients into BF16 `.grad` tensors. In a scalar
900-contribution control, BF16 addition loses more than 25% of the expected sum;
FP32 accumulation followed by a single BF16 projection preserves it within 0.4%.
This establishes a numerical failure mode, not its measured size on the real
model. The research accumulator preserves task weights, clipping and Adam
arithmetic while isolating that precision difference.

Adoption should be based on matched held-out outcomes, numerical controls and
recorded training lineage. Do not increase miner quotas as a substitute for
repairing the optimizer or training objective.


## Current eight-rollout data diagnostic

Epoch 61's native-selected population contains 224 tasks, 896 positive and 896
negative rollouts. Positive trajectories average 115.5 output tokens. Negative
trajectories average 614.7 tokens; 509 of 896 reach the 1,024-token cap. Thus 56.8%
of the negative population is capped. This is a concrete reason to investigate
trainer weighting of unfinished negatives while retaining the miner protocol.
It is not evidence that every capped trajectory would eventually succeed.

The new matched BF16 screening parent solved 96/128 held-out tasks. Treatment
comparisons are still running. The inference exports for the archived one-update
baseline and positive-only branches change about 6.90% and 6.64% of BF16 weights,
respectively; their relative L2 changes are about 0.000474 and 0.000465. Changing
weights, finite gradients and improving training margins do not prove learning.
