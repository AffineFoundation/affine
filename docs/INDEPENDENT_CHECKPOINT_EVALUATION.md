# Independent checkpoint evaluation

Prospective configuration `evaluation_mode: independent-checkpoints-v1` removes
heldout GPU evaluation from the mining / audit / training publication barrier.
The synchronous default remains unchanged for existing epochs. This changes
scheduling only; inference verification and trainer receipt admission are unchanged.

The controller queues the original complete manifest, checkpoint identity,
remote cache reference, fixed heldout indices and seeds, harness, experiment
ID and training-step counter before and after training. The after request is
queued only after actual trainer checkpoint and persistent state publication.
The controller compares the committed trainer journal to its exact next state
and next inference checkpoint before opening another epoch. Empty epochs do
not increment optimizer steps or count as completed training updates.

Run `python -m subnet.checkpoint_evaluator --config CONFIG` as a separate service
on the authority host. It dispatches only to the existing independent evaluator
box. It does not start another verifier coordinator server, write the shared
role cache map, or dispatch trainer/miner jobs. A local exclusive process lock
prevents duplicate evaluator services. Deploy from the same immutable source
bundle as the prospective controller; the service template contains explicit
SOURCE/CONFIG placeholders and is not an installation instruction for current E10.

The queue is private local state, with completed authenticated evaluation reports
published to the epoch's public R2 prefix. Existing exact completeness, task hash,
seed, runtime/source and worker completion-time validation is reused. Queue
requests and comparison cohorts never mutate on retry. Observation timeouts keep
the original signed request and job ID. Terminal evaluation failures require
operator recovery rather than silently relaunching with a new seed or request.
Infrastructure failures do not become model losses or fabricated completed scores.

`checkpoint-evaluation-progress.json` and the signed public stream equivalent
label `latest_training_checkpoint` and `latest_evaluated_checkpoint` separately.
A report from an older checkpoint is not relabeled as current performance.
Evaluation errors are published as errors and cannot claim successful evaluation.
This module evaluates the exact cohort configured for the deployment, whether
32 or 200 tasks; changing that configuration requires a prospective experiment,
not rewriting an existing comparison. Independent evaluator backlog is visible.

`controller-timing.json` reports measured phase timestamps, actual training update
count and public optimizer counter. Its duration does not assert chain submission:
`full_hourly_epoch_confirmed` remains false until an independent validator receipt
proves the complete epoch including actual chain weights. Mining/audit time budgets
must account for measured training, 91.4 GB optimizer-state publication and 15.24 GB
inference checkpoint publication. A configured budget alone proves no one-hour SLA.
