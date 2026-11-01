# Network Spec (continued)

Continues [spec.md](spec.md) and [spec-2.md](spec-2.md). Section 12 is a planned design, not live.

## 7. Efficiency

- The baseline must be fast, so it gives the best probability approximation and a correct baseline. That reduces sampling variance and forces miners to produce good batches.
- The probe must be fast, so it can sample the most tasks against the baseline in reasonable time, perhaps several times, for the most accurate estimate of whether the miner truly improved the model.
- If we cannot do this fast enough, we cannot get a better estimate. Higher variance makes the incentive mechanism worse and raises the chance that miners submit random or null batches to maximize earnings.

## 8. Auditability

- The training system must be auditable on behalf of miners.
- With the trainer we built, and the breadth miners have over batches, learning rates, sizes and so on, they can always find a batch that will be applied.
- Miners can test offline before submitting to see whether a batch will perform well. That reduces the variance of their own submissions and makes them better miners.

## 9. Research question

- How should we sample in the baseline and the probe, under a time constraint, to get the best estimate of the batch's improvement on the model?
- Which statistical techniques make sampling faster, reduce variance and improve accuracy?

## 10. Expressiveness

- The submission structure must be expressive enough that an intelligent miner can always submit a batch that truly improves the model, while our estimation stays accurate.

## 11. Disk space

- The system runs continuously for days at a time, so anything left behind by an epoch builds up and eventually overloads the machines.
- Every stage must clean up after itself between epochs. Each machine keeps only the current king and the next model queued to be checked, and nothing else grows without bound: no old checkpoints, candidate copies, experiment outputs, backups, temp files or unrotated logs.
- Design every component with its steady-state disk footprint in mind, and check free space before a job starts rather than failing mid-run.

## 12. Planned: era-based baseline (Jacob, Oct 31, 2026)

**Status: planned, not live.** Nothing in this section is implemented yet. The live gate still works as described in sections 2 to 6 and in the miner docs.

- **Era:** the period from one GREEN to the next. The head is constant within an era.
- **Background sampler:** one pod runs the current head full time over the full training task pool. It keeps a running prior table task_id -> pass rate (counts and successes), as accurate as possible.
- When a GREEN ends the era:
  1. Pick ONE new baseline task set from the rough priors, choosing high-variance tasks (p near 0.5, highest p(1-p)).
  2. Spend a flat 30-minute warmup rolling the new head on just that set, starting from the priors, to make the per-task estimates accurate.
  3. Then open miner submissions.
- Every epoch in the era judges candidates on that same fixed set against that same baseline.
- No per-epoch subsets: eras are short, and per-epoch draws add submission-level variance we do not want. No per-epoch baseline rollouts are needed.
- The warmup is a fixed 30 minutes, not adaptive. We measure the baseline error over time and raise the warmup by hand if needed.

### Motivation

- It scales to expensive tasks (coding, agents, 100B models): per-epoch cost is only candidate rollouts.
- For big models, prefer more distinct tasks with fewer rollouts each.
- The baseline is exact for the whole era, since the head does not change.
- Env rotation (MATH, then coding, then agent) fits on top: the sampler focuses on the active env.

### Open notes

- Include the baseline error in the gate SE (z).
- Grade the head and the candidate identically.
- Keep the task set and the priors private. Only a GREEN reveals, per the existing reveal policy.
- Each GREEN costs a pause of about 30 minutes with no gate decisions.
