# Network Spec (continued)

Continues [spec.md](spec.md) and [spec-2.md](spec-2.md).

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
