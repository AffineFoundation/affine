# Network Spec (continued)

Continues [spec.md](spec.md). Continued in [spec-3.md](spec-3.md).

## 4. Queue

- Take the next batch out of the queue.
- Miners do not need to specify that the batch relates to the current model, or when it was submitted.
- After the batch is applied in the trainer, check the performance increase with the probe.

## 5. Jargon

- **Baseline:** the king's per-task success estimate on the held-out tasks, built once per GREEN.
- **Trainer:** applies the miner's batch to the king to produce a candidate model.
- **Probe:** runs after the training step. It gives the best estimate, in the shortest time, of whether the model trained on the miner's batch objectively improved performance on the samples.

## 6. Green vs red

- **Green:** the candidate beats the baseline with very little variance. Promote it as the new king and produce another baseline.
- **Red:** throw out the step. The king stays the same. Go back to the trainer step, which should happen incredibly fast, then run just a probe for each step.
