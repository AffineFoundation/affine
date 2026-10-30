# Network Spec

Spec for the subnet's incentive and validation loop, as dictated on 2026-10-30.
Continued in [spec-2.md](spec-2.md) and [spec-3.md](spec-3.md).

## 1. Dynamic registration

- Miners register and submit a batch into a queue.
- We set the rate at which batches are taken from the queue.
- Registration price moves to pick the next batch: it is decided by who believes their batch can pass the gate.
- Good batches justify a higher price.

## 2. Baseline

- Once per GREEN, run the king (the current head) on about 7000 held-out tasks to build a baseline score: the likelihood of success per task.
- Spend up to an hour on it.
- Use all 8 B200 GPUs at full capacity, with vLLM tuned for maximum tokens per second.
- Plain local disk only. Never the encrypted disk.
- Past mistake: checkers loaded from the encrypted disk took about 10 s per math check. That was wrong and must not happen again.
- Purpose of the baseline: download the model and sample as much as possible, to get the best estimate in the shortest time.

## 3. Miner submissions

- Miners submit a batch of text rollouts (not necessarily tokens).
- The format must be broad enough for miners to optimize, and interpretable enough for us to audit for overfitting.
- Future: an optional learning rate.
- Current: 9 to 125 tasks, about 8 rollouts each. Relax the specifics over time.
