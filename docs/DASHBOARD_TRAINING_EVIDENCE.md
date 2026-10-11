# Finalized training counters

The dashboard marks a modern epoch finalized from its authenticated learner completion, independently of legacy reward-score files. Loss, pair count, checkpoint and objective are projected from the original signed training receipt bound to that completion. Missing or invalid evidence stays unavailable. Open epochs do not expose training receipts or rollouts through this projection.

For persistent updates, `training.losses` contains each recorded update loss, `training.input_pairs` is the reported pair count, and `training.objective` identifies the declared training policy. These are training diagnostics, not held-out scores.

The current one-step DPO update records loss before that update, using the epoch-input model as its reference. The initial preference margin is therefore zero and a loss near `log(2)` is expected. A flat initial loss does not establish a stalled optimizer, and a lower training loss does not establish generalization. Held-out answer accuracy, measured under a matched evaluation protocol, remains the efficacy metric.

Operators can supply `--training-receipts /path/to/read-only-cache` to the dashboard server. The cache contains original ROOT-signed finalized receipts; the projection verifies signatures and epoch/checkpoint binding and emits only allowlisted summary fields. Updating the dashboard's authenticated verifier-source admission is also necessary when the network source changes. This changes public data projection, not training, mining, proof rules or reward weights.
