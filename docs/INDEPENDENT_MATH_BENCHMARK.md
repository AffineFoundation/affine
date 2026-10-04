# Independent paired math evaluation

The first complete comparison finished on October 4, 2026. Root independently
checked all eight original signed jobs and their bound reports, the signed cohort
and final result, the actual successful supervisor wait, and complete paired
indices, seeds, task hashes and model/native verification results. The base
checkpoint solved **88/128 (68.75%)**; the nine-update checkpoint solved
**89/128 (69.53%)**. There were six paired gains and five losses; the exact
two-sided McNemar p-value is **1.0**. This does **not** demonstrate improvement.
The 95% Wilson accuracy intervals are 60.27–76.13% and 61.08–76.84%, respectively.
Full aggregate results are in [the result record](data/math128-20261004.json).
Reserved task identities remain private so miners cannot target the cohort.

This finding supersedes the earlier pending benchmark status. It does not
measure the subsequent epoch-six checkpoint or establish a trend. The fixed
32-task diagnostic's earlier gain is not a substitute for this larger result.
Continue measuring later checkpoints under precommitted matched settings; treat
this cohort as a monitoring set after publishing its aggregate outcome, and
reserve a fresh untouched set for a later independent confirmation.

The fixed 32-task chart is a diagnostic, not sufficient evidence of sustained
learning. A separate comparison uses 128 tasks selected before any outcomes,
from the 750 reserved MATH tasks, excluding both mining tasks and the diagnostic
32. It pins the zero-update baseline, nine-update learned checkpoint, original
task asset, source, autoregressive harness and matched seeds in a signed plan.
Four 32-task jobs are planned for each checkpoint on a separately retained
Hopper GPU, after actual allocation and full control admission.

Admission checks the complete installed source, package versions and checkpoint
bytes. Before benchmarking, a separate control exercises original grading,
tokenization, an honest generated rollout with independent model reload, and
token/proof/logprob/outcome/weight-hash rejection. Controls use an old diagnostic
task, keeping the new comparison cohort untouched.

The operator checks original signed jobs and reports, their time budgets,
checkpoint/source/runtime/profile bindings, exact indices and seeds, and every
native/model verification result. Any missing task or grader error stops the
comparison; it is not silently removed or counted as a model failure. Original
handles and failed attempts remain preserved. Observation timeouts do not
authorize duplicate GPU jobs. Benchmark jobs have no live reward contract and
cannot generate payouts or chain transactions.

`subnet.paired_evaluation.summarize` reduces these authenticated outcome rows.
It checks complete paired coverage, task hashes, seeds, binary native rewards
and verification flags, then reports accuracy intervals, paired gains/losses
and an exact two-sided McNemar probability. This arithmetic function does not
authenticate execution itself. Its caller must first check the original jobs
and admitted runtime. A favorable comparison at one checkpoint is not proof
of stable long-term improvement. Pretraining exposure to public MATH problems
also remains unproven.

Current experiment records live in the private operator state under
`independent-math128-precommit-v1`. Public status must distinguish admission,
running benchmark jobs and fully checked final results.

The first extra H200 hydrated the pinned checkpoints and passed CPU controls but
failed to create a CUDA context in both the approved and provider-default Torch
runtimes. No benchmark shard ran on it. A separately retained single H100 passes
actual CUDA allocation with the approved runtime and has independently checked
approved source bytes. It passed full checkpoint, original grader and honest/
tampered rollout admission. The actual-wait parent checked original process exit
and an idle GPU before starting the first baseline shard. A preceding attempt
failed its interpreter allocator-zero cleanup assertion; the successful attempt
records that residual allocation and checks actual process/GPU quiescence after
exit. Original failures remain retained. This changes the dedicated machine and
cleanup observation, not the precommitted cohort, model weights, sampling settings
or verification tolerances; it does not qualify H100 for public mining.
