# Conservative FP32 training

The 2026-10-09 fresh run starts from the original Qwen2.5-Math-7B-Instruct checkpoint. The previous run and its reports remain historical evidence. Mining still follows the signed epoch manifest: four distinct completed correct answers and four distinct completed wrong answers per task batch, up to the signed `max_batches` task batches per UID, attempt nonces 0–999, and a 2,048-token output budget. The original published mining/sampling source remains pinned separately from the training execution source.

The corrective trainer accumulates each backward pass into FP32 gradient buffers, computes the global norm and clipping in FP32, and passes those buffers directly to the persistent CPU AdamW optimizer. This avoids repeatedly adding small gradients into BF16 `.grad` buffers. The task-normalized preference objective, clipping limit, Adam coefficients and weight decay retain their declared settings.

An explicit ROOT-signed training release selects an effective learning rate of `5e-7`. It binds the actual execution source, its GPU qualification, the original public input contract and a unique fresh genesis. Initial training starts with fresh optimizer state. Each later update preserves that run's FP32 masters, moments and counters. The coordinator automatically derives an exact job-specific grant after authenticating the native-accepted input subset; no manual signature is required for each epoch. Reports and V2 state descriptors record the effective rate and the original grant.

The private full-model qualification completed two actual GPU training steps, covering both fresh genesis and continuation at `5e-7`, with the same master/moment storage retained between them. It tested computation and state semantics; it did not claim a held-out learning improvement or a durable optimizer transport test. Native eligibility, public artifact admission, publication, retention and successor opening have separate checks.

The coordinator activated this qualified training release on October 9, 2026. Its first corrected production update, epoch 91, completed and published the new checkpoint at 10:55 UTC; epoch 92 subsequently opened on that checkpoint. This establishes a working update and handover, not sustained held-out improvement. The signed live manifest, checkpoint records and training reports are authoritative for deployed behavior. Training can consume eligible unaudited submissions while independent audits determine reward evidence.

Local optimizer supersession requires an explicit signed run transition. The coordinator first journals the exact initial training job, retires only the approved old optimizer after acquiring its leases, checks actual capacity, and dispatches that same job. New-model publication and acknowledged cache retention remain separate. Historical artifacts are not relabeled as outcomes of the new run.

Beginning with epoch 93, the signed schedule allocates 20 minutes to collection and 25 minutes to training/publication, with 12 minutes of reserve and three minutes for capture, audit bookkeeping and weight scheduling. These are budget allocations; they do not prove a complete epoch meets the one-hour target. Existing epoch 92 retains its signed 30-minute collection window. Miners always follow the signed opening and deadline.

## Bounded positive-NLL experiment: optimizer steps 26–42

On October 10, 2026 at 11:17:52 UTC, the coordinator activated a signed trainer-only release prospectively for epoch 117. Its baseline is the published and authenticated epoch-116 checkpoint `81950ed5adf5242498e579edc19be1e112329bdd7317d6b00fe30105781341fc`, at successful optimizer counter 26. The model, FP32 master parameters, Adam moments and counters continue from that baseline; there is no optimizer reset.

For the next 16 successful updates, the trainer adds a positive-answer negative log-likelihood (NLL) term with coefficient `1.0` to the existing preference loss. The added term is the negative mean generated-token log probability of the correct rollout. Pair losses are averaged within each task, then across tasks. The existing preference reference, beta, FP32 gradient accumulation, clipping, Adam settings and effective learning rate of `5e-7` remain in place.

The horizon is defined by successful optimizer counters, not elapsed epochs: the added term applies when the input counter is 26 through 41, producing endpoint 42. At input counter 42 and later, its coefficient automatically returns to `0.0`, retaining the resulting model and optimizer state. Failed attempts do not consume a successful update. The observation plan does not select a best intermediate checkpoint.

Mining continues with the same approved sampler and source, four completed correct and four completed wrong rollouts per task batch, up to nine batches per UID and up to 512 distinct training tasks per epoch. Representative intake selects at most one native-valid batch per task; duplicate-task rewards and independent proof audits retain their existing rules. Miners need no extra action for this trainer-only release and continue following the signed checkpoint and limits. Hourly weights remain independent of training.

The evaluation plan compares the fixed baseline at counter 26 with the exact endpoint at counter 42 on the same two already-exposed 128-task monitoring cohorts. These cohorts are development measurements, not independent confirmation. A separate reserved 512-task cohort has been privately materialized and its reference grading checked. Historical exposure review and qualification of an original-base-versus-trained-checkpoint evaluation remain prerequisites to using it for an independent confirmation claim. Changes in the submitted task population also limit attribution to the added loss term. Activation does not establish a completed NLL update or improved held-out performance. Signed training reports, publication acknowledgments and the fixed endpoint evaluations provide those results when available.

### Measured progress, October 10, 2026 at 12:53 UTC

Epoch 117 completed the first of the 16 planned updates, advancing the retained
optimizer from 26 to 27 and publishing checkpoint
`3900e029e0f27e3e6204b10877294555ac033343f1972d986e65101b34a09b3b`.
It trained on 512 distinct tasks from 192 external miner identities: 2,048
disjoint correct/wrong pairs, or 4,096 rollouts. This is the actual training
population; submitted batches, reward-eligible tasks and audited batches are
different counts. The full controller cycle took 3,681.798 seconds (61 minutes
21.798 seconds), so it did not meet the one-hour target. Model publication and
automatic local retention completed without manual recovery or optimizer upload.

Both fixed baseline evaluations at optimizer counter 26 have completed:

| Monitoring cohort | Correct | Completed wrong | Unresolved | Infrastructure errors |
| --- | ---: | ---: | ---: | ---: |
| Original 128 tasks | 78 | 11 | 39 | 0 |
| Separate exposed 128 tasks | 91 | 6 | 31 | 0 |

All 128 tasks remain in each denominator, including unresolved responses. The
cohorts remain separate. Endpoint 42 has not been evaluated, so these baseline
scores and the completed update do not establish a learning gain. The reserved
final 512-task cohort is separate from these monitoring evaluations; its
confirmation prerequisites are described above. The experiment still uses its
fixed endpoint and does not select an intermediate checkpoint based on
monitoring results.

### Second completed update and preparation speedup

Epoch 118 completed the second planned update, advancing optimizer 27 to 28
and publishing checkpoint
`02e346a2afa42bcb71076a85e1291a50da6f366fccfd06eb74c972eedc31faca`.
Its actual training population was 512 distinct tasks from 179 external miner
identities, again 4,096 rollouts. Training took 1,467.984 seconds; the complete
controller cycle took 3,672.797 seconds (61 minutes 12.797 seconds). Publication,
acknowledgment and automatic local retention completed. This is two of the
16 planned updates; no endpoint learning improvement is claimed yet.

Epoch 119 opened with this checkpoint and the same 20-minute mining window.
The coordinator now supports four concurrent document downloads and 32 native
grading workers for new preparations, preserving the original grader and all
training settings. Historical jobs keep their original preparation authority.

At 13:41 UTC, epoch 119's original training job was running with 512 distinct
tasks from 178 external miner identities and the same 4,096-rollout intake.
Native selection finished in 150.735 seconds, compared with 288.861 seconds in
epoch 118; the two current grading waves account for 96.242 seconds of that
interval. This measures preparation on the actual submitted populations, not
a completed training update or a full-epoch speedup. The fixed 16-update study
still has two acknowledged updates until this job's publication and ACK finish.
