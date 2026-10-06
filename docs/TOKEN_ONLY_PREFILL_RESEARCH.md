# Token-only prefill research (default off)

The scaffold in `subnet/token_only_verifier.py` is not imported by any deployed
worker and has not been added to a production runtime inventory. It requires an
explicit `token-only-prefill-research-v1` research policy plus the existing signed
`forced-inverse-cdf-prefill-support-v3` checkpoint-bound calibration. The caller
must authenticate that manifest, load its exact checkpoint and supply its
approved training-task indices, excluding the reserved heldout cohort.

A token document retains the task, attempt, public-draw receipt, prompt, output,
decoded text and environment outcomes. The verifier reconstructs canonical
prompts and performs one causal teacher-forced model prefill per trajectory. It
computes reference logits locally and checks every claimed token against the
original independent public inverse-CDF draws. It replays the native environment
and requires the signed positive/negative quota and distinct output sequences.
The normal path neither regenerates autoregressively nor constructs TOPLOC.
The existing v3 support/boundary exception path retains cached adjudication;
unavailable adjudication stays infrastructure/ambiguity, not fraud.

This validates a checkpoint-consistent, policy-consistent token sequence; it
cannot demonstrate historical miner execution. TOPLOC fingerprints can also be
freshly computed on a deliberately chosen sequence, so removing the sampler
check would admit genuine-proof off-policy forgeries. Uploaded full-vocabulary
probabilities are redundant for this verifier's sampler check: it must compute
the full reference distribution itself. The current compact policy already
uploads only selected-token logprobs; eliminating those removes an explicit
miner probability claim, not the verifier's probability computation. It does
not eliminate model execution, native grading, tokenization or object/signature
admission costs.

The calibrated CDF contract remains approximate. Its existing exact fallback
covers zero support and draws outside a prefill interval but inside its calibrated
uncertainty band. Draws inside an interval near a boundary currently pass without
cached adjudication. This research preserves that behavior and makes no stronger
on-policy claim. Changing that policy requires a distinct signed contract and
cross-device evidence. Approved attempts/task choice are permitted selection;
reusing another attempt's tokens with a freshly rebound receipt must fail.

## Two existing idle H200 controls for ROOT review

No machine, source grant, service or GPU job is changed by this proposal. ROOT
must identify two genuinely idle existing physical GPUs and exclude the current
learner, miner, reserved independent reader and any original evaluation job.
Use one bounded nonpayable namespace, immutable source/module pins, full
checkpoint SHA readback, strict FP32 profile, exact current calibration and native
math spec, and an explicitly signed token-only policy. Do not widen tolerances.

1. Generate fresh genuine positive/negative pairs from two precommitted approved
   task indices on GPU A with the existing cached generator/public draws. Save
   the original compact+TOPLOC artifacts as comparison evidence, and export
   token-only documents without resampling. Set a signed bounded attempt cap.
   An inability to acquire both classes is a truthful control failure.
2. Independently verify identical documents on A and B using the current compact
   verifier and token-only prefill. Require matched honest verdicts and native
   outcomes. Include at least 80 precommitted output-position/token swaps,
   freshly recomputing genuine probability/TOPLOC evidence for each modified
   sequence. Also test seed changes with original receipt, valid in-budget seed
   changes with rebound receipt, exhausted attempts, changed task/prompt,
   replayed checkpoint, duplicate outputs and quota changes. Independent draws
   must still reject teacher-forced off-policy tokens despite genuine proofs.
3. Record per-trajectory output length, prefill count, fallback count, synchronized
   CUDA wall time, object GET time/bytes, tokenizer/native grading time, packing
   time/bytes, peak VRAM and every original process/terminal/report digest. Time
   warm and cold model paths separately. The 0.2–0.4-second assertion is an
   unmeasured hypothesis until these full native-task controls complete; CPU
   tiny fixtures cannot establish production H200 latency.
4. Keep numeric ambiguity separate from invalidity. Measure actual CDF/logprob
   differences across A/B, especially nucleus support and both sides of interval
   boundaries. No finite small calibration proves a global bound. For future
   stronger assurance, a distinct policy can replay the entire causal sequence
   when a public draw is within delta of either interval boundary, or include
   unpredictable postcommit exact-reference audit draws. For genuinely uniform
   independent draws over a bad population fraction f, n full-trajectory checks
   detect at least one with probability 1-(1-f)^n. That formula does not establish
   f or cover adaptive selective submission without a proper population design.

CPU controls use a real tiny causal model and genuine fresh TOPLOC, not native
H200 performance evidence. They establish honest-pair acceptance, one-prefill
normal execution, 80 off-policy swap rejections, receipt/seed binding, task
eligibility, prompt/outcome/framing checks, quota/deduplication and session cleanup.
