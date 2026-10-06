# Prospective prescribed-token proof contract

**Default-off.** The live manifest still selects selected-token logprobs plus
TOPLOC. A new format is usable only when a signed epoch explicitly selects it;
existing submissions and historical verdicts retain their original contracts.

The prospective manifest selects `prescribed-token-artifacts-v1`,
`small-commitment-token-pairs-v3` and the calibrated three-way prescribed-CDF
sampling contract. It omits `probability_artifact_policy`. Generation uses the
checkpoint-bound public draws and approved sampler. Both CPU and GPU rollout
methods skip the additional probability/hidden-state pass and TOPLOC construction.
They retain tokens, prompts, attempt receipts, native observations and outcomes.

Each immutable batch artifact is a bounded canonical `tokens.json` ZIP. Its
miner-signed commitment binds epoch, checkpoint, source, task index, batch hash,
artifact hash and size. Separate `committed-token-training-documents-v2` records
carry the same committed trajectory for cheap learner admission. Miners cannot
supply probability arrays or activation fingerprints in this format.

The verifier reconstructs canonical environment context, makes one causal
teacher-forced prefill per checked rollout, and checks every output position
against the prescribed draws using the approved checkpoint-specific CDF bound.
It independently replays the native environment and checks the reported outcome.
Numerical boundary uncertainty is UNKNOWN, with no cached autoregressive fallback;
a later definite invalidity still rejects the batch. UNKNOWN is neither accepted
proof credit nor evidence of fraud. Matching computations do not establish a
historical execution claim.

An optional authority-signed, finite, exact-job native-source validation scope
hashes the pinned native source once. Subsequent fresh sessions validate ownership,
file membership and inode/time metadata; every rollout still receives a fresh
native grade. It caches source validation, never model answers or grading results.

Training remains decoupled: eligible committed unaudited batches can train without
waiting for proof reports. Audit evidence affects statistical reward estimates
separately. Job-owned local models and downloaded batches retire automatically
only after their required durable R2 acknowledgements and absence of active leases.

CPU codec, actual GPU-method dispatch, fresh-process source admission and historical
compatibility controls pass. Two H200s also matched on the separate pinned research
matrix. Neither result substitutes for fresh adopted-checkpoint generation,
transport, native-cache and backend qualification of this new production source.
That qualification and explicit deployment admission are still pending.
