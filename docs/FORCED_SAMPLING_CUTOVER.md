# Forced sampling cutover

The public miner must discover the approved model's successful and unsuccessful
rollouts. Model probabilities and TOPLOC alone verify computation on supplied
tokens; they do not establish that a sampler selected those tokens.

## Contract

`forced-inverse-cdf-replay-v1` activated from signed epoch nine on October 4,
2026. Changes remain prospective: old epochs retain their original rules. The
controller generates 32 bytes of epoch randomness before opening uploads. Public
SHA-256 draws bind the contract version, epoch, checkpoint, environment, task
index, bounded attempt number, turn number, and token position. Miner identity
is excluded: extra identities cannot obtain different draws for the same task.
Attempt numbers range from zero to the signed maximum minus one. A miner can
search those attempts for one success and one failure, but cannot choose a new
seed, change temperature/top-p, or supply a different sampling algorithm.

The shared sampler applies signed temperature and top-p, stable ordering for
top-p ties, and inverse CDF in ascending token-id order. The first release pins
uncached eager autoregressive computation and exact replay, with no match-rate
threshold, short-trace exemption, or numerical tolerance that admits wrong picks.
An audited turn is regenerated under the pinned model and numerical profile;
every token and the stopping position must match. The independent existing
probability/TOPLOC and original environment replay checks remain required.
This proves consistency with the approved generation procedure, not historical
execution or unbiased selection of successful/unsuccessful attempts.

## Execution and acceptance

1. Add the contract, source binding, shared sampler, and fail-closed validation.
2. Wire epoch opening, CPU/H200 miners, CLI, verifier, training re-audit, and
   report attribution. Keep held-out evaluation on its precommitted old sampler;
   explicitly mark it diagnostic and prevent it from becoming mining evidence.
3. Test honest generation and synthesized answers with genuine recomputed
   probabilities/proofs, arbitrary/out-of-range attempts, settings/context
   changes, prefixes, EOS manipulation, malformed contracts, and legacy isolation.
4. Run real H200 positive and negative rollout controls and independent replay.
   Record wall time and false rejection rate. Failure keeps production on hold;
   do not loosen the sampler to make a control pass.
5. Publish an immutable source bundle and admit it on miner, both verifiers,
   trainer and evaluator. Preserve checkpoints, frozen inputs, job history,
   original earned rewards, and chain writer cursor.
6. Activate only at a completed boundary, or an explicit suspended-epoch recovery
   that preserves the old contract and pending work. Never apply new fraud rules
   retroactively to existing submissions. Exclude old-contract artifacts from
   new-contract training/reward evidence.
7. Publish the actual new manifest and guide, announce the coordinated client
   update, and run an owned miner through the public submission/audit/training
   path. Recheck signed reports, checkpoint publication and weights attribution.

R2 remains durable history; local worker files are disposable cache. This change
does not introduce additional one-off cleanup watchers. Numerical portability
and verifier throughput must be measured; strict replay adds generation cost.
