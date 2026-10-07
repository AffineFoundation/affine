# Miner-bound K2/L2 cutover candidate

This source is a prospective versioned contract, pending genuine GPU qualification and ROOT activation. Historical signed openings remain unchanged.

`forced-inverse-cdf-prefill-miner-bound-v5` preserves the active v3 calibrated-prefill and exact cached support-adjudication semantics and numerical thresholds. Its changed draw context includes the authenticated miner identity (existing signing-key public hex64), epoch and checkpoint; uniform draws also bind environment, task hash/index, attempt, turn and token position. It does not prove physical historical execution or generation time.

V5 requires K=2, L=2, max_batches=3, and max_attempts=1000. Nonces are the existing integer `rollout.seed` / `sampling.attempt`, 0 through999, scoped per miner/task/epoch/checkpoint. Miner nonce search is allowed within that finite range. A new prompt prefix is unnecessary.

Cheap admission checks exactly four samples, two claimed positive and two claimed negative, four distinct attempt IDs, exact miner-bound sampling receipts and distinct generated output-token trajectories. Content identity excludes prompts, labels, filenames, timestamps, proof metadata and attempt IDs, so changing metadata or prefixes cannot turn copied outputs into diversity. Full inference/environment verification remains the audit role; unaudited cheap admission does not assert actual success, failure or correct model sampling.

Authenticated frozen child ownership supplies verifier miner identity. The signed audit report carries `sampling_miner`; API admission compares it with the signed job's unique frozen submission owner. The worker never trusts a rollout's claimed miner.

The shared new module `subnet/sampling_uniqueness.py` joins scientific source inventory. Any live activation requires a fresh exact-source qualification and new approvals. Old source/version signatures and old calibration approval flags must not be reinterpreted. This candidate does not activate token-only transport, change learning objectives, reset optimizer history or increase task-batch rewards.
