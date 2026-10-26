# Affine inference-verification rewrite

This is the clean rewrite workspace for Bittensor subnet 120 (Finney).
The intended direction is inference verification through sampling, inspired by Reliquary (SN81). The synchronous epoch architecture and duplicate-zero scoring are specified in IMPLEMENTATION_PLAN.md and docs/ARCHITECTURE.md; audit economics and the research training objective remain provisional.

Read STATE.md for transition status and archive locations.
Keep this file short; put detailed design and operational records in separate documents.
Do not carry forward the archived distillation contract as the new design.
Keep credentials in 1Password or the operator environment; never commit tokens or wallet private keys.
R2 is the durable record; machine storage is disposable cache. Integrate automatic retention into normal job/epoch completion: keep current weights and in-flight inputs, then remove completed rollout downloads and obsolete weights once their durable bucket records are confirmed. Prioritize this lifecycle over one-off machine cleanup.
Production remains in /home/const/subnet120 until the operator specifies how services should transition. Changes in this checkout do not deploy automatically.
The operator's name "Arbos.life" (sometimes transcribed as "Arbos Lite") refers to this current machine: the code workspace and existing validator host. Keep the new validator here when separating Lium miner, verifier and trainer nodes. This naming clarification does not authorize changes to an Arbos website or DNS.

Live operations: derive the running protocol and source from the systemd learner service's signed policy and authenticated epoch manifest. Do not start the archived `affine-live-math-controller-v1.service` or run `subnet.gpu_service` with `config.active.private.json` against live state: that historical configuration omits the current sampler and decoupled training contract. The obsolete unit is masked following the 2026-10-09 recovery. Preserve the current checkpoint/optimizer lineage and use the supervised learner, API and auditor services for recovery.

Weight-setting instruction, superseding the previous qualification payout hold (2026-10-09): the user explicitly requires hourly network weights from the best current authenticated miner assessment, retaining the last valid assessment during evidence or training outages and mapping identities to current registrations. Never burn or fall back to owner-only weights. Training qualification must not disable reward updates. Preserve transaction evidence and reconcile uncertain submissions without replaying old windows. The legacy burn service and timer are masked.

Active hourly writer runtime and policy are recorded at the top of STATE.md; preserve the single transaction cursor when upgrading. Never restart archived payout or burn entrypoints.

Learning authorization (2026-10-09 20:04 UTC): the user explicitly permits changing batches per miner, samples per batch, training batch size, learning rate and optimizer, plus speed optimizations, to obtain measured held-out improvement. Preserve running experiments until their original results are captured; keep miner-facing public code and llms.txt compatible with any prospective protocol change. This supersedes earlier fixed-quota/no-protocol-change limits for future changes, not historical signed epochs. Continue hourly best-assessment weights with no burn.

Adversarial investigation priority (2026-10-11): treat miner submissions as untrusted. When learning deteriorates or trajectories look suspicious, investigate prompt/answer leakage, trusted checkpoint and randomness binding, grading exploits, duplicate credit, selective submissions and verifier bypasses early, alongside optimizer hypotheses. A correct answer or passing sampling replay does not establish honest historical execution or sound reasoning. Compare miners under matched task/checkpoint/sampler conditions; distinguish a demonstrated exploit from permitted search, statistical suspicion and ordinary model error before attributing misconduct or applying penalties.
