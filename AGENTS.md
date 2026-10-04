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
