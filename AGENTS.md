# Affine inference-verification rewrite

This is the clean rewrite workspace for Bittensor subnet 120 (Finney).
The intended direction is inference verification through sampling, inspired by Reliquary (SN81). The synchronous epoch architecture and duplicate-zero scoring are specified in IMPLEMENTATION_PLAN.md and docs/ARCHITECTURE.md; audit economics and the research training objective remain provisional.

Read STATE.md for transition status and archive locations.
Keep this file short; put detailed design and operational records in separate documents.
Do not carry forward the archived distillation contract as the new design.
Keep credentials in 1Password or the operator environment; never commit tokens or wallet private keys.
Production remains in /home/const/subnet120 until the operator specifies how services should transition. Changes in this checkout do not deploy automatically.
