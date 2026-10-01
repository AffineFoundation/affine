# Temporary equal registration weights

Only newly activated model registrations from Finney SN120 block 9181759 (2026-09-30 15:54:34 UTC) qualify. Require a model upload whose manifest the existing R2 registration handler has verified (queued or crowned), an activation and ready reveal at or after the cutoff, and a currently registered hotkey. Old activations, UID registration alone, unuploaded or rejected models and the owner are excluded. Deduplicate by hotkey/UID. Each eligible UID receives 1/N.

Runtime: ~/.local/share/affine-burn/equal_registration_weights.py using the existing Bittensor 11 environment. Config, lock and last status: ~/.local/state/affine-transition/. A user-systemd timer checks every minute; rate limits defer submissions and SDK planning checks chain policy, including cardinality constraints. No eligible models means no submission; existing on-chain weights remain until a permitted update. Exact equal weights are subject to chain quantization. Chain denial is recorded and never silently replaced with owner weights.

The owner burn timer is disabled. The original validator was restarted with a guard in _maybe_set_weights that checks ~/.local/state/affine-transition/active. Intake/evaluation remain running, while legacy payout writes are suppressed. The guard persists after transition expiry to avoid automatically restoring historic payouts.

Submissions stop after 2026-10-02 15:54:34 UTC, the announced maximum two-day transition. At expiry, on-chain weights remain until an explicit next policy; no automatic fallback. To resume competition payouts intentionally, first stop/disable affine-transition-weights.timer, then remove the active marker. Do not run competing weight writers.

Verified eligibility cutoff, incomplete/rejected model exclusions, deduplication, live dry run and live scheduled run with zero qualifying models. No weight extrinsic was submitted on installation. Timer active, original validator online. Legacy affine1 public HF submissions are not included by this implementation; current intake is affine2/private R2 registrations.
