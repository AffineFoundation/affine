# Successor calibration (prospective)

Fast sampler epochs require `successor_calibration` with version
`bounded-successor-calibration-v1`, an approved `env_id`, (no mutable token-budget knob). Before publishing an opening manifest, the coordinator launches an
operator-signed diagnostic evaluation-role job on the sole trainer after the
previous training publication has finished. This avoids competing with the
independent held-out evaluator. The job uses the newly committed checkpoint,
the opening's exact runtime/profile/harness, two actual approved environment task prompts at the exact production output cap, cached draws,
full teacher-forced comparisons, and genuine native TOPLOC replay.

The diagnostic manifest uses its own nonpayable qualification namespace and
explicit qualification-only predecessor-calibration v3 draw context. It is persisted before dispatch. Retries retain
the same original job/manifest; a completed report must pass ordinary remote
job/source/runtime bindings. Its numerical measurements then pass the existing
hard calibration bounds and zero-error native replay checks. The proposal is additionally confirmed by a fresh original job using final-v3-contract uniforms. Only then is the
opening's fast calibration replaced with the actual successor's policy.
A refusal, missing control, wrong checkpoint, native mismatch, or infrastructure
failure holds opening; it never reuses a predecessor's calibration or widens
thresholds. Actual diagnostic reports and digests remain in the local journal.

This is a bounded checkpoint probe, not assurance for every possible long
trajectory or a cross-device qualification. Physical verifier qualifications
remain separate. Training objective/dtype changes need their own signed role
contract; diagnostic calibration does not authorize them. Historical strict
sampler openings bypass this hook and preserve their old behavior.
