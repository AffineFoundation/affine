# Explicit compact-proof three-way sampling

This prospective option is default-off. It changes only sampling adjudication; selected-token-logprobs-v1 artifacts still require their selected probability arrays and valid TOPLOC proofs. The model runtime, artifact parser, and trainer are unchanged. Their existing checks run before CDF adjudication.

A new opening must explicitly select:

```json
{"version":"forced-inverse-cdf-prefill-threeway-v4","max_attempts":16,"calibration":"exact checkpoint/profile/harness-bound admitted calibration object","uncertainty_adjudication":"numerical-inconclusive-no-replay-v1"}
```

The actual calibration value must be an object accepted by the existing validator. The signed contract uses verification=prefill-cdf-calibrated-threeway and generation=cached-eager-inverse-cdf. All public draws, attempts, token/stop framing and provenance remain checked. A distinct contract changes the public draw domain; previous v3 trajectories or research reports cannot be relabeled as v4 qualification. The existing successor calibration hook runs for v4, including final-contract confirmation.

Outside-calibrated-region draws reject. Supported draws safely inside both boundaries pass. Near-boundary draws and zero-support positions inside the admitted uncertainty region are numerically inconclusive, with no cached autoregressive fallback. Zero support never passes. A known outside-bound failure wins over another uncertain position in the same prefill. Unknown reports have valid=null, fully_audited=false, failure_kind=numerical_ambiguous, sampling/environment verification incomplete, and bounded token-position metadata when available. They do not enter accepted training receipts or confirmed-cheating penalties. Existing v1/v2/v3 behavior remains unchanged.

Unaudited committed-input training remains independent. Its immutable reference is already computed by task_normalized_training.train_epoch from checkpoint-local BF16 causal log probabilities under no_grad before gradients or optimizer steps. It does not read miner-uploaded reference probabilities or substitute zeros.

Before activation: seal/qualify the new source, verify the full coordinator API dependency closure accepts v4 reports under an explicit source admission, and issue fresh v4 honest generation/compact-TOPLOC verification controls with forged tokens and seed rebinding on the exact checkpoint/profile/harness. The observed two-H200 v3-draw research was three accepted short trajectories and one long numerical unknown, not four equivalent accepted fast verifications. A representative precommitted cohort must quantify honest uncertainty and targeted uncertainty avoidance. Unknown coverage requires an explicit incentive rule; omitting unknowns from conclusive validity estimates alone may reward avoidance. No coverage, reward, training objective, production configuration or activation changes are made here.
