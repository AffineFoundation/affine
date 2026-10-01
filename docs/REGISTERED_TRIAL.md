# Registered miner end-to-end trial

This trial uses one genuinely registered Ed25519 hotkey on Finney subnet120 and a rented remote miner. Its epochs are permanently nonpayable. The experiment never calls chain weight submission or performs payout writer cutover. Registration and activation were separate explicitly authorized setup transactions; the unrelated old transition writer remained active.

Public identity: `5E68nqmVj1gSjoJHbusG2o4SiJ1dq17M7QG5PVzFKK2ic49u`, UID131. Registration confirmed at observed block9182681; actual coldkey balance decrease was1.061166556TAO, including registration burn and fee. Activation revealed at block9182704. Operator wallet remains local; remote miner receives only an expiring single-epoch upload capability decrypted locally from its encrypted mailbox.

The remote miner uses Torch2.14.0+cu130, Transformers5.14.1, verifiers0.3.1 and rebuilt native TOPLOC0.1.6. Inference runs on CPU with float32 eager attention. The signed manifest fixes:

```sh
export MKL_CBWR=COMPATIBLE
export ATEN_CPU_CAPABILITY=default
export ONEDNN_MAX_CPU_ISA=SSE41
```

An initial honest batch with unconstrained CPU kernels failed the unchanged full-logprob absolute1e-5 bound (max delta0.00006103515625). Its rejection is retained under `state/registered-test/`. The profile above aligns execution; no acceptance tolerance was increased. The aligned honest batch passed a separate full audit, including all token probability arrays, strict TOPLOC fingerprints and actual environment replay.

The environment is the real upstream Prime Intellect Mastermind adapter with four short binary instances, two turns and one positive/one negative trajectory per batch. A135M SmolLM2 model samples guesses within a curated text template. This verifies target-model computation, rather than proving unconstrained sampling provenance. One separate trainer process performs a real preference update and changes model weights.

Remote testing exposed a tokenizer packaging gap: Transformers5 saves `chat_template.jinja` separately. Checkpoint hashing/publication now includes it. The incomplete checkpoint remains historical; the repaired complete checkpoint is an immutable new identifier.

Evidence and dashboard sources:

- `state/registered-test-registration/receipt.json`: public paid-registration receipt.
- `state/registered-test-registration/activation-receipt.json` and `activation-verification.json`: protocol activation.
- `state/registered-test-compatible/report.json`: current end-to-end trial result, training and next checkpoint.
- `state/registered-test-compatible/health.json`: current stage, explicit nonpayable/no-weight flags.
- `state/registered-test-compatible/*-report.json`: independent per-miner audits, bound to frozen artifact hashes.
- Public HTTPS gateway: `https://the-organization-initial-latex.trycloudflare.com`.
- Public manifest/audit/score paths: `/public/<epoch>/manifest.json`, `/public/<epoch>/audits/<ed25519-public-key>.json`, `/public/<epoch>/scores.json`.
- Signing authority: `93c8b5e7e86f7378bdc911afa17f0ddcdf29b078e18690a2229cc7aa7dae2e8b`.
- First accepted epoch: `nonpayable-registered-1790795876`.
- Completed trained-checkpoint epoch: `nonpayable-registered-1790795876-next-retained` (the earlier incomplete attempt remains historical).

The public gateway is retained by `affine-registered-test-gateway.service`, with a separate temporary tunnel `affine-registered-test-tunnel.service`. The URL may change if the temporary tunnel restarts. The underlying R2 bucket remains private; only frozen artifacts are exposed publicly. New production payout execution is inactive.

Originally retained Lium pod: `816e4656-05f5-4b11-afee-cd823068ef17`, name `affine-registered-miner-test`, oneRTX3090 at$0.22/hour. It was externally deleted at19:34:48→19:35:27UTC with authoritative Lium reason `user_initiated`; this agent did not remove it. The replacement completed the final second remote round. Historical SSH connection: `root@90.95.12.246:20062`; use dedicated known-hosts file `state/registered-pod-known-hosts`. Runtime and miner source live under `/root/miner-venv` and `/root/affine-miner`. Capabilities and operator keys must never be rendered in the dashboard.

Current overall status is **complete**: the first remotely produced positive/negative batch passed full independent audit, one actual trainer step changed weights, the complete checkpoint was published, and a fresh remote next-epoch batch passed full audit against those trained weights. All epochs remain explicitly nonpayable and no chain weights were submitted.

The original rental was removed by the existing automatic pod reaper after 90 minutes because its ownership registry entry was missing. The replacement pod is `0a67adc9-b484-4172-96c1-073efec7d8be` (`zesty-comet-b2`, `affine-registered-miner-retained`), one RTX3090 at $0.22/hour, SSH `root@90.95.12.246:20059`. It is running and registered with owner `manual:user-retained-verification`, expected lifetime `0` (indefinite), source `explicit`. The global production reaper was left active. Its known-hosts file is `state/registered-pod-retained-known-hosts`. No operator coldkey or private hotkey was exported.

13 protocol/chain checks pass. Four actual adversarial controls against the trained checkpoint also reject: fabricated score, fabricated environment feedback, altered full log-probabilities, and a TOPLOC proof substituted from another trajectory. Acceptance thresholds were not widened. Evidence is `trained-checkpoint-tampering-controls.json`; signed public copies are under the first accepted epoch, alongside `experiment.json`, `health.json` and `chain-activity.json`.

Every future rental must be added to the existing Affine pod ownership registry immediately. For an explicitly retained experimental pod, `/home/const/subnet120/.venv/bin/python /home/const/subnet120/ops/pods/registry.py register NAME --purpose verification_experiment --owner manual:user-retained-verification --hours 0 --price 0.22` prevents the automatic ownerless-pod timeout without changing global cleanup policy.
