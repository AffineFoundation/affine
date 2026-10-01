# TOPLOC full-trajectory verification experiment

A real GPU experiment passed the first milestone: a genuine three-turn Prime Intellect Mastermind rollout passed independent verification; an externally curated control also passed; fourteen resealed adversarial artifacts were rejected. This is an initial same-runtime test, not a security proof or a measured fraud-detection rate.

## Original experimental machine (now externally removed)

- Lium name: `affine-proof-experiment`, HUID `golden-comet-16`
- Pod ID: `8c416640-e9dd-453a-b78f-ce1d9a866df8`
- SSH: `ssh -p 20062 root@90.95.12.246`
- One RTX3090, 24GB, **$0.22/hour**. Created 2026-09-30 15:35:12 UTC.
- Operator requested keeping it running; Lium subsequently records an external `user_initiated` removal at2026-09-30 17:20:20UTC. This agent did not terminate it. Final ledger charge$0.382979.
- Remote source/artifacts: `/root/affine-proof`; venv: `/root/proof-venv`.
- At15:55UTC the billing ledger recorded $0.072714 for1,189.86seconds (19.83minutes). `artifacts/billing.json` preserves the ledger row; `pod.json` preserves connection details. The machine is no longer running.

Production validator services and existing pods were not changed.

## Fixed reference and verified claim

Model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision `31b70e2e869a7173562077fd711b654946d38674` (~2B requested size). Approved safetensors, tokenizer, configuration, and environment-source hashes are in the validator-owned `artifacts/trusted-policy.json`, outside miner ZIP submissions. The verifier checks these hashes before loading the trusted local model. Never accept a replacement trusted policy supplied by a miner.

Environment: pinned Prime Intellect community Mastermind (`vendor/mastermind/UPSTREAM_REVISION`), two digits, four symbols, duplicates allowed, three turns, environment seed42. The harness invokes the actual upstream setup and transition methods through Verifiers0.3.1. Primary reward is binary `solved_reward`, rather than the upstream composite shaping score. Genuine model rollout failed to solve (reward0); the curated control solved (reward1). Environment failure is a legitimate negative training example and does not imply failed inference verification.

For each turn, the artifact includes exact chat-context token IDs, output IDs, decoded text, environment feedback, completion flag, reward, TOPLOC base64 fingerprints, and a float32 matrix of full-vocabulary conditional log-probabilities for every output token. The model forward pass is bfloat16, eager attention, torch2.12.0+cu130. TOPLOC uses128 largest-magnitude final-hidden-state elements, one prefill proof and one proof per16 output-token block. Full verification uses zero exponent/mantissa differences; log-probability tolerance is absolute1e-5.

Generation samples at temperature0.7, per-turn torch seeds100,101,102, max96 output tokens. Fingerprints and probabilities are then produced by a full teacher-forced forward pass on the submitted context/output. This deliberately verifies target-model computation on supplied tokens, not original sampling provenance; externally selected tokens with correct approved-model computation are accepted.

The honest trajectory has3 turns and127 output tokens (56,11,60), and12 TOPLOC proof segments. Warm full verification measured approximately0.47 seconds, including ZIP reading, probabilities, all fingerprints, and environment replay. It excludes loading/hash-checking the3.4GB model and artifact packaging.

## Artifacts

`artifacts/` is ignored by Git and preserved on both the local workspace and experimental machine:

- `honest.zip`, `curated.zip`: complete trajectories, full log-probability arrays, model-state hash manifest, and proofs.
- `complete-rollout.tar`: **single portable artifact containing the honest ZIP and complete safetensors/config/tokenizer model state** (3,438,131,200bytes; local transfer SHA256 independently verified). It also contains a copy of the policy for documentation; that copy must not become the validator's authority.
- `report.json`:16 checks,2 positive controls accepted and14 attacks rejected; per-case timing and resealed SHA256.
- Named attack ZIPs: modified real layer weights with regenerated probabilities/proofs, isolated altered-weight fingerprints, changed claimed state, context, output IDs, missing/extra/copied proofs, observations, rewards, truncated turns, changed probabilities, and NaNs. Every attack is repackaged and rehashed before verification.
- `runtime-freeze.txt`: installed package versions. `*.sha256`: artifact hashes.

## Original GPU reproduction commands (historical; pod removed)

```bash
ssh -p 20062 root@90.95.12.246
cd /root/affine-proof
/root/proof-venv/bin/python pipeline.py generate
# Independent process and independently loaded trusted model:
/root/proof-venv/bin/python pipeline.py verify
/root/proof-venv/bin/python pipeline.py suite
```

`verify` prints JSON with `valid:true` and exits0, or `valid:false` and exits1 for a failed artifact check. For a particular attack use `--artifact wrong_proof.zip`. Policy/model-loading errors fail the process before the verification result. Assertions are protected against optimized Python execution; do not run with `python -O`.

The GPU runtime required building TOPLOC against the installed torch ABI; the published0.1.6 wheel failed an undefined-symbol import. To rebuild:

```bash
MAX_JOBS=4 /root/proof-venv/bin/pip install --force-reinstall --no-deps \
  --no-binary toploc --no-build-isolation toploc==0.1.6
```

The local workspace `.venv` borrows archived dependencies. Its native TOPLOC extension has now been rebuilt against local torch2.14.0; CPU verification works. See `EXTENDED_RESULTS.md` for cross-runtime and random-audit experiments. Production Python was not changed.

## Remaining experiments

Cross-GPU/backend/precision tolerance calibration, repeated trials, segment-level random spot-check budgets, tampering outside TOPLOC's selected activation elements, parser/native-library fuzzing, and larger multi-turn environments remain untested. No claim of cryptographic soundness, universal counterfeit resistance, or unbiased sampling is established. TOPLOC fingerprints do not themselves certify output-head logits; the separate probability checks cover the reported distributions in this full-verification experiment. Partial audit assurance depends on the exact selection policy and fraud distribution.

Upstream references: https://github.com/PrimeIntellect-ai/toploc and https://github.com/PrimeIntellect-ai/community-environments/tree/main/environments/mastermind.

## Extended results

See [EXTENDED_RESULTS.md](EXTENDED_RESULTS.md): CPU/GPU cross-runtime differences, actual altered-weight acceptance under one tolerant configuration, and1.1million frozen-artifact random audit decisions. Neither tested tolerant threshold is installed in the primary verifier.
