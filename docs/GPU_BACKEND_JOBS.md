# Signed retained-GPU role jobs

`subnet.backend_jobs` is an isolated remote worker for operator-authorized verify, train, evaluate and upload jobs. It never imports a chain weight writer, wallet loader or Cloudflare/R2 account credential. Controller integration is separate; the existence of this module does not establish that the live service selects it.

The operator signs both the outer job payload and its nested `manifest` envelope with the approved Ed25519 authority. An envelope is `{signer, signature, payload}`, using base64 signatures over canonical sorted compact JSON. The worker authenticates both envelopes before checkpoint or submission reads, validates expiry and role, and checks the same approved checkpoint identity and exact file hashes. Each role uses the strict manifest contract:

```json
{
  "model_runtime_revision": "cuda-bf16-eager-sm86-v1",
  "numerical_policy": {
    "logprob_atol": 0.00001, "logprob_rtol": 0,
    "toploc_exp_mismatches": 0,
    "toploc_mant_err_mean": 0, "toploc_mant_err_median": 0
  },
  "backend_profile": {
    "device": "cuda", "dtype": "bfloat16", "attention": "eager",
    "sm": [8, 6], "tf32": false, "deterministic_algorithms": true,
    "cublas_workspace_config": ":4096:8",
    "native_toploc_threads": 2, "torch_threads": 2
  }
}
```

CPU numerical policy remains unchanged. A CUDA job cannot relax TOPLOC errors or full-logprob tolerance or claim CPU/GPU interchangeability. Runtime package versions are pinned separately in the job as exact `torch`, `transformers` and `toploc` installed versions. `source_files` must include every `SOURCE_FILES` Python path with its SHA256; additional subnet source files can be pinned. The fresh worker verifies these paths, rejects symlink sources and refuses already imported model/runtime modules, then installs a source-only loader for subnet imports, avoiding untrusted cached `.pyc` execution. Environment source/dataset/harness validation remains the existing approved spec boundary. Execution resource enforcement is reported false until the resource-session adapter is genuinely integrated.

Run in a fresh process on the retained pod, with deterministic environment set before importing Torch:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 /root/miner-venv/bin/python -B -m subnet.backend_jobs \
  /root/private-job.json --authority APPROVED_PUBLIC_HEX \
  --workspace /root/gpu-role-jobs \
  --checkpoint-cache /root/approved-cached-checkpoint
```

The outer payload requires `schema:1`, an alphanumeric/hyphen/underscore `job_id`, `role`, `created_at`, `expires_at` (at most one day), `manifest`, `source_files`, and `runtime_versions`. The approved manifest uses the normal epoch/environment registry, K/L quotas and checkpoint `{id, files, read_urls}`. Cached model bytes are accepted only after hashing the complete relevant file allowlist. Without a cache, the worker streams exact per-object presigned R2 GET URLs into its remote checkpoint directory. Capability HTTP requests reject redirects and never fall back to a tunnel.

- **verify:** `submissions:[{url,sha256}]` contains at most 256 frozen ZIP capabilities. Each ZIP is digest-checked, safely unpacked and fully checked for epoch/checkpoint/environment/index binding, duplicate outputs, positive/negative quotas, every TOPLOC fingerprint, full probability arrays and environment replay. Reports preserve each artifact hash and audited outcomes. Only fully audited batches are training eligible.
- **train:** the same submissions are independently audited in the trainer process, even if an earlier verifier report exists. `steps` is 1–32. Accepted positive/negative pairs are routed to their exact environment and harness. Each actual output-head preference optimizer update writes a complete new model/tokenizer checkpoint remotely. Metrics state `full_model_finetune:false`, reference-relative frozen-decoder-feature objective and whether the input embedding is tied to the updated output head. This is real partial-parameter optimization, not a claim of task-quality improvement or full-model training.
- **evaluate:** `heldout:[{env_id,indices,seeds,harness}]` pins fixed heldout indices and seeds, disjoint from training indices. The heldout harness must be free autoregressive with no curated turn overrides. Each generated trajectory is independently replayed and its TOPLOC/probabilities verified. Scores report the observed task rewards without inventing improvements.
- **upload:** after training, the operator retrieves the report over its existing authenticated SSH connection, checks checkpoint hashes independently, and creates a second signed job whose approved checkpoint is the new exact file map. `put_urls` binds exactly one presigned PUT to each file, with `Content-Type:application/octet-stream`. The worker sends only those approved remote bytes. No permanent bucket credentials are exported. Operator-side publication/signatures and anonymous read URLs happen after independent hashes are checked.

Each unique job gets a private workspace directory; reusing its ID is rejected. Reports at `workspace/jobs/job_id/report.json` include the operator public key, signed job digest, checkpoint identity, exact source/package pins, strict numerical/backend policy, epoch, truthful resource-enforcement status and `chain_transactions:false`. Reports are worker outputs, not independently signed operator evidence. The operator must verify and sign published provenance rather than treating a remote report as a cryptographic attestation.

The additional **mine** role creates a fresh GPU batch for the signed challenge. It receives only `miner_id` (the owned public Ed25519 key), `seed_start`, `search_budget` (at most128 attempts per authorized index), and `capability:{put_url,headers:{"Content-Type":"application/octet-stream"}}`. It searches only manifest-approved indices, retains distinct target-generated positives/negatives until K/L quotas, packs one cumulative ZIP and uploads to that one delegated R2 object. This demonstrates an operator-delegated owned identity in a permanently nonpayable experiment; it does not claim that a remote miner independently signed its registration or that an operator signature proves original stochastic sampling.

`ops/new_subnet_gpu_role_smoke.py` orchestrates isolated signed roles from the operator machine against the retained pod. It preserves the archived old environment source version, records the existing UID131 registration snapshot block (without pretending it is a new chain query), opens a separate direct-R2 nonpayable Controller/Gateway namespace, delegates bounded mining, freezes its final upload with an explicit early-freeze test policy, fully audits, trains one real head update, evaluates fixed heldouts, uploads the complete new checkpoint using six object capabilities, independently streams and hashes every R2 model/tokenizer object on the operator, and only then signs the new descriptor and opens a fresh checkpoint challenge. Reports are under `state/multi-environment/gpu-service/<timestamp>/`; successful completion must be established from actual reports, not this procedure description. The retained GPU pod stays running.
