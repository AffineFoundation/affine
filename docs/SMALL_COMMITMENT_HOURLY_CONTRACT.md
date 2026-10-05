# Prospective hourly commitment contract

Review candidate only. Current live epochs keep their originally signed contracts.
Activate only after source admission, new contracts, actual GPU qualification and a safe completed epoch boundary. A phase policy is a budget, not evidence that a complete epoch has met an hour.

## Submission

`submission_transport_policy = small-commitment-pairs-v1` uses one cumulative small signed JSON commitment and up to `max_batches` separately uploaded ZIP objects. Each ZIP contains one complete batch for one approved task: K successful and L unsuccessful rollouts under the exact epoch checkpoint, forced sampler and TOPLOC rules. Upload the heavy slot objects first, then overwrite the small commitment. The provided clients use deterministic per-pair ZIP framing and a durable append-only acknowledged-slot journal, skipping unchanged successful object PUTs. A failed later commitment update therefore does not overwrite prior acknowledged pair bytes; confirmed by cumulative-upload/restart controls. The server-side cumulative commitment and fixed slot objects are still mutable before freeze. There is no earlier acknowledged snapshot history: a later replacement can supersede the previous commitment or make it ineligible if completed late. Only the exact first-observed commitment, ETags and snapshots actually journaled by freeze survive recovery; failed pre-freeze replacements have no automatic rollback guarantee.

The commitment is Ed25519 signed by the registered miner public key. Its payload binds version, epoch, miner, checkpoint SHA, source SHA and ordered batch slots. Every slot binds environment/index, canonical batch SHA, ZIP SHA and size. The commitment limit is 65,536 bytes. Duplicate task indices or payload hashes within a miner's commitment are excluded. The object-scoped presigned URLs expire at the mining deadline and grant no listing, reading other miners, authority signing or bucket credentials.

Upload count, deadline and eligible object bytes are bounded. Direct presigned R2 PUT does **not** enforce an absolute byte limit before storage accepts the upload: HEAD size eligibility is checked before snapshots and selected downloads. This is a storage ingress abuse limitation, not proof that unlimited uploads are eligible.

The owned miner uses `owned_miner_identity_files` mapping registered hex public keys to miner-host-only absolute 0600 seed files containing the 32-byte Ed25519 seed as hexadecimal. Only the local miner reads the seed; operator jobs contain the path, never the seed or authority key. External `Miner.upload()` uses its existing local identity.

## Freeze and audit

Writes close at the declared deadline. The coordinator reads only bounded signed commitments, HEAD metadata and conditional server-side copies. Original commitments, ETags and partial copied slot receipts are journaled before continuing; restarting cannot substitute changed objects. Each malformed/missing commitment is structurally excluded independently, with no fraud penalty by default. Transient infrastructure problems retry; at the signed freeze cutoff incomplete miners are recorded as infrastructure-budget-deferred, while completed miners proceed.

After the frozen eligible population is closed, a fresh unpredictable challenge is persisted once. Existing bounded-random-v1 allocation and selection choose complete pairs fairly. No mandatory audit for every UID is required when the configured budget is smaller than the population. Zero-allocation miners have explicit no-credit/unverified reports and no invented GPU jobs. Unverified task claims cannot cancel an honest verified point.

Selected verifiers fetch only selected heavy objects and check actual full SHA before decode, original canonical batch commitment, target model computation, forced sampling, TOPLOC and environment outcome. Selected computations retain the existing numerical policy. Only actual fully audited accepted batches earn points and enter compact authenticated training inputs. Trainers admit signed verifier receipts and do not repeat model verification.

At the signed audit cutoff unfinished/late audits receive `budget_deferred` no-credit reports. Execution failures receive `infrastructure_deferred` reports. Neither can assert accepted samples, fraud, or fabricated job IDs. Original signed queue requests/leases remain intact and workers may finish forensic reports later; a closed epoch never awards retroactive credit.

## Penalties and temporary exclusion

Suggested prospective penalties are `invalid_batch_multiplier=0`, `zero_epoch_after=1`, `penalize_structural=false`. Only explicit `valid=false`, `fully_audited=true`, `failure_kind=confirmed_invalid` counts as fraud. Infrastructure failures, malformed unaudited structure, budget deferrals, timeouts and unselected submissions do not.

Optional `temporary_exclusion_policy` has version `confirmed-invalid-temporary-exclusion-v1`, `threshold_epochs`, `lookback_epochs`, and `exclusion_epochs`. Repeated confirmed-invalid epochs temporarily allocate zero audits/credit to that identity. The signed next manifest contains its exact signed historical snapshot and exclusion list. The reward writer independently checks the historical signed reports and original authenticated worker lineage; a fabricated history cannot authorize rewards or exclusions. Cooldowns count completed epochs and expire prospectively.

## Synchronous training and independent evaluation

The next mining epoch waits for durable trainer optimizer state and checkpoint publication. The actual persistent parent/counter is retained; source changes do not reset the genesis or relabel historical training. Optional signed `optimizer_state_transport={"version":"bounded-parallel-fp32-state-v1","concurrency":4}` admits bounded parallel FP32 export and restore only when actual resource headroom passes; absence remains serial. Root-approved bounded parallel optimizer readback still reads every shard and verifies its actual size/SHA before signing the descriptor last.

`evaluation_mode=independent-checkpoints-v1` keeps the fixed held-out task cohorts, seeds, runtime and comparison identity, while evaluation observes committed checkpoints independently of the mining/training critical path. Latest training checkpoint and evaluated checkpoint remain different until an actual report completes. No synthetic evaluation score, cumulative training step, or performance improvement is implied by source activation.

Suggested `hourly_execution_policy` (version `bounded-hourly-phases-v1`): mine 600 seconds, freeze 300, audit 600, train/publication 1200, weights 300, slack 600, totaling 3600. Total includes durable publication and actual chain transaction. The candidate enforces freeze/audit budget closure; trainer/publication capacity and actual consecutive completed epochs still require measured qualification. A no-update or chain-disabled epoch is not proof of the full hourly goal.
