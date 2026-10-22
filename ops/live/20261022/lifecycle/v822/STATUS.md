# MiMo v8.2 release-succession — STATUS (option B)

Branch: release/mimo-v8.2-release-succession @ a4742217 (off v8.1 8ce3c966)
Worktree: /home/const/affine-mimo-release-v82

## DONE
1. Release-succession mechanism implemented + committed (a4742217):
   - New subnet/mimo_release_succession.py: ReleaseChain (signed, append-only chain
     of release policies ordered by first governing round) + load_release() dispatcher.
   - Drop-in guard: selects the exact signed chain member each manifest names,
     re-authenticates it, validates against it; newest member governs new epochs.
   - Lifecycle: optional release_succession; _adopt_succession() re-stamps a CURRENT
     bound to a predecessor onto the active member IFF the signed chain links from it
     AND the cursor sits at the active initial (cutover) checkpoint; drops the stale
     checkpoint_authorization. Fail-closed otherwise.
   - Trainer: _adopt_release_succession() re-stamps the private CURRENT (release +
     source closure) onto the successor, preserving optimizer/model lineage, when a
     signed chain in the owner-only role config authorizes it and the cursor continues
     the durable checkpoint/step. role config now allows a 'release_succession' field.
   - Evaluator: accepts a state release that is any chain member; validates each opening
     against the member it names; baseline = chain root initial checkpoint.
   - Writer (EvidenceSource): load_release() so the reader policy's release_policy may be
     a chain; guard + policy_sha256 come from the chain.
   - ops/mimo_lifecycle.py: builds the chain from config.release_succession and wires the
     chain guard into the audit scheduler, evaluator and lifecycle.
   - Minimal: single-policy path unchanged when no chain configured.

2. Tests (tests/test_mimo_release_succession.py) — 34 pass, incl. the required cases:
   v7->v8.1 succession accepted; tampered/unsigned/wrong-authority/non-append-only chain
   rejected; a predecessor manifest still verifies after the switch while the single
   active policy rejects it; new epoch planned under the active member; trainer adoption
   keeps the optimizer lineage. Regression: 79 passed + 18 subtests across every touched
   suite (release_policy, lifecycle, evaluation_scheduler, audit_bridge, hourly_writer,
   current_assessment_compact_release, grpo_worker).

3. DRY-RUN against a COPY of live state (never the live dirs) —
   DRY-RUN-RESULT.json, chain = [v7 df2d2b80, v8.1 c50804b6], run affine-mimo9b-grpo-20261013:
   - AUDIT: 34/34 live historical populations authenticate under the chain guard;
     34/34 rejected by the single active policy; AuditScheduler.populations() returned 34.
   - EVALUATOR: state policy is a chain member; 35/35 stored openings validate.
   - LIFECYCLE: live CURRENT (df2d2b80, round 223, epoch 34, ckpt c1c4c5e2...) is a chain
     member -> eligible for the in-code adoption.

## NOT DONE (gated on GPU + authority-signed re-staging of the cutover-scoped v8.2 scope)
A. Re-stage the v8.2 candidate policy/chain scoped to the CURRENT cutover checkpoint
   (c1c4c5e2... at activation time; it advances each epoch) carrying KL 0.02, length
   penalty, max_tasks 256, first-claim. v8.2 scope = v8.1 scope with source_bundle ->
   v8.2 source archive (now includes a4742217) and initial_checkpoint -> live cutover.
B. Re-run all 7 qualifications FRESH against the v8.2 source where changed code is
   exercised (runtime, trainer-update incl. a GPU job on ONE affine-* Lium pod torn down
   after with secret wipe; cross-device-sampling, heldout-evaluation, model-publication,
   mutation-controls, native-grading). Old v8.1 receipts cannot be reused: source changed.
C. Sign everything with the on-box authority key (3301134b), v7 pattern: v8.2 policy,
   the release-succession chain envelope, qualifications, writer-reader (approved_execution
   for BOTH v7 and v8.2 source shas), writer-composition, HOST-COMPOSITION-accel (re-pin to
   v8.2 bridge hashes), lifecycle config carrying release_succession + v8.2 release_policy,
   trainer role config carrying release_succession.
D. Full dry-run with the REAL signed v8.2 member (initial_checkpoint = cutover) + a trainer
   CURRENT copy from the trainer host.
E. STOP; report exact activation command + rollback + target epoch boundary. No activation,
   no restart, no Discord.

## Activation shape (final command pending C)
Switch is in-code: deploy the chain config (lifecycle config gains release_succession=[v7,v8.2]
and release_policy=v8.2; audit/evaluator use the chain guard; trainer role config gains
release_succession) + repoint source registry to v8.2, then restart the services at an epoch
boundary. The lifecycle/trainer auto-adopt the CURRENT (df2d2b80 -> v8.2) because the cursor
sits at the active initial checkpoint. Rollback = restore prior config (no release_succession,
release_policy=v7) + prior source registry + restart; CURRENT re-stamps back only if a v7->..->v7
chain existed (it won't), so rollback is: keep the pre-switch CURRENT backup and restore it.
BACK UP CURRENT.json (all services) before the switch.

Notes: pytest was installed into /home/const/subnet120-rewrite/.venv (dev dep). Dry-run copy
left at .../20261015-mimo-v8.2-release-succession/dry-run-copy (~300MB, safe to delete).

## RESUME 3 — 2026-10-15 19:08 CT
- Git HEAD on branch is ba6faa64 (lifecycle re-attest fix) on top of a4742217.
- Step1 pod: affine-v82-qual = golden-comet-e2, id 9dc84f00-abb5-4994-bb76-ba5f4f18cb8c, created 2026-10-15 22:43Z (17:43 CT) by previous v8.2 worker, $3.40/h, cap $20.40, TTL removal 2026-10-16 04:43Z. Reachable. REUSING.
- Trainer job private-mimo-v82-qual-queue-training-1 (job c1777af8, source 8527e17d) still leased; worker pid 2360 in D-state reading back optimizer shards on gocryptfs (post-write verify). serve (pid 2884886) + tunnel (2889292) alive on Arbos port 27492.
- 19:12 CT Step2 finding: live CURRENT re-read = epoch 35, round 224, optimizer_step 35, ckpt bc364541… (policy df2d2b80, has checkpoint_authorization), written 22:52Z. c1c4c5e2 is stale.
- DESIGN BLOCKER on "initial_checkpoint = cutover": trained checkpoints carry no merges.txt / vocab.json and a different config/tokenizer_config. ops/mimo_lifecycle.py authenticated_tokenizer(tokenizer_dir, scope.initial_checkpoint.files) requires TOKENIZER_FILES ⊆ files -> REJECTED ("complete explicit MiMo tokenizer file inventory required"; proven read-only against the live tokenizer dir). Also adapters.initial_checkpoint_keys / NativeGradingPool key off those files. => v8.2 scope keeps the genesis initial_checkpoint 87de8304 (as v7/v8.1); the cutover is handled in-code by ba6faa64 (_adopt_succession re-attests the predecessor handover for the live cursor ckpt). Staging binds the cutover cursor in CUTOVER-BINDING.json + chain first_round instead.
- 19:13 CT STEP2 DONE: stage_v82.py -> staged-v82-cutover/ (CUTOVER-BINDING.json, CURRENT.at-staging 6c18efdb, epoch 35 / round 224 / step 35 / ckpt bc364541). Archive 8ae233ad (source 8527e17d, 1572 files) verified; scope 27f7e123; policy file 8298f5f1; lifecycle file 6a8fc8b7 (max_tasks 256, overlong 7000/8192/0.5, KL 0.02).
- 19:15 CT RUNTIME (fresh v8.2): qual-v82-fresh/runtime/stage_source_v82.py staged source-final-v82 + source-registry.PREPARED-v82.json on trainer/evaluator/verifier-1/2/3: 1572 files verified each, --help flags valid, runtime fingerprint == v8.2 policy, role_launched=false, no services touched. -> SOURCE-STAGING-v82.json, REMOTE-COMMAND-CHECKS-v82.json.
- CPU suites + native bench/reset running in background: qual-v82-fresh/run_cpu.sh (tests-src = git archive ba6faa64; 1557 subnet/ops files byte-identical to archive 8ae233ad).
- 19:18 CT CPU: NATIVE 50/50, TRANSACTION (incl overlong) 89/89 pass on tests-src ba6faa64. Prepared (not yet run) on pod: controls-v82 driver (+REFERENCE-MANIFEST 87082f42) and heldout-v82 fixed32 job (job 0e024e58, run private-mimo-v82-heldout-fixed32-1792109885, run-scoped authority 6ad2ccb3, seed 0600 in qual-v82-fresh/heldout-v82/). Both start after the trainer job frees the GPU.
- 19:22 CT V82-DELTA (succession+first-claim+release-policy+runtime+audit-bridge+mining+host-cache+eval-scheduler+hourly-writer+compact-release) 159 passed + 22 subtests. Trainer job: optimizer verify done, now in publication phase (uploading model shards to private R2).
- 19:23 CT RUNTIME public GET re-verified fresh (qual-v82-fresh/runtime/PUBLIC-GET-VERIFIED-v82.json, lock 57442fdd + toploc 84c8db26, ARTIFACTS bb3e613d unchanged). RUNTIME kind = READY.
- writer-accel-v82/ staged (pin-only: ASSESSMENT 49af94e6; derivations RELEASE pin -> mimo_release_succession.py 5e70f784 because reader release is now a ReleaseChain).
- 19:25 CT sign_v82.py written (5ef2163433f0); --dry-run (throwaway key, temp dir, placeholder quals) PASSED all steps: receipts, policy, chain [v7->v8.2] ReleaseChain-validated, writer-reader (chain; approves 69ae058b+8ae233ad) EvidenceSource OK, composition, host + host-accel, lifecycle ACTIVE w/ release_succession, capture/uid operators, trainer role field; uid launcher --check OK. Real run waits for 7/7 receipts.
- 19:27 CT Wrote qualify_v82_unsigned.py (fresh-only, no carry-over) and dry_run_v82.py (copies only; audit/evaluator/lifecycle-adoption/trainer-adoption). Receipts ready: runtime (1/7). Native bench rc=0; 6746-task reset running. Trainer: uploading model shards (8/10).
- 19:38 CT TRAINER-UPDATE + MODEL-PUBLICATION PASS: job private-mimo-v82-qual-queue-training-1 (c1777af8) complete, report d4b22c25, signed trainer a9e787e5, publisher efe6a41a receipt verified, model_full_get_verified, 87de8304 -> ef8cac20 (step 1), KL 0.02, overlong 7000/8192/0.5 (1/16 past soft start, rewards match equation), no chain txs. ROOT-REVIEW.json written. Controls-v82 started on pod (GPU free).
- 19:39 CT STEP5 DONE: R2 upload private/mimo-cutover-20261013/sources/8ae233ad.../source.tar.gz (bucket affine-verification-mock-20260930, 6418748 B), was absent, full GET read-back sha == 8ae233ad. Record R2-SOURCE-UPLOAD-v82.json.
- 19:40 CT Pod secret wipe #1: shred -u r2.private.env, publisher.private.seed, trainer.private.seed, role.private.json on 9dc84f00. Evidence copied to qualification-v82/private-training-queue-v82/pod-copy/. Killed my old tunnel (2889292); serve had exited (port 27492 closed). Removing pod trainer-state (137G) to give heldout cache headroom.
- 19:43 CT controls attempt 1 FAILED after attempt-0: "checkpoint local state changed: chat_template.jinja" — caused by my rm of trainer-state (removing the KL-ref hardlinks changed st_nlink/ctime of the 87de checkpoint inodes; runtime stat-guard tripped). Not a v8.2 code fault. Kept as controls-v82.attempt-1-hardlink-ctime/. Relaunched controls fresh (pod disk now 129G free).
- 19:55 CT MUTATION-CONTROLS PASS (pod H200, v8.2 source): attempts nonce0 353 tok / nonce1 677 tok (eos); honest 2/2 accepted (rc0), mutations rejected by sampling (rc0); RESULT fresh-v82-honest-and-recomputed-proof-mutations-passed. Evidence (no seeds) in qual-v82-fresh/controls-v82/.
- NATIVE-GRADING PASS: cold/warm bench exact grade-hash match; 6746/6746 task reset, 0 failures, 0 model calls (native/RESULT.json).
- qualify_v82_unsigned: READY 5/7 (model-publication, mutation-controls, native-grading, runtime, trainer-update). Pending: heldout (running on pod since ~19:54 CT, ~50 min), cross-device-sampling (needs a 2nd, different GPU device -> not authorized to rent; see blockers).
- NOTE live CURRENT has advanced to epoch 36 / round 225 / ckpt 1f6db19f (00:26Z); handled in-code by re-attestation.
- 19:56 CT Prepared cross-v82-prep/ (cross_v82_driver.py + RUNBOOK.md) and heldout-v82/review.py. Cross-device option if approved: 1xH100 gentle-comet-28 $2.50/h or 2xH200(NVL) brave-fox-66 $3.85/h, ~30 min (~$1.3-2).

## RESUME 4 — 19:57 CT
- 19:43 CT disturbance identified: MY OWN rm -rf of the pod trainer-state (to free disk) removed the KL-reference hardlinks to the 87de8304 checkpoint files -> st_nlink/ctime of every checkpoint inode changed -> mimo_runtime stat-guard aborted controls attempt 1 ("checkpoint local state changed: chat_template.jinja"). Content unaffected (hardlink removal only).
- Repair: attempt-1 quarantined at pod controls-v82.attempt-1-hardlink-ctime/ (not used as evidence); controls re-run from scratch -> PASS (19:55). Trainer evidence was copied + reviewed BEFORE the rm (unaffected). Native/runtime ran on Arbos (unaffected). Content re-verification of 87de vs CHECKPOINT-DESCRIPTOR running: pod pid 5508, log /root/affine-mimo-v82-qual-20261015/ckverify.log, result CHECKPOINT-REVERIFY-after-1943.json.
- HELDOUT running in background: pod pids runner 5000 / worker 5004 (timeout 10800 s), log /root/affine-mimo-v82-qual-20261015/heldout-v82/base/worker.private.log, exit record heldout-v82/base/EXIT.json; cache copy of 87de done 00:57Z (pod clock). Review: qual-v82-fresh/heldout-v82/review.py.
- 19:58 CT REPAIR VERIFIED: 87de8304 on pod re-hashed 12/12 == descriptor, no extra files, nlink=1 (hardlinks gone). Trainer-host CURRENT copied read-only to dry-run-v82/trainer-CURRENT.copy.json (df2d2b80, step 36, ckpt 1f6db19f, e000035).
- 20:03 CT heldout: model loaded, evaluating (GPU 18.8 GB / 38%). Expect ~50 min (v8: 3005 s).
- 20:08 CT heldout still evaluating (poll).

## RESUME 5 — 20:47 CT
- 20:48 CT heldout still evaluating: pod pids 5000/5004 elapsed 53 min, GPU 20.9 GB / 42%, no EXIT.json yet. Log /root/affine-mimo-v82-qual-20261015/heldout-v82/base/worker.private.log.
- 20:49 CT POD SECRET WIPE #2: shred -u controls-v82/{authority,miner}.private.seed and controls-v82.attempt-1-hardlink-ctime/{authority,miner}.private.seed. Pod scan for *seed*/*.env/*private.json/*mnemonic* now finds no secrets (controls evidence already on Arbos). Heldout authority seed lives only on Arbos (qual-v82-fresh/heldout-v82/authority.private.seed, 0600).
- 20:51 CT Arbos watcher launched: heldout_watch.sh PID 3634572, log heldout_watch.log. Polls pod every 60 s for EXIT.json (max 4 h), then runs heldout-v82/review.py and qualify_v82_unsigned.py. Read-only to pod; does NOT tear down.
- NEXT: once watcher log says DONE -> check ROOT-REVIEW passed + quals 6/7 -> lium rm 9dc84f00 (by ID) -> lium ps confirm -> record cost.
- 20:48 CT HELDOUT finished (EXIT at ~01:48Z pod clock, 3223 s). Watcher review failed only for wrong python (no nacl); re-ran with /home/const/subnet120-rewrite/.venv/bin/python -> ROOT-REVIEW.json (sha 1b177e74) PASSED: rc0, 18 success / 3 wrong / 11 unresolved, 0 errors, report 2c5c35b3 validated, ckpt 87de, identical outcome to v7 baseline. Capture (EXIT/REPORT/OPENING/worker log) in qual-v82-fresh/heldout-v82/capture/.
- 20:49 CT qualify_v82_unsigned: READY 6/7 -> QUALIFICATIONS.v8.2.UNSIGNED.private.json sha d83e086bd66ddfc181309a98d15e558c57058f3bcc11858555d9d3050b519aa4. Pending only cross-device-sampling (2nd GPU rental not approved).
- 20:51 CT POD TORN DOWN: final scan no secrets, no python procs; LIUM_NO_POD_INDEX=1 lium rm 9dc84f00-abb5-4994-bb76-ba5f4f18cb8c --yes -> "removed affine-v82-qual — 3.1h at $3.40/h ≈ $10.55"; lium ps no longer lists it (13 active, none ours). Watcher 3634572 exited (DONE).
- STOP POINT: signing (step 4) + real dry-run (step 6) blocked on cross-device approval. Box ssh key shredded at end of RESUME 5.

## RESUME 6 — 20:55 CT (approved: rent affine- pods; cross-device on gentle-comet-28; sign first_round 228; dry-run; write activate/rollback; delete leftover seeds)
- 20:56 CT lium up gentle-comet-28 --name affine-v82-cross --ttl 2h --budget 8 launched in background on Arbos; output qual-v82-fresh/cross-v82/lium-up.{json,log}.
- 20:52 CT POD RENTED: affine-v82-cross = eager-matrix-0c, id 0317a837-c074-4de9-a828-1a37fe610db9, node gentle-comet-28, 1xH100 PCIe 80GB, $2.50/h, created 01:51:16Z, TTL removal 03:51Z, spend cap $5. ssh root@<QUAL_POD_HOST> -p <QUAL_POD_SSH_PORT> (known_hosts /home/const/.lium/known_hosts/0317a837-...).
- 20:55 CT staged on pod /root/affine-mimo-v82-qual-20261015: source.tar.gz 8ae233ad, RELEASE.json, CHECKPOINT-DESCRIPTOR (v8-qual), requirements.lock 57442fdd, toploc 84c8db26, prepare_remote_v82.py, cross_v82_driver.py, cross-v82/{GENERATION-RESULT,OPENING.signed,IDENTITIES,manifest,attempt-0/1.zip} from controls-v82. No credentials copied.
- 20:56 CT pod background chain run_cross.sh PID 808: prepare (bootstrap + HF 87de fetch/verify) -> cross_v82_driver.py. Logs: prepare.private.log, PROGRESS.jsonl, driver.private.log, chain.log (DONE marker); result cross-v82/RESULT.json.
- 20:58 CT CROSS-DEVICE on H100 PCIe = FAIL (real result, not infra): prepare OK (source 1572 verified, 87de 12/12), control rc0 but RESULT status observed-disagreement: both H200 attempts (nonce0 353 tok, nonce1 677 tok) rejected at index 5 "original cached group sampler mismatch". Evidence: qual-v82-fresh/cross-v82.attempt-h100-disagreement/. Every prior pass (v7 cross-device-v1, v7 fresh-original-cross-host-v6, v8 cross-v8) was on H200 / H200 NVL. v7->v8.2 mimo_runtime diff only ADDS trainer-only score_support/score_candidates; sampler path unchanged.
- 21:00 CT Regression check running on the same H100: v7 final source (deployment-final-source-v7) + v7 controls (fresh-original-controls-v6, same 87de) -> /root/v7check (pod pid 1654, log /root/v7check/v7cross/control.private.log, rc in EXIT.txt).
- 21:03 CT v7-on-H100 REGRESSION CHECK: v7 source + v7 controls on the same H100 PCIe -> observed-disagreement too (nonce0 accepted, nonce1 rejected index 5 "original cached group sampler mismatch"). => H100-vs-H200 sampler disagreement is PRE-EXISTING (live v7 has it), not a v8.2 regression. Evidence cross-v82.attempt-h100-disagreement/v7-on-h100-regression-check/.
- 21:04 CT H100 pod torn down: lium rm 0317a837-c074-4de9-a828-1a37fe610db9 -> "10m at $2.50/h ≈ $0.40"; no secrets were ever on it (scan empty). lium ps: gone.
- 21:05 CT Renting H200 NVL (brave-fox-66, -c 1, $3.85/h, ttl 2h, budget $8) as affine-v82-cross to match the v8 cross-device precedent (cross-v8 = NVIDIA H200 NVL).
- 21:02 CT POD RENTED: affine-v82-cross = eager-fox-b1, id 97c4b233-d821-47b2-8da8-5d638bc1aa8b (node brave-fox-66), 1x NVIDIA H200 NVL 141GB, $3.85/h, created 02:01:29Z, TTL removal 04:01Z. ssh root@<QUAL_POD_HOST> -p <QUAL_POD_SSH_PORT>. Same inputs staged (no credentials). Pod: prepare pid 781 -> waiter run_driver.sh pid 955 -> cross_v82_driver.py; logs /root/affine-mimo-v82-qual-20261015/{PROGRESS.jsonl,prepare.private.log,driver.private.log,chain.log}.
- 21:08 CT activation material built (NOT installed): prepare_activation_v82.py (read-only to hosts/units) -> activation-v82/: per-host owned-worker-v82.json (= live v7 config with source-final-v7->v82 and registry->qualified-v82 only; sha trainer 872489b6, evaluator 7df82104, verifiers d161bf3d), source-registry.qualified-v82.json (v7 69ae ∪ v8.2 8ae2), supervised drop-ins zzzz~v82-owned-worker.conf, lifecycle drop-in zzzz~~~~v82-release-succession.conf, writer drop-in zzzz~~~mimo-v82-population-index.conf, ACTIVATION-PLAN.json, MANIFEST.sha256.
- 21:14 CT Wrote (NOT run) activate_v82.sh + rollback_v82.sh in /home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession. activate: default = read-only preflight (manifests, SIGNING-RESULT, hosts staged, units active, fresh-copy dry-run gate); --execute only when CURRENT.round==228: backups (units tar, CURRENT, evaluator/state, roles.sqlite3, trainer CURRENT + role) -> stop lifecycle -> move in-flight v7 epochs/<idx> (its PLAN.input can never match the adopted cursor) into backup -> stage owned-worker-v82.json+registry (--check) -> merge trainer release_succession -> drain v7 workers -> install drop-ins -> daemon-reload -> start -> verify. rollback: reverse, restores pre-v82 state and moves v8.2-era epochs aside (kept).
- 21:17 CT CROSS-DEVICE PASS on NVIDIA H200 NVL (v8.2 source 8ae233ad, 87de): both H200-generated attempts accepted (nonce0 353 tok, nonce1 677 tok), RESULT all-observed-attempts-accepted, driver rc0. Evidence qual-v82-fresh/cross-v82/.
- 21:18 CT H200 NVL pod wiped+torn down: no secrets were ever copied (scan empty, no procs); lium rm 97c4b233-d821-47b2-8da8-5d638bc1aa8b -> lium estimate "10m at $3.85/h ≈ $0.65" (wall clock 02:01:29Z->~02:18Z ≈ 17 min ≈ $1.10). lium ps: no affine-v82 pods.
- 21:12 CT CORRECTION: box clock; H200 NVL pod wall time ≈10 min (≈$0.65) — lium figure correct. qualify_v82_unsigned: READY 7/7, QUALIFICATIONS.v8.2.UNSIGNED sha ae7818a56658d6cd715ee0ce6d4914dff7062759ac9d6537c99df5d8d482e910. Live round 226 -> first_round 228 reachable. Signing launched (nohup): sign_v82.py --first-round 228, log sign_v82.log.
- 21:13 CT STEP4 SIGNED (on-box authority 3301134b, first_round 228, checks_passed, services_changed false, chain_calls 0) -> release-v82-signed/ (all 0600): QUALIFICATIONS 21fa8f5f, PUBLIC-POLICY 0355640a (=policy_sha256), RELEASE-SUCCESSION v7->v8.2 2dfa5b54 (=chain_sha256), writer-reader 0d9944e6, writer-composition 42d6bf6e, HOST-COMPOSITION feed31fa, HOST-COMPOSITION-accel 3603344a, lifecycle.v8.2.ACTIVE 4128158e, capture-operator 4721c0b1, uid-operator 86a7f80d, TRAINER-ROLE field 9f32d45d, SIGNING-RESULT 4033787c; SIGNED-MANIFEST.sha256 05252ccf. uid-launcher --check OK; hourly-writer-accel --check OK.
- 21:14 CT STEP6 real dry-run launched: dry_run_v82_real.sh (fresh copies of CURRENT/auditor populations/evaluator + trainer CURRENT) -> dry-run-v82/real-<ts>/out, log dry_run_v82_real.log.
- 21:15 CT STEP5(user) SEEDS DELETED (shred -u): qualification-v82/private-training-queue-v82/{trainer,publisher}.private.seed and qual-v82-fresh/heldout-v82/authority.private.seed (throwaway qualification identities; verified distinct from the live authority seed; not referenced by signing/dry-run/activation). No *.private.seed remains under /home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession.
- 21:22 CT real dry-run running (pid 3749452, CPU-bound authenticating 37 populations x2). Fixed find -exec bug in activate_v82.sh/dry_run_v82_real.sh; activate --execute now requires activation-v82/PREFLIGHT-PASS (<3 h) from a preflight run instead of re-running the ~10 min dry-run at the boundary.
- 21:25 CT STEP6 REAL DRY-RUN PASS (dry-run-v82/real-20261016T021341Z/out/DRY-RUN-RESULT-v82.json sha 55fc82d1): audit 37/37 populations chain-authenticated (37/37 rejected by single v8.2 policy, as designed); evaluator state df2d2b80 is chain member, 38/38 openings validate; lifecycle adoption e37/r226/step37/ckpt 49476857: policy df2d2b80->0355640a, max_tasks 32->256, lineage preserved, handover re-attested, gate passes, idempotent; trainer adoption bound to v8.2, lineage preserved (ckpt 49476857, step 37). live_touched false. first_rounds [189, 228].
- 21:26 CT activation-v82 MANIFEST OK, SIGNED-MANIFEST OK. activate_v82.sh / rollback_v82.sh written, NOT run. No live services touched, no chain txs, no Discord. No affine-v82 pods remain.
## STOP — awaiting Jacob: run activate_v82.sh (preflight) before the boundary, then activate_v82.sh --execute when CURRENT.round == 228 (~23:40 CT).

## RESUME 7 — 21:25 CT (Jacob approved activation at boundary; task: replan test on copy -> preflight -> report; NO --execute)
- 21:31 CT STEP1 REPLAN TEST (copy only): replan_test_v82.sh + replan_test_v82.py -> replan-test-v82/20261016T022847Z/. Uses the EXACT move-aside block from activate_v82.sh and the EXACT epoch-restore lines from rollback_v82.sh (sed-extracted), and production _plan (uid-capacity-1000 replacement via launcher.checked(uid-operator.v8.2)). Stand-in for the future round-228 dir = copy of live in-flight epochs/37 (PLAN.input == live CURRENT, OPENING, admission.sqlite + POPULATION = v7 submissions). Attempt 1 without the capacity launcher failed on the >256-registration bound (replan-test-v82.attempt1-no-capacity-launcher/): harness fidelity issue, not a code issue.
  * Both scenarios: moved-aside dir byte-identical (A: 64 files incl. admission.sqlite/POPULATION/captured; B: PLAN only); v8.2 adopts the cursor (policy 0355640a, max_tasks 256, lineage preserved) and plans the epoch cleanly (PLAN written, input == adopted cursor, manifest carries the v8.2 envelope and passes validate_manifest, 309 miners, 1800 s window, reload idempotent); rollback epoch-restore puts the v7 dir back byte-identical, keeps the v8.2 plan under backup/v82-era, CURRENT restored.
  * FAIL (scenario A = realistic): same run_id -> same R2 key mimo/affine-mimo9b-grpo-20261013/epochs/<run>-e<idx>/mimo/opening.json. Once v7 has opened (live boundaries: CURRENT -> ~30 s -> PLAN -> ~50 s -> OPENING + immutable R2 put), v8.2 Coordinator.open hits _immutable IfNoneMatch -> "immutable capture collision" -> lifecycle crash-loop. activate_v82.sh --execute as written fails unless run within ~80 s of the boundary.
  * Scenario B (before v7 opens) is clean.
  * rollback_v82.sh --check was not run against the test backup (it verifies live trainer backups that do not exist for a test); its epoch-restore lines were executed on the copy instead.
- 21:33 CT STOPPED per instruction (failure). Preflight NOT run. Wrote activate_v82.armed-candidate.sh (sha 1ca67134, does NOT replace activate_v82.sh): --arm, started while round==227, polls CURRENT every 1 s, stops the v7 lifecycle the moment round==228 (before PLAN/OPENING), aborts with v7 restarted and nothing changed if coordinator/OPENING.json already exists; otherwise the reviewed execute path. Needs Jacob's approval.
- No v8.2 DISCORD-DRAFT.txt exists (only the v8 first-claim and v8.1 overlong drafts, both effective-epoch TBD, archive placeholder 26ed4ddf).
- Live: epoch 37 / round 226 (CURRENT 01:47Z). ETA 227 ~03:07-03:22Z (22:07-22:22 CT), 228 ~04:27-04:57Z (23:27-23:57 CT).

## RESUME 8 — 21:31 CT (Jacob approved activate_v82.armed-candidate.sh; test arm logic -> preflight -> arm after round 227 goes live, by ~23:20 CT; no --execute by hand)
- 21:33 CT ARM SIMULATION (copies only): arm_sim_v82.sh -> arm-sim-v82/20261016T023310Z/. Runs the EXACT --arm block and move-aside block sed-extracted from activate_v82.armed-candidate.sh under set -euo pipefail, against a copy of live CURRENT set to round 227. systemctl is a recorder; a background "v7" flips CURRENT to round 228.
  * S1 flip, no epoch dir: v7 stop recorded 0.07 s after the flip, no start, rc 0. The move-aside block skips cleanly (rc 0, nothing moved, no EPOCH_MOVED).
  * S2 bail-out (epoch dir with coordinator/OPENING.json = R2 opening published): stop, then start (v7 restarted), ABORT "v7 already opened epoch ...; NOTHING changed", rc 99. Backups/move-aside never reached; the epoch dir is byte-identical to its source.
  * S3 plan-only (v7 planned but did not open): stop only, rc 0. The move-aside moves the dir into backup and records EPOCH_MOVED (clean replan path, as proven in RESUME 7 scenario B).
  * S4 adversarial torn (non-atomic) CURRENT write: the poll ignores the partial JSON and only fires after the full write (stop 0.64 s after completion). Production writes are atomic anyway.
- Half-written state cannot happen: save_manifest (identical in v7 and v8.2) = mkstemp + fsync + os.replace. The roles DB is SQLite. The live lifecycle has no SIGTERM handler, so systemctl stop kills it immediately (past stop/start took <1 s). A kill can leave at most a stray temp file, never a partial CURRENT or PLAN.
- No early R2 writes: v7 Coordinator.open saves OPENING.json locally BEFORE the immutable R2 put, capabilities are presigned URLs only, and _plan makes no bucket writes. So no OPENING.json means no R2 opening. If OPENING.json exists but the put has not happened, the bail-out restarts v7, which re-opens the same signed opening idempotently (412 + identical bytes): the designed crash-recovery path, so v7 stays healthy.
- Timing margin: at the last 3 boundaries CURRENT -> PLAN took ~30 s and PLAN -> OPENING ~50 s. The arm poll is 1 s and stop takes <1 s.
- Linger=yes for const, so a systemd-run --user unit survives session end.
- 21:42 CT STEP2 PREFLIGHT PASS (activate_v82.sh, no args; log preflight-20261016T023154Z.log): manifests OK, SIGNING-RESULT clean, 5 hosts staged, 6 units active, fresh-copy dry-run PASS (/home/const/v82-preflight-73Wo/out/DRY-RUN-RESULT-v82.json sha da140043): audit 37/37, evaluator 38/38, lifecycle df2d2b80->0355640a lineage+handover+gate OK, trainer bound to v8.2 lineage OK. activation-v82/PREFLIGHT-PASS written 02:41:51Z (valid for --arm start until ~00:41 CT). Waiting for round 227 to arm.
- 22:06 CT Round 227 live (CURRENT epoch 38 / round 227 written 03:06:11Z). Pre-arm probe: a systemd-run --user transient unit has XDG/systemctl access, ssh OK to all 5 hosts, manifest OK.
- 22:06:43 CT ARMED: systemd-run --user --unit=affine-v82-arm --collect --working-directory=$D -p StandardOutput=append:$D/arm-v82.log -p StandardError=append:$D/arm-v82.log /bin/bash $D/activate_v82.armed-candidate.sh --arm (script sha 1ca67134). Log $D/arm-v82.log: preflight checks OK, PREFLIGHT-PASS 02:41:51Z accepted, "ARMED: polling CURRENT.json every 1 s for round 228" at 03:06:52Z.
- DISARM (safe only while the last log line is still ARMED, i.e. before "boundary seen"): systemctl --user stop affine-v82-arm.service. If the log already shows "boundary seen", do NOT just stop the unit: let it finish, or use rollback_v82.sh (if backups exist) / systemctl --user start affine-mimo-lifecycle.service (if it stopped before backups).
- ETA round 228: epoch 38 = 30 min collection + ~4 min seal/audit + 41-59 min training (e34-e36) -> ~04:20-04:45Z (23:20-23:45 CT).

## RESUME 9 (00:22 CT): e39 investigation (read-only)
- Findings: e39-window-investigation/FINDINGS.md. Evidence: per-miner-breakdown.json, CAPTURE-REJECTIONS.0452.json / .final.json, journal-e39.jsonl.
- Summary: rejections are a frozen-now artifact of the host capture cache (slow serial passes, ~5x groups); all 17 self-healed; final rejections 0. e39 sealed 05:16:50Z with 323 eligible -> 256 selected; training job 05:19:20Z (expires 07:19:20Z). Recommendation: no rollback; watch training time.
- INCIDENT (mine): at 05:20:41Z an eval of a sed-extracted block of activate_v82.sh (meant to load only rssh()) ran its preflight section with all variables empty and exited at `exit 0`. Effects: created empty /home/const/v82-preflight-ak9y; read-only systemctl is-active; two ssh calls failed (empty port); cp/find from / failed; PREFLIGHT-PASS write to / was denied. Verified unchanged: activation-v82/PREFLIGHT-PASS (02:41:51Z), LAST-ACTIVATION, all drop-ins, all 6 units active. Stray dir left in place (deleting needs approval).

## RESUME 10 (01:03 CT): e39 training failure, read-only diagnosis, NO changes made
- Error: ValueError('original recovery optimizer/objective settings') at deployment-final-source-v82/subnet/mimo_grpo/recovery.py:33. Reproduced read-only on the trainer (/root/repro_e39.py): the e38 published transaction history/update-368630c3... report.config (v7: objective prompt-mean, version reference-grpo-v1, kl 0, max_tasks 32) != v8.2 TrainerConfig.document() (19 fields differ). The worker is cold after activation, so execute() takes the restore branch (current optimizer_step 39) -> restore_transaction_state(config=v8.2) -> raises before load_runtime (GPU 0 MiB). optimizer.py:219 (_restore_local) has the same config-equality check.
- The worker swallows the message (distributed_worker.py:469 writes only error_type). The trainer CURRENT was adopted to v8.2 at 05:21:43Z; attempt 1 failed at 05:21:44Z.
- Not causes: KL ref, overlong keys, 2048 rollouts, tokenizer. max_tasks is irrelevant (lowering it does not help).
- Recovery semantics: queue row failed/expired -> TrainingRecoveryRequired on every tick (mimo_lifecycle_adapters.py:157-177). Expiry (07:19:20Z) only changes the status to expired; the lifecycle stays on e39, CURRENT stays at round 228, and no next epoch opens. The operator path (precedent 20261014-trainer-recovery-v1) is: back up roles.sqlite3, reset the job row, move TRAINING-JOB.json and RECOVERY-REQUIRED.json aside, and the lifecycle re-issues the same job id with a fresh expiry. It only helps once the trainer can pass restore.
- Fix requires signed trainer source changes in 2 places (restore with the journal config, then rebind to v8.2; plus the source_closure lineage re-stamp), a new lifecycle config, the operator chain re-signed, the trainer registry re-staged, and a worker restart. Not small; not done.
- rollback_v82.sh caveat: v8.2 already wrote the immutable R2 e39 opening (04:23Z). Rollback re-plans e39 under v7 and would hit an immutable capture collision (scenario A of RESUME 7). Not run.

## RESUME 11 — v8.2.1 trainer restore fix (2026-10-16 ~01:20 CT)
- Root cause e39 training failure: recovery.py:33 required the recorded optimizer config (v7 doc from e38 txn update-368630c3) == v8.2 TrainerConfig (19 fields differ). Same check optimizer.py:219.
- Branch release/mimo-v8.2.1-trainer-restore off ba6faa64, worktree /home/const/affine-mimo-release-v821, commit 265ec9e9:
  restorable_config(): exact doc match, or same optimizer + valid recorded TrainerConfig + identical Adam settings (lr,beta1,beta2,eps,wd,wd_policy; type+value), else fail closed. Restored optimizer adopts current config. distributed_worker.py:469 records str(error)[:2048].
  tests/test_mimo_grpo_release_restore.py (15). Targeted 85 passed; wide 609 passed + 102 subtests.
- Step test (b) on trainer copy: /workspace/affine-mimo-trainer-prod/v821-test (~136 GB copy, chmod 700), step_test.py with stub bucket (no R2 writes), authentic e39 attempt-3 inputs (256 tasks / 2048 rollouts).
  Result: v7->v8.2 optimizer restore PASSED the config check, model + KL reference loaded, GPU training. (Further results below.)
- Throughput finding: e38 = 5.3 s/rollout backward (mean 4937 tok). e39 = 2048 rollouts -> ~3 h backward + OLD/ref prep + ~27 min save/publish >> job_seconds 7200. Needs a signed job_seconds increase (and likely lower max_tasks for later epochs) before (c)/(d).
- Stopped before (c)/(d) pending a decision on deployment variant (full v8.2.1 cutover vs trainer-only launcher override) — writer-reader single approved inventory per archive key 8ae233ad conflicts with v8.2 audit evidence already produced.

## RESUME 12 — v8.2.1 trainer-only override deployed for e39 (2026-10-16 01:49 CT)
- Variant: trainer-only (full cutover would put two inventories on archive 8ae233ad in writer-reader). Audit/eval/writer/lifecycle source unchanged (v8.2, 42f70582).
- Code: branch release/mimo-v8.2.1-trainer-restore, commits 265ec9e9 (restore fix) + 64b40d81 (signed same-release trainer source succession). Tests: 59 targeted; 493 + 68 subtests wide, all pass.
- v8.2.1 source inventory d0b0475d (773 files; 4 files differ from v8.2). Trees: Arbos v821-trainer-override/deployment-final-source-v821, trainer /workspace/affine-mimo-trainer-prod/source-final-v821.
- Signed (authority 3301134b): SOURCE-SUCCESSION.v821 (42f70582 -> d0b0475d @ ckpt 9f73e214 step 39), operator.v821-trainer.ROOT-ADMITTED (sha 907850d9) for mimo_trainer_source_override_launcher.py: train jobs get the v8.2.1 inventory; dispatch 454dc168 (e39) only gets job_seconds 18000. --check passed; offline render: same job id/docs/config, only source/lifetime/URLs differ.
- Trainer: owned-worker-v821.json (sha 6f9a034f), source-registry.qualified-v821.json (8ae233ad -> source-final-v821), trainer-role.json + source_succession. Backups: trainer backup-v821-20261016T064632Z.
- Cutover script v821-trainer-override/cutover_v821_trainer.sh at 06:48Z. Backups v821-trainer-override/backup-20261016T064805Z: unit drop-ins, roles.sqlite3(+journal), TRAINING-JOB/RECOVERY-REQUIRED .moved. Deleted only the failed e39 train job row.
  Drop-ins: lifecycle 'zzzz~~~~~v821-trainer-source.conf', trainer 'zzzz~~v821-owned-worker.conf'. Revert = remove both + daemon-reload + restart.
- Hiccup: attempt-1 crashed with FileExistsError (old attempt dirs, same deterministic job id). Moved the old workspace dir to the trainer backup dir. Attempt 2 claimed ~06:54Z; trainer CURRENT re-stamped to d0b0475d by the signed succession; GPU prep running.
- Step test result (before kill): 269 rollouts, 3.03 s/rollout, finite losses, drift 0, KL ~0.06. v821-test killed and deleted.
- Writer admission check on a copy (pre-reset backup): 20/20 completed verify jobs (last 6 h; epochs 1-27, v7 archive) admitted, 0 refused. No e39 verify jobs exist yet (audit backlog), so nothing produced for e39 could be invalidated.
- Next: successor release v8.2.1 (max_tasks 80, job_seconds 7200), prepare only.

## RESUME 13 (in progress) - v8.2.1 successor prep
- staged-v821-successor via stage_v821.py: archive 1aa805b5, scope 32526ff1, lifecycle max_tasks 80 / job_seconds 7200.
- sign_v821.py dry-run (throwaway key) PASS; real signing PASS: release-v821-signed, policy bfec0942, chain f4b01f07 [v7 189, v8.2 228, v8.2.1 229]; uid-launcher + hourly-writer --check OK. Not activated.

### RESUME 13 (cont., 07:52Z / 02:52 CT): successor STAGED (not activated) + new blocker
- (a) v821-test: GPU run killed and /workspace/affine-mimo-trainer-prod/v821-test deleted (re-verified 07:44Z: path absent, no step_test process; trainer disk 423G/1.3T used).
- Dry run on a copy (v821-dryrun-copy-20261016T073116Z/out/DRY-RUN-RESULT-v821.json): PASSED. audit 40/40, evaluator 40/40,
  lifecycle 0355640a->bfec0942 at epoch 39/round 228, step 39, checkpoint 9f73e214, max_tasks 256->80, lineage preserved, handover
  re-attested, CURRENT gate passes, rerun idempotent; trainer adopts bfec0942 (checkpoint 9f73e214, step 39).
- Scripts (NOT run): activate_v821.sh (no args = preflight; --arm [--allow-move-unopened-plan]), rollback_v821.sh [--check],
  upload_v821_source.py, merge_trainer_role_v821.py. bash -n OK. activate_v821.sh.bak-preguard = pre-guard copy.
- Preflight attempt 07:43Z aborted (nothing changed): affine-mimo-verifier-1-supervised is not active (crash-loop, NRestarts 73).
- NEW BLOCKER (pre-existing v8.2 defect, since ~04:21Z activation): audit/eval jobs for v7-archive epochs (source_bundle 69ae058b)
  carry the v8.2 inventory (8ae233ad tree, 773 files) -> workers: "job runtime source inventory differs from ROOT registry"
  (registry entry 69ae058b = 770 files). Audits 0d3818f4/0e12d98c/42758a4f/d203abfb/d7538568/8a808fc8 and evals 49100d8f/f86e8ee2
  fail at attempt 3 repeatedly; verifier-1 worker log has 44 such tracebacks; the verifier-1 watcher crash-loops on the owned-worker lock.
  The same defect would give e39 audits (archive 8ae233ad) the v8.2.1 inventory after activation -> activate_v821.sh now refuses to
  run until activation-v821/BLOCKER-AUDIT-SOURCE-FILES.resolved exists. Nothing changed for this (read-only diagnosis).
- e39 (07:44Z): job mimo-train-454dc168 attempt 2 leased; 278/2048 rollouts at 3.12 s each (rollouts started 07:29:52Z); GPU 77%, 141 GB;
  losses finite. Expected rollouts done ~09:16Z, job done ~09:45Z (04:45 CT); lease deadline 11:49Z (06:49 CT).

### RESUME 14 (08:00Z / 03:00 CT): audit/eval per-archive inventory fix folded into v8.2.1
- Fix: affine-mimo-release-v821 commit 547573dd: ops/mimo_lifecycle.py archive_execution_resolver: audit/eval job metadata =
  manifest -> signed ReleaseChain member -> member archive -> ROOT-approved inventory (config `archive_execution`, == writer
  approved_execution); every chain archive required at startup; active archive == live closure. tests/test_mimo_archive_execution.py
  12 passed (v7-era, v8.2-era, mixed, refusals). Related suites: one pre-existing unrelated failure (test_audit_queue_snapshot,
  ContinuousAuditor.historical_report_admission; untouched file).
- Old never-activated v8.2.1 dirs moved to superseded-v821a-20261016T075410Z/ (and v821b-pre-retries/ inside it).
- New tree v821b-tree = override tree + fixed ops/mimo_lifecycle.py (worker inventory unchanged d0b0475d; live override untouched).
- Restaged + ROOT-signed: archive e6bba768 (1572 files), scope b75d786a, policy 4f017106, chain b7fc211f (first_rounds 189/228/229);
  lifecycle: max_tasks 80, job_seconds 7200, archive_execution {69ae058b, 8ae233ad, e6bba768}, audit max_execution_retries 3->6
  (sanctioned re-dispatch of draws exhausted by the defect, same selected rows). Launcher + writer --check OK. Logs sign_v821c.log.
- Hosts: new inert source-final-v821b (tree-ok 1572) on all 5; PREPARED-v821s registry replaced (old kept as .superseded-*).
  config_v821s: trainer f25699df, evaluator 418bab22, verifiers b51707bc.
- activate_v821.sh patched (.bak-v821a): new policy/archive pins, plan-based source dir check, e39 opening copied for the preflight
  dry run, worker watchers may be active|activating (verifier-1 loop). upload_v821_source.py pinned to e6bba768.
- Impact: since 04:21Z 9 audit draws selected; 5 exhausted (gen 3 failed; epochs 5, 9, 18x2, 30; 5 miners), 3 leased (will fail),
  1 complete. Evaluator: 3 checkpoints exhausted. Infra failures add no observation (not a miner penalty).
- 08:20Z dry run (v821b-dryrun-copy-20261016T075622Z/out) PASSED incl. new JOBS section: audit jobs 39 (v7 69ae058b, 770 files)
  + 1 (v8.2 8ae233ad, 773) accepted by real distributed_worker.source_for_job on verifier-1/2/3 prepared registries and equal to
  writer approved_execution; eval jobs 39 + 1 accepted (evaluator); e39 opening (8ae233ad, member 0355640a) accepted by all 3
  verifiers + evaluator + writer; counterfactual (old live-inventory) 40/40 refused. Lifecycle -> 4f017106 (max_tasks 80), trainer
  adopts 4f017106 at step 39. (first attempt failed in the dry-run harness only: placeholder URL; fixed to local presign.)
- Stale trainer-role source_succession (e39) is harmless after adoption: _adopt_source_succession returns before reading it.
- Weights: hourly writer submitted 05:14Z, 06:07Z, 07:07Z (normal). Failed audits = no observations, no penalties.
- 08:22Z activation-v821/BLOCKER-AUDIT-SOURCE-FILES.resolved created; preflight started (preflight-v821b.out).
- 08:32:52Z preflight PASS (v821-preflight-lolE/out). 08:34:38Z ARMED: transient user unit mimo-v821-arm.service running
  activate_v821.sh --arm --allow-move-unopened-plan (log arm-v821.out); polls CURRENT every 1 s for round 229. On the boundary it
  stops the lifecycle; aborts + restarts v8.2 if epoch 40 is OPENED; moves an unopened epoch-40 PLAN into the backup (approved).
  Rollback: rollback_v821.sh (backup dir in activation-v821/LAST-ACTIVATION). Disarm before the boundary: systemctl --user stop mimo-v821-arm.service.

### RESUME 15 (09:45Z / 04:45 CT): v8.2.1 ACTIVATED (script aborted on a false alarm at the last unit check)
- Armed run: boundary 09:29:19Z (e39 finalized, round 229); backups activation-v821-backup-20261016T092919Z (no EPOCH_MOVED: e40 had
  no plan yet); R2 source e6bba768 uploaded (key was absent) + read-back verified; hosts staged + --check OK; trainer role merged;
  workers drained; drop-ins installed; units started 09:30:28Z. 09:31:45Z ABORT "evaluator supervised unit not active": the
  evaluator watcher restarted 17x ("Existing lock ... startup incomplete") while its slow worker started, attached 09:33:05Z,
  stable since. No rollback (not needed).
- verify_v821.sh (new, READ-ONLY; .bak1 = first version with a unit-state match bug) => ALL-OK: 5 units active running, v821s
  drop-ins identical + sorting last + effective ExecStart pins v821s config; host config hashes trainer f25699df, evaluator 418bab22,
  verifiers b51707bc; registry hashes trainer 1a653e29, evaluator 379f9c26, verifiers 812a18ac; all 5 workers alive on v821s;
  lifecycle active on uid-operator.v8.2.1, override drop-in retired, writer v821 drop-in, writer --check OK;
  CURRENT round 229 / e40 / 4f017106 / max_tasks 80 / ckpt 571a1932.
- activate_v821.sh post-start check staged fix (.bak-v821b): per role, up to 5 min, accept active|activating while the v8.2.1 worker
  reports alive.
- Production: new eval job mimo-eval-8fcd0c25 (e40 checkpoint, archive e6bba768, inventory d0b0475d) built under v8.2.1 and leased by
  the evaluator. eval 861e (built 09:22Z under v8.2, wrong inventory) will fail once; scheduler renews with fixed metadata.
- Audit scheduler: first cycle after restart re-admits + re-authenticates 40 populations (CPU-bound, no errors); 9 draws at gen 3 /
  failed await renewal (cap 6).
- 09:55:58Z audit scheduler renewed the 9 exhausted draws to gen 4 (cap 6); 3 enqueued with the v7 770-file inventory, leased by
  verifier-1/2/3, then FAILED (3 attempts each) with a DIFFERENT error: "MiMo worker exact installed source closure differs"
  (mimo_worker.source_guard). MiMo verify/evaluate jobs run in-process in the warm worker; source_guard requires job.source_files
  == the worker's own running closure, while source_for_job requires == registry[archive]. Both can only pass on a worker running
  that archive's code. So v7-epoch audits have been impossible on any worker since v8.2 (04:21Z) whatever inventory they carry, and
  e39 (8ae233ad) audits are impossible on v8.2.1 workers. e40+ (archive e6bba768, closure d0b0475d) pass both. The dry run only
  exercised source_for_job, not source_guard; that is the gap. Fix 547573dd is correct but not sufficient for old archives.
- Exposure: populations hold 7640 v7 records (498 drawn) + 383 e39 records (0 drawn); random selection will mostly pick
  un-auditable rows, starving e40+ coverage (max_inflight 3; tick ~10+ min because every tick re-authenticates 40 populations).
- No state corruption; no rollback (v8.2 workers cannot audit v7 rows either). Nothing changed; options reported to the user.
- e40: collection deadline 10:00:36Z; captures finished 10:25:15Z (90 listed, 0 rejected); native grading pool (8 workers) started 10:25:17Z for the seal.
- e40 SEALED 10:35:54Z (AUDIT-POPULATION: 487 records, 426 eligible vs max_tasks 80, 90 miners). TRAINING-JOB 10:37:05Z: mimo-train-0e279e6b, 7200 s lifetime (expires 12:37:05Z), max_tasks 80 x group 8, source d0b0475d; leased attempt 1 by the trainer 10:37Z; GPU idle at 10:37:54Z (model load; e39 needed ~40 min to first rollout).
- e40 TRAINING on GPU: claim 10:37Z, model load ~18 min, first rollout 10:54:53Z; at 11:00Z 108 rollouts @ 2.94 s, losses finite (max 0.0014), KL ~0.06, drift 0, GPU 86% / 132 GB. 80 tasks x 8 = 640 rollouts -> ~11:26Z; job done ~11:50Z (06:50 CT) vs expiry 12:37:05Z (07:37 CT).

## RESUME 16 — v8.2.2 (archive-scoped audits, retries 3, max_tasks 128) — 2026-10-16 ~11:15Z
- Code: commit 4d20dc70 (worktree affine-mimo-release-v821): ops/mimo_lifecycle.py archive_scoped_audit_scheduler; resolver.auditable_archives = archives whose
  inventory == running closure. Unauditable epochs' draws parked (never renewed), reported to <auditor>/UNAUDITABLE-BY-DESIGN.json. tests 18/18 pass.
- Staged: staged-v822-successor/ (stage_v822.py 128). Archive A3=23207f85, closure d0b0475d (unchanged -> e40 stays auditable). Only changed file ops/mimo_lifecycle.py.
  Lifecycle delta exactly: source_bundle_key, retries 6->3, lifecycle/trainer max_tasks 80->128; archive_execution +A3.
- Signed: release-v822-signed/ (sign_v822.py --first-round 231): policy 06eac7c5, chain 7dbe6623, checks passed. Target boundary ROUND 231 (230 too tight).
- max_tasks 128 check: e40 3.13 s/rollout overall -> 1024*3.13=53 min + ~18 prep + ~19 publish = ~90/120 min, ~30 min margin (>=20). Confirm with e40 publish.
- VERIFIER-1 LOOP ROOT CAUSE: all 3 verifier workers crash in distributed_worker._mimo_claim -> mimo_worker.source_guard on renewed v7 audit jobs
  (e7/e9/e18, archive 69ae058b); exception escapes once() -> process exits -> watcher restart -> "lock ... startup incomplete" while reloading.
  Not state-corrupting (failed audits only). Fix = v8.2.2 (no renewals of unauditable draws). Remaining queued v7 jobs exhaust their 3 attempts then die.
- NEXT: hosts staging (registries qualified-v822s +A3 -> source-final-v821b), dry_run_v822 (+source_guard, scheduler ticks, never-drawn proof), activate_v822.sh, preflight, arm 231.
- 11:12Z REBUILD (bug found in 4d20dc70 before activation: base completed_queue_rows() used the parking _load -> completed v7/e39
  audit rows would have been hidden from that reader; no live caller, on-disk state always full). Fix a7bc7f77: park only inside tick(). 19/19 tests.
  First build superseded -> superseded-v822a-20261016T111203Z (hosts keep inert source-final-v822 + PREPARED-v822s.superseded-*).
- v8.2.2 FINAL: archive A3=a2a03cb3, policy dfe0cbf7 (release-v822-signed, first_round 231, checks passed), SIGNED-MANIFEST (1586 files).
  Hosts: source-final-v822b + PREPARED-v822s on all 5 (tree-ok 1572). activation-v822/ (18 files).
  Scripts: dry_run_v822.py (+SOURCE_GUARD in-process + real hosts on v822b/v821b trees; +SCHEDULER real ticks on copies),
  activate_v822.sh (TARGET/NEWPOL/MAXT from signed result; arm round 230/e41 -> boundary 231/e42; post-start tolerant check),
  rollback_v822.sh (removes v822 drop-ins; v821s ones beneath stay), upload_v822_source.py.
- 11:15Z preflight running (preflight-v822.out). e40 429/640 rollouts @2.96 s.
- 11:58Z PREFLIGHT PASS (v822-preflight-hJJr; harness fixes only: pop enabled/directory, snapshot insert columns; release unchanged,
  SIGNED-MANIFEST + activation MANIFEST verified, lifecycle == a7bc7f77). source_guard: in-process + 4 real hosts (v822b & v821b trees)
  pass only e6bba768/a2a03cb3, fail v7 69ae058b + e39 8ae233ad. Scheduler: real scoped ticks drew/enqueued only e40; 498 parked v7/e39
  draws unchanged, 0 renewed; 9 jobs pass source_for_job + source_guard; 486 completed rows preserved; report 40 epochs / 8023 records.
- 12:00:04Z ARMED: transient unit mimo-v822-arm (arm_wait_v822.sh -> activate_v822.sh --arm, no --allow-move-unopened-plan), log arm-v822.out.
  Live round 230 / e41 / 4f017106 / 80 / ckpt 3b69e30a (step 41). Fires at round 231 (~14:30Z est).
  Abort: systemctl --user stop mimo-v822-arm (before boundary). Rollback after activation: rollback_v822.sh.
- e40 RESULT: 466 correct / 17 wrong / 157 unresolved of 640 (72.8% pass), preclip grad norm 0.0825, loss 0.01375, step 40->41.
  Timeline: claim 10:37:05, first rollout 10:54:53, last 11:26:01 (2.92 s/rollout), report 11:52:50 (75.8 min total; post-rollout ~27 min).
  128 tasks est: 18 + 1024*2.92s(50) + 27 = ~95 min of 120 -> ~25 min margin (>=20) -> kept 128.
- verifiers still crash on renewed v7 audits (v1 133 restarts, v2/v3 18); stops being fed at v8.2.2; leftover queued v7 jobs die after 3 attempts.

## RESUME 17 — "weights not setting" diagnosis (2026-10-16 ~14:30Z / 09:30 CT). Read-only. NO tx, NO restart, NO rollback, NO manual set.
- "const" validator = UID 0 = owner hotkey 5HmYnmUY...Qzb8 (identity name "const"). It is the only permitted validator (max_validators 1) and also the hotkey the writer signs with (chain.py OWNER).
- Commits land: CR v4 timelocked, reveal period 1, rate limit 100, version_key 0. LastUpdate[0] moves every hour (~:07): 9295552, 9295816, 9296115, 9296416 (14:07Z / 09:07 CT, which matches the writer "submitted" log for window 14:00Z). One pending TimelockedWeightCommits entry (block 9296416). No version, rate-limit or permit rejections.
- BUT the weight CONTENT has been frozen since about 2026-10-14 17:00 CT: on-chain Weights[120][0] hash cbb27279f0 (270 nonzero, top uid 251 at 1.7%), unchanged at every sample from 10-14 18:57 CT to 10-16 09:21 CT.
- Root cause: since assessment hour 1792018800 (10-14 18:00 CT), every hourly assessment has assessment_stale=true, evidence_error=ValueError, evidence_cutoff=1792015200 (10-14 17:00 CT). The writer silently falls back (best_fallback) to the 17:00 CT assessment and re-commits the same vector every hour.
  ValueError reproduced read-only (cutoff 1792159200): current_assessment_compact_queue.scan -> "bounded distinct original job identifiers". controller-state/continuous-audit/audit-state.json has 21994 jobs > the compact queue POLICY maximum_rows=20000 (ROOT-admitted constant; scan requires policy == POLICY).
  Unrelated to v8.2.2/e41 (started about 33 h earlier). FRESHNESS monitor has said "assessment-evidence-stale" (unit failed) the whole time.
- Impact: 36/36 UIDs registered since the freeze get 0 weight; 69 assessed hotkeys are excluded as unregistered; MiMo e39-e41 results do not reach weights. 273 UIDs still get emission from the frozen vector.
- A manual set_weights was NOT done: it would recommit the same stale vector the writer already commits each hour, and the only validator key is the owner hotkey.
- Fix (needs root re-sign, not applied): raise the compact-queue maximum_rows (e.g. 20000 -> 50000, checking the time/memory budget) or bound identifiers to the history window (drop jobs whose populations are older than history_hours) in a new signed writer composition. Then check that --check gives a fresh assessment (assessment_stale=false) and let the next hourly window commit. Also make stale fallback older than about 2 h alert loudly.

## RESUME 17b — writer fix window-v7 INSTALLED (2026-10-16 15:51Z / 10:51 CT), authorized by Jacob. Signed with on-box authority 3301134b. SUBNETOWNER mnemonic not used.
- New dir: state/root-audits/20261016-writer-audit-window-v1/ (backup/, ROLLBACK.md, sign_host_v7.py, dryrun_v7.py).
- HOST-COMPOSITION.v8.2.2-accel-window-v7.ROOT-SIGNED (sha 8d9451d7) = v8.2.2-accel payload + entry v7 + window_module + evidence_function_source_sha256. Parent writer composition, writer policy (e43e63db), wallet, cursor and lock are unchanged.
- Fix 1, audit_history_window.py: the copied legacy _load_evidence_uncached drops populations with round <= floor and jobs whose draw rows are all <= floor, where floor = (lowest round of populations committed in the 168 h EMA window) - recent_epochs(8). Such rows cannot reach points or weights (EMA window, recent_epochs audit window, per-epoch uniqueness). At 16:00Z: pops 166->99, jobs 22068->10374 (< scan maximum_rows 20000).
- Fix 2, a latent v8.2 bug the scan crash had masked: the MiMo EvidenceSource reducer (historical indexed snapshot) has no claim_order, so v8.2+ populations (first-claim) raise TypeError("unexpected keyword argument 'claim_order'"). v7 routes only claim_order calls to the signed v8.2.2 continuous_audit_policy.snapshot; all other calls are unchanged.
- maximum_rows 50000 NOT done: it lives in the compact-queue module plus the parent signed writer policy (a much wider re-sign), and windowing alone gives about 2x headroom.
- Fix 3, the admission cache was full (10000/10000 since 10-13 14:23Z; 3133 entries unreferenced by any in-window job), so every newer job was a slow full re-admission. Moved those 3133 entries to backup/admission-cache-dead-entries (reversible). A dry run re-warmed the cache to 10000.
- Drop-in installed: ~/.config/systemd/user/affine-live-reward-writer.service.d/zzzz~~~~~~mimo-v822-window-v7.conf, then daemon-reload. No service restarts. Trainer, lifecycle and v8.2.2 not touched.
- RESULT 16:00Z window (11:00 CT): assessment-1792166400 assessment_stale=false, evidence_cutoff 16:00Z, 142 epochs, 319 positive miners; writer "submitted" 16:07:27Z. 282 UIDs, max share 2.7% (uid 147), UID 0/burn not weighted, 37 unregistered hotkeys excluded, 7 of the 38 UIDs registered since the freeze now weighted; L1 distance vs frozen cbb27279f0 = 0.94.
- ON CHAIN (a): block 9297017 (11:07:00 CT), extrinsic 9297017-12 SubtensorModule.TimelockedWeightsCommitted(owner 5HmYnm..., netuid 120, 0xfac9c9d4..., reveal_round 33122484) + System.ExtrinsicSuccess. LastUpdate[0]=9297017.
- Freshness monitor turned "healthy" at 16:07Z (it already flags assessment-evidence-stale). New read-only check: ~/subnet120-rewrite/tools/verify_weights_fresh.sh (exit 1 if last_update > 75 min old or the newest assessment is stale; exit 2 on error).
- Rollback: see 20261016-writer-audit-window-v1/ROLLBACK.md (rm the drop-in + daemon-reload; optional cache restore).
