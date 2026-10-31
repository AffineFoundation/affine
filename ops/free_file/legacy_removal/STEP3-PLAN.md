# Step 3 (NOT done 2026-10-31): legacy code removal from the live closure — staged plan

Why it was left in place on 2026-10-31 (see PROGRESS.md / DECISIONS.tsv L06-L08):
1. The free-file lifecycle still runs ON the legacy machinery every epoch. ff_launcher.py = L (v829x launcher, admitted sha)
   + ff_lifecycle. Each epoch still: signs an opening (Coordinator.open, wrapped by v829u dual-accept), pre-opens (v829v),
   collects (v829n capture), seals (red-skip early seal, AUDIT-POPULATION / coordinator POPULATION), runs v823 superseded
   cleanup, and only then does ff_lifecycle pick the turn. mining.json discovery is bound to that signed opening
   (DISCOVERY.json, OPENING.json). Removing "seals / hashes / upload windows" from the source therefore is a redesign of
   the epoch lifecycle (open -> seal -> select -> train -> gate -> finalize), not a deletion.
2. Workers (trainer, evaluator, verifier-h200-1/2) hash subnet/*.py (source_files). Any source change = new source archive
   + ROOT-signed PUBLIC-POLICY / QUALIFICATIONS / RELEASE-SUCCESSION / TRAINER-ROLE-RELEASE-SUCCESSION + CURRENT re-attest
   at a boundary + worker drop-ins on all pods (the v8.2.3u release: RA/20261029-v829u-125task, ~1 day incl. sim, dry run,
   qual, one abort). On 2026-10-31 Allan's trainer v8.3.3i release (RA/20261031-renorm-step-live) switched the trainer at
   the e353 gap; a source release changes the same trainer role succession and must not overlap it.
3. The flags already make the legacy paths inert (legacy_off.sh, live from e353/e354): no capture, no pregrade/regrade,
   no seal prefetch, no capture PUT fan-out, no extended (upload-grant) identities.

Proposed order (each its own decision gap, own backup + revert + watchdog):
A. Host-only (no source release): ff_launcher.py stops installing L's v829n (capture_put/pregrade), v829k (seal prefetch),
   v823_lifetime (superseded cleanup). Edit ff_launcher.py -> re-pin release/FF-PINS.json in the SAME step -> activate_ff.sh
   MODE=restart (idle trainer, both locks). Revert: previous ff_launcher.py + FF-PINS.json + MODE=restart.
   v829u dual-accept stays until (B): DA.policy is in the signed manifest; without the host wrapper open() changes.
B. Source release v8.2.4. Base: the deployed source is branch release/mimo-v8.2.3u-dual-accept (a658eb80, archive fb870257),
   NOT GitHub main (main has no subnet/dual_accept.py or batch_lifetime.py); main's client removal must be ported onto it.
   TOPLOC is referenced by ~25 subnet modules (controller, backend_jobs, mimo_worker, mimo_runtime, complete_group_*, ...),
   so it is a refactor with its own tests. Drop subnet/dual_accept.py, batch_lifetime.py,
   complete_group_capture/transport/mining/sampling (old miner upload path), miner.py, mimo_parallel_mining.py,
   mimo_audit_bridge.py (TOPLOC audit bridge), regrade/pregrade helpers; manifest without max_batches /
   required_client_features / sampling_contract / dual_accept_policy; keep the signed opening only as the epoch clock
   (or replace it). Must also fix subnet/source_bootstrap.py (admitted_files requires subnet/cli.py, which main no longer
   has) and subnet/multi_environment.py (launches subnet.cli). Follow build_v823u.py / stage_v823u_policy.py /
   sign_v829x.py / activate_v829x.sh (MODE=workers then boundary) with a sim + dry run first.
C. Then mining.json: drop start/deadline/status/accepting_submissions/nonce_* together with the site Mining tile
   (site-metrics.js, dashboard/site_metrics.py read them).
