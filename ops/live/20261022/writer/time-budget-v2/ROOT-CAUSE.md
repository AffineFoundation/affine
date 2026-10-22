# Why "original queue job not completed" exclusions grow (2026-10-20, read-only findings)
Queue: live-controller-v1/controller-state/roles/verifier-queue.sqlite3 (65 GB), role=verify, continuous-audit-group-* jobs (~3.6 MB envelopes).
- Through 10-20 ~06 UTC: ~30-35 jobs/h, all complete (+~2/h expired).
- From 06-07 UTC failures start; since 11 UTC (05:00 CR): 0 completions, ~107-118 failed/h. Every job is claimed and
  worker_failed within ~3-5 s on all 3 attempts by both remaining claimers (workers 56bc3046..., ec76838f...). Last complete: expires 10:26 UTC.
- Queue totals at 17:45 UTC: complete 16240, failed 978, expired 347, leased 1.
Why the writer re-checks them each hour: the evidence loader (20261012-writer-miner-observation-index-v1 current_assessment_evidence.py)
keeps every audit-state job in the 8-epoch window (jobs_kept 7886 of 25841 at 17:00 UTC) and compact_queue.scan reads+hashes each job's full
envelope before the status check excludes it. failed/expired are terminal but are never retired from audit-state.json (lifecycle-owned, 1 GB),
so each hour adds ~110 x 3.6 MB of dead rows to the scan -> ~+20 s/h evidence time.
Source fix = verifier side (why every verify claim fails since ~06 UTC). Not touched: remote worker hosts, lifecycle unchanged.
Writer-side hygiene (NOT deployed): pre-filter identifiers whose queue status is terminal failed/expired with a cheap status-only SELECT
before scan(). Same candidates/weights; only the queue-binding digest set changes. Needs dry-run equality check + new signed composition.
