# Prospective bounded audit groups

This option is default off. It changes scheduling, not model sampling, proof checks, miner commitments, per-child outcomes, or training eligibility. No production activation is asserted.

An explicit ROOT-signed service admission may bind `job_grouping_policy` to `{"version":"bounded-checkpoint-audit-groups-v1","max_submissions":4}` (bounded integer 2–4). Config and signed admission must match exactly. The service persists original unpredictable draws and group plans before proof capture. Each group stays within one immutable opening, which also fixes checkpoint, source, sampler, environment and numerical cohort. Each child retains its miner signature, commitment digest, batch/proof SHA, slot and frozen object capability. Different miners and multiple slots from one miner are supported.

A worker loads one model and checks every group's child using the existing multiple-submission execution path. The existing worker releases its checkpoint only after the entire group's authentic report is acknowledged; there is no disk retention between completed groups. The job expiry remains bounded; increasing its lifetime is not part of this change.

Original issued group envelopes survive retries without new draws, changed timestamps or repeat computation. Infrastructure capture failures remain retryable original draws; confirmed artifact failures retain their separate typed evidence. Successful children are not recaptured in those retries. Scoring joins each original signed child independently and deduplicates repeated queue evidence, so grouping cannot multiply rewards or penalties.

Measured precursor evidence: the latest 40 authenticated completed jobs had median claim-to-ACK 264.22 seconds, checkpoint materialization/authentication 186.81 seconds, model construction 18.47 seconds, and proof download/authentication 8.59 seconds. Remaining time also includes grading, native proofs and report transport; it is not an isolated prefill measurement. These observations justify amortizing model startup, but do not establish a grouped production speedup. Measure genuine group completion before changing throughput claims.
