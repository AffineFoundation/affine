# Preserving the original epoch 157 training update

Epoch 157's training job was prepared at 04:50:16 UTC with a 05:50:16
completion deadline. A coordinator preflight error delayed its actual dispatch
until 05:27:28. The original worker started before the signed deadline and
continued executing, but a valid report completed after 05:50:16 would fail the
coordinator's completion-time check. Restarting the worker would repeat the
training experiment and is not the recovery.

The operator recovery authorizes only that original job to complete
before 06:27:28 UTC, one hour after its observed start. It preserves the original
signed job, deadline, report timestamp, model, optimizer lineage and worker.
Its separate signed scope binds the complete job payload, dispatch journal,
manifest, checkpoint, source/runtime, original file hash and observed host and
process identities. The local observer resumes the existing remote handle;
failed or missing original jobs remain ineligible for automatic relaunch.

The recovery was installed proactively at 05:49:42 UTC. The original job then
completed at 05:49:58.753, **17.736 seconds before its original deadline**, and its
worker exited successfully. Consequently the original ordinary time predicate
accepts this report; the additional completion exception was **not exercised**.
At this observation, checkpoint publication and optimizer acknowledgment remain
pending. Training completion alone is not a finalized epoch.

The report validator retains its exact code and all non-time checks. Its existing
time predicate, including the earlier epoch 136 exception, remains in place;
only the independently authorized epoch 157 exception is added. Reports for
other jobs receive no new exception. This does not change the mining contract,
the prospective epoch 158–160 task-priority trial, hourly reward assessment, or
the website design. Epoch 157 has still missed the one-hour operational target;
the recovery does not conceal or reset that elapsed time.

`ops/original_job_completion.py` provides the portable predicate and function
wrapping primitives for this design. Their caller must authenticate the signed
operator scope and observed timely start before use. Deployment uses a separately
pinned operator wrapper with the exact original job and observation bindings;
private deployment records and credentials are not included in the public repo.

CPU controls cover strict time bounds, changed identities and optimizer inputs,
preservation of source and optimizer checks, ordinary historical behavior, and
the fact that extending the new function leaves the original function unchanged.
Operator-specific controls additionally exercise the actual existing-job resume
path and public discovery selector. Deployment and successful numerical completion
must be established from their own receipts; these controls alone prove neither.
