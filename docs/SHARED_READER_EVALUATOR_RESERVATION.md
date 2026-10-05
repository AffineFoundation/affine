# Optional shared evaluator / independent-reader scheduling

The initial hourly deployment should prefer its separately qualified CPU reader.
This prospective option lets the existing evaluator machine also serve its
unchanged qualified four-stream independent state reader. It does not create a
second evaluator, change reader modules, relax any of the 23 object hashes, or
make a long evaluation fit the hourly budget.

Both services must opt in with exactly the same root-local configuration:

```json
{
  "version": "shared-reader-evaluator-reservation-v1",
  "directory": "/absolute/operator-private/shared-node-reservations",
  "provider_UUID": "c1a01718-a9a1-4726-92fe-a568caf561dd"
}
```

Set this under `shared_node_reservation` in the evaluator configuration, and
under `independent_state_reader.shared_node_reservation` in the controller
configuration. The directory must be private to the operator (0700); each
journal/lock is 0600. The evaluator pins the UUID to its known-hosts filename;
the reader pins it to its existing qualified physical-host record. No seed,
credential, capability URL, or checkpoint data is written to this journal.
Without this option, current behavior is unchanged. Both dispatchers must be
configured together before use; the option cannot coordinate unconfigured
legacy dispatchers or manual GPU launches.

After the original trainer report and staged descriptor validate, the reader
records durable priority intent. An existing evaluation keeps its reservation
until its original remote runner records actual `child.wait()` completion.
New evaluations wait behind the reader. No process is killed or restarted to
preempt it. Atomic locked journal updates and fsync preserve the original
request binding across operator crashes; an additional per-request lock stops
two observers from dispatching the same new request concurrently.

Reservations retain the original signed-request hash and exact child PID/start
ticks. Same-request retries observe the original job. Clock expiry, SSH loss,
missing markers, report existence alone, and reused PIDs cannot release a
reservation. Successful and failed actual terminal waits release it while
keeping their original evidence. A completed request can be observed again
without recreating a lease or relaunching the reader. Partial preparation with
no authoritative terminal remains blocked for explicit operator review; it is
never an automatic restart or stale-lease takeover.

This scheduler offers no finite wait bound behind a currently running full
held-out evaluator job. The historical active evaluator allows 10800 seconds
for report admission, runs its full cohort in one job, and its runner does not
have a deadline watchdog. A 200-task evaluation could therefore block a ready
reader beyond an hourly epoch. Future bounded evaluation chunks would need
separately signed exact cohort plans and honest partial/completion provenance,
or the deployment should retain a dedicated reader. The measured c1 reader
alone hashed 91,387,491,264 bytes in 278.08 seconds using four 1MiB streams,
34,740kB observed RSS, and CPU nice19; that is not a measurement of simultaneous
evaluation and readback.
