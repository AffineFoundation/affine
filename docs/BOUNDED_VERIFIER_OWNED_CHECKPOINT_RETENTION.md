# Bounded verifier checkpoint reuse (prospective operator policy)

The operator worker can opt into one owned, complete checkpoint retained after
its authenticated coordinator report ACK. The checkpoint inventory came from
the ROOT-signed job and its already durable publication. This ACK does not claim
a fresh full R2 readback of the audit report. The next scientific backend still
checks every checkpoint file against its signed full SHA inventory; no proof,
sampler, job admission, source registry or grading behavior changes.

Both explicit limits are required:

```
--owned-checkpoint-ttl-seconds 1800 --owned-checkpoint-disk-floor-bytes 2147483648
```

With neither flag, the existing immediate post-ACK deletion remains the default.
The TTL is at most one day. An idle worker retains only the newest complete
`authenticated-job-ACK` receipt within TTL. Partial and failed-download receipts
are not retained as current. After ACK the same sweep applies. The next job on
a different checkpoint retires older owned checkpoint bytes before download,
while excluding its requested checkpoint. Retry workspaces remain disposable.
Low disk headroom overrides reuse. Every retirement still requires exact owned
paths, unchanged inode/stat/member inventory, and a nonblocking exclusive lease.
A leased obsolete checkpoint may coexist temporarily; it is retired on a later
poll after release. Changed/unowned/external bytes are deliberately not deleted.
No automatic adoption of historical directories occurs.

ROOT rollout should change only the CPU operator overlay pair
`subnet/cache_lifecycle.py` and `subnet/distributed_worker.py`, plus the two CLI
flags. Keep all scientific source trees and existing source admission registries
immutable. Reuse the existing seven-worker bounded roll: authenticate each
current worker PID/start ticks/unit invocation, queue and pending report state;
wait for its original backend and report transport to terminate; stop that CPU
parent only at a genuine idle boundary; stage/hash the new overlay in a fresh
namespace; launch the same worker identity, endpoint, workspace, mapped caches,
registry and backend source with the explicit limits. Preserve all original
leased attempts and reports. Do not interrupt GPU jobs or discard unowned legacy
models. Rollback drops these flags and returns to the previous pinned overlay at
another idle boundary. No source grant or science bundle replacement is needed.

Before claiming reuse operationally, observe two distinct authentic jobs on the
same checkpoint with full backend map verification and unchanged owned model
inodes, then observe an idle TTL or superseded-checkpoint retirement. Record
actual free bytes and lease-protected exceptions. CPU controls establish the
policy, not a measured live throughput improvement.
