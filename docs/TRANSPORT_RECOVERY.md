# Miner deadlines and verifier transport recovery

The registered UID131 test miner completed a local candidate after the original
October 3 deadline. Its expired upload capability returned HTTP 403; no late
submission was credited or admitted to training. The prospective miner now
checks the deadline before search, after a rollout, after batch construction,
before upload preparation and before the request. A 403 received after expiry
becomes an explicit closed-window result, while an early 403 remains a transport
error. The CLI stops searching that epoch and retains earlier local batches;
it does not extend the deadline or report the late candidate as accepted.

Six deadline controls and the existing native-error, capacity, public-contract
and transport controls cover this behavior. Generation itself is not yet
preempted midway through a forward pass, and upload duration cannot be guaranteed.
Miners should leave time for proof construction, compression and transmission.

The second verifier was observed absent while its SSH process was still alive.
The retained forwarding process mapped remote port 19082, but the new worker
requested 19081. Process existence alone therefore did not prove transport
health. Root preserved the original forwarding evidence and launched a distinct
correctly bound listener rather than stopping the retained 19082 tunnel or
changing any signed job. Recovery is checked against the actual remote listener,
worker PID/start ticks and coordinator lease state.

The prospective daemon retries connection errors and timeouts every five
seconds using the same worker and coordinator lease protocol. Its one-shot mode
still reports transport failures. Authority, integrity and other unclassified
failures still stop it for investigation; they are not converted into fraud
reports. Two focused outage/integrity controls and fourteen existing worker
controls pass. This code requires future pinned-source admission; restoring the
current verifier's correct listener does not replace its scientific source.

The operator provisioner now checks actual `/proc` start ticks, process state,
arguments and working directory before reusing a process marker. Zombie processes
do not qualify. A forward must bind the requested remote port to the configured
coordinator and SSH endpoint. A mismatched tunnel is preserved with its original
marker; the provisioner attempts a separate correct tunnel without signaling the
old process. It checks the remote TCP listener before starting a worker.
An existing live worker must match the authority, coordinator, seed path,
workspace, checkpoint-cache arguments and source directory; a mismatch stops
activation rather than replacing it. Five controls cover delayed/coordinator
startup and actual process binding, including a real zombie. These operator
changes do not replace the source or processes of an active signed epoch.

## Prospective streaming snapshot freeze

The next transport change removes the operator's body-sized submission snapshot
and re-upload. `Bucket.freeze_snapshot` obtains completion metadata and the full
body from one atomic GET, hashes bounded 1 MiB chunks, and conditionally copies
the exact source ETag into its digest-bound private frozen key. It rejects a
completion outside the original signed window before reading the body. ETag
only prevents a read/copy overwrite race; full SHA256 remains the byte binding.
Publication still independently checks the frozen bytes before the public copy.

Gateway freeze schedules at most the configured 1–8 publication workers. It
persists successful siblings even if another stream or copy fails. Infrastructure
failures remain retryable errors, never miner fraud or point penalties. Already
persisted snapshots are reused on retry, so later staging changes cannot replace
accepted frozen bytes. Legacy bucket adapters keep their serial snapshot path.

Four focused tests plus sixteen existing publication/direct-R2 controls cover
bounded streaming, deadline/size rejection, conditional overwrite refusal,
partial failure and durable retry. A real private R2 qualification froze four
8 MiB objects concurrently in 1.69 seconds, independently read back all four,
and rejected an actual overwrite between hashing and copying. That small test
does not establish production throughput for hundreds of large submissions.
This change is prospective: it needs a new admitted source and a completed epoch
boundary before activation. Existing signed epochs retain their original code.
