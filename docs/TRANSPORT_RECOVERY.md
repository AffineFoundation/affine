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
