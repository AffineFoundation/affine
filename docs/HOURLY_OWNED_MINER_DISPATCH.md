# Owned mining does not block collection

Prospective signed `bounded-hourly-phases-v1` epochs dispatch the owned miner
once and persist its original job ID and request digest. Dispatch observations
are not successful computation reports or rewards. The controller immediately
enters collection and captures submissions at the original deadline without
waiting for that remote miner's terminal report. Generation, sampling, upload
expiry, cumulative quotas, proofs and frozen-receipt admission stay unchanged.
Historical epochs without the signed hourly policy retain serial observation.

A mining deadline does not release the physical GPU. Before a later owned job,
the controller authenticates original hourly requests and probes their original
runner markers, retaining the same PID/ticks records. An early report file is
insufficient: the physical probe must observe the supervisor's child-wait
terminal, or all recorded processes gone. Running, absent, ambiguous or
unreachable originals block a new miner dispatch while external collection
continues. Failure is retained as failure, never inferred as successful mining.
New dispatches record their workspace; changing it requires an explicit original
reservation reconciliation, not probing a different empty workspace. The current
V12/V13 owned workspace is stable; historical non-hourly requests do not acquire
new retrospective reservations. Signed original requests are never relaunched
on resume, including missing launch markers.

Focused controls exercise actual service transition at the deadline with an
unfinished miner, immutable adoption, absent/ambiguous/SSH-failed reservations,
original terminal release, and report-before-child-wait physical probing. This
patch does not modify a running epoch or promise a wall-time bound for SSH
provisioning: bounded physical probes occur before dispatch, rather than while
collection is due. Existing freeze and audit cutoffs remain authoritative.
