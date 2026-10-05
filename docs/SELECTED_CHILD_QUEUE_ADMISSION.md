# Selected child queue admission

Small-commitment epochs enqueue the selected heavy child artifacts, not the small
parent commitment hash. The signed original verification job contains the child
SHA and upload capability plus its miner, parent commitment SHA, slot, environment,
index, batch hash, size and frozen key. Admission authenticates the original miner
commitment embedded in the ROOT-signed frozen receipt, checks canonical parent digest and checkpoint/source,
approved task membership, and compares every child field with both inventories.
The capability must address that exact immutable epoch/miner/parent/slot object on
the frozen receipt's storage origin. Refreshed capability queries are allowed;
a different origin, path, slot, size, miner or declared task is rejected.

Queue retries preserve all child metadata. The actual authenticated worker report
must bind the original job and child SHA; accepted pairs must name the committed
environment/index and full canonical accepted batch digest. Legacy whole-ZIP epochs retain their original hash admission.
These controls establish metadata and queue binding. Full selected payload SHA,
forced-sampler replay, log probabilities, TOPLOC and environment grading still run
on the scientific worker; this change does not claim numerical execution.

E11's admission bug and freeze-budget deferral remain infrastructure failures.
Its published frozen challenge and signed original jobs must never be relabeled,
backdated, rewarded without audits or treated as fraud. After its signed audit
cutoff the unchanged once controller can close honestly with zero updates and
unchanged learned optimizer parent. Apply this fix in a newly approved source and
fresh epoch alongside the two-phase freeze fix, not by modifying E11 source bytes.
