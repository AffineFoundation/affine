# Training job disk admission

The prospective controller computes download space from the frozen receipts of
every miner whose audited batches enter its training request. Receipt sizes must
be positive bounded integers and match the audited submission hash; at most 256
submission files may enter a job. It counts each retained download, even if two
request entries share the same hash. Unselected uploads do not consume this
job's download reserve.

The trainer keeps those compressed ZIPs throughout the job, but decodes one
submission at a time. Admission therefore reserves the total planned compressed
bytes (at least one complete artifact budget), one raw-artifact working budget,
the existing conservative model snapshot/export allowance, any missing input
checkpoint and the fixed safety reserve. Previously the compressed reserve
covered just one submission even when the job downloaded many files.

Five new controls cover selecting/counting frozen files, substituted hashes,
invalid sizes and populations, aggregate disk refusal and integer bounds.
Existing routing, remote-backend and capacity controls pass. This changes future
disk admission, not training mathematics or artifact/numerical budgets. Active
signed jobs retain their original code and capacity receipts. This source needs
future role admission before activation.
