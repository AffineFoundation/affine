# Bounded trainer input transport

The prospective backend downloads compact committed training documents four at
a time, with at most four input reads ahead of ordered eligibility admission.
Each executor thread keeps its own HTTP session for this job and closes it on
success, transport failure, digest rejection or admission failure. Other roles
and noncompact training retain their existing serial transport.

R2 host validation, disabled redirects, byte limits and complete SHA checks
remain enforced. Downloader threads only transport and authenticate bytes.
They never admit training pairs, validate native prompts, modify optimizer
state or perform inference audits. Eligibility admission retains original
submission ordering. All cache ownership receipts are written serially by the
consumer, including completed read-ahead inputs after an admission failure.

The reported `submission_download_and_authentication` value sums individual
request durations and is therefore **worker time**, not elapsed wall time.
`submission_prefetch_pipeline_wall` records the elapsed pipeline, including the
ordered eligibility admission that consumes it. Comparing the sum with the
pipeline duration directly would overstate a speedup.

Inspection confirms that the previous backend issued serial `requests.get`
calls with a new session for every small token document. E15 did not contain
startup stage timings, so its unallocated residual runtime cannot truthfully
be attributed to downloads. This change needs a real qualified epoch before
claiming a measured improvement. It does not remove the separate full-size
FP32 optimizer restore, export or independent durability readback costs.
