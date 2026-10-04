# Arbitrary model text at the native grader boundary

The independent 200-problem baseline completed 80 verified tasks before its
sixth shard exited with `ValueError: embedded null byte` wrapped as a native
`TaskError`. The original runner recorded child exit 1; independent observation
confirmed that both original processes were absent and the GPU was idle.
This was a grader transport failure, not a verified incorrect answer or a
completed benchmark. Original jobs, partial results and the precommit remain.

The original native MATH taskset passed the gold answer and the full generated
reply directly as subprocess arguments. Unix process arguments cannot contain
null bytes. The prospective repair serializes the two strings as an escaped
JSON array, then decodes that array inside the same grader. The full original
text is preserved, including nulls and Unicode. Last-boxed selection,
brace handling and the math-verify predicate remain unchanged. The grader also
keeps the original two-argument CLI for compatibility.

Four controls invoke the real local math-verify script. They reproduce the
original OS failure, check exact roundtrips and positive/negative outcomes with
null characters, compare encoded and original grading on ordinary responses,
and reject malformed argument arrays. This is CPU transport qualification;
independent remote native/proof controls are still required before deployment.

The two changed native files change the environment source hash. Existing
signed epochs and source directories must not be edited in place. A new pinned
source and updated environment binding are required for future work. Resuming
the independent benchmark requires an explicit transport-only amendment that
preserves its task cohort, seeds, checkpoint and sampling contract. Do not drop
the failing problems or treat transport errors as model failures. The complete
baseline must be measured under the repaired source before the precommitted
post-training comparison is judged.
