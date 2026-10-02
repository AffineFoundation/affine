# Single-environment MATH pilot

The next pilot focuses on the original `affine_math` environment, starting from
the actual upstream base checkpoint rather than a checkpoint already trained by
the multi-environment experiment. The pilot must remain nonpayable while its
mining, verification, training and checkpoint handover are demonstrated.

The original loader yields 7,496 distinct problems from the MATH training split;
four of the nominal 7,500 rows lack a usable boxed reference answer. The original
I3Math loader yields 7,583 problems under its default difficulty filter. MATH is
selected for its nearly equal population and more attainable initial successful
samples. Its original public prompts, answer format and grader are preserved.

Preparation reserves 750 problems for evaluation and authorizes the remaining
6,746 for mining. A fixed initial evaluation cohort contains 32 problems across
subject/level strata. The reserved problems cannot earn mining scores or enter
training. Evaluation cohorts retain their identities across checkpoints; broader
evaluation must not be merged into a supposedly comparable series silently.

The public challenge authorizes the full 6,746-problem mining pool. An operator
may set `owned_mining_subset` to a bounded starter list for its own signed mining
job; the worker checks it is a subset of authorized training indices. This does
not restrict external miners or expose an alternate heldout admission path.
The preparation recipe instead uses `owned_mining_schedule`: 422 groups of up
to 16 training indices, rotating by epoch round. This avoids repeatedly starting
the owned miner on the same problem while preserving the full external pool.

All task prompts were rendered using the upstream tokenizer: the largest prompt
is 2,098 tokens, within the 8,192-token context with a 512-token output reserve.
The initial candidate configuration uses genuine autoregressive sampling with a
256-token output budget and K1/L1 batches. It supplies no answer candidates.
Full probabilities and TOPLOC records are still required; the existing upload
budget is 100 MB compressed, and each probability tensor is bounded to 512 rows.
Rejected cumulative additions must preserve previously acknowledged uploads.

## Required demonstration

1. Authenticate the untrained upstream model revision and actual file hashes.
2. Publish a new source-bound challenge with the full mining pool, direct R2
   discovery and encrypted private upload capabilities for approved identities.
3. Run a miner against that public interface, generate genuine positive/negative
   batches, freeze them, and independently verify inference and original grading.
4. Score unique valid indices and persist proposed normalized weights, with no
   blockchain weight transactions.
5. Train only on accepted mining batches, publish the changed checkpoint, and
   demonstrate consecutive epochs and recovery from an empty epoch.
6. Run fixed heldout evaluations from the base checkpoint onward and expose their
   comparable performance plus epoch batch counts on affine.io.
7. Repeat the miner-facing setup with an external approved identity and document
   the reproducible installation, pinned runtime, discovery and upload flow.

MATH problems and reference answers are publicly available benchmark data. This
pilot does not claim secret-test protection or proof that submitted tokens were
naturally sampled. Inference verification establishes approved model computation
on the submitted trace; original grading establishes the outcome. The owned
baseline and miner controls must use genuine autoregressive sampling. Public
benchmark contamination remains a limitation of any improvement claim.

The taskset, split and prospective configuration are prepared. Fresh base-model
qualification, deployment, real math-only epochs and performance improvement are
not yet demonstrated. More accepted data and repeated comparable evaluations are
needed before claiming effective learning or public release readiness.
