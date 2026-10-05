# Prospective native prompt session reuse

`validate_native_prompt` previously constructed and closed one environment
session per admitted pair. Each construction rechecks environment source/data
hashes, dependency versions, and creates an event loop. The proposed change
keeps one session per exact canonical specification for one invocation only,
then closes every session in a finally block. Every pair still calls reset
with its original task index and environment seed; native reset closes prior
runtime state before preparing the next task. No global cache, model call,
outcome grading, proof verification, or cross-checkpoint state is added.

Native task hash, seed, tokenizer-rendered prompt, output length, vocabulary
bounds and one-turn checks remain identical for both positive and negative.
Checkpoint/epoch/commitment identity is still admitted by the original caller;
this function neither changes nor invents those receipts. Different source
hashes, grader dependency/config versions or other spec fields produce a new
session key and independently run native construction checks.

Per-code rationale, not a measured end-to-end speed claim: for161 same-spec
admitted pairs, native constructions/source+dependency checks/event loops drop
from161 to1. Reset still constructs the taskset, parses the snapshot and
prepares each task, so this does not eliminate that per-task cost. The session
is retained only through prompt eligibility and fully closed before gradients.
No optimizer cache, storage cleanup or epoch timestamp is affected.

Five focused controls plus37 learner admission controls pass: same exact spec
constructs once and resets each original index/seed; changed source constructs
separately; changed prompt/cross-task negative is rejected; reset errors close
the session; no model/grader method is available in the test runtime.

This is undeployed. committed_training_inputs.py is an executed source pin,
so activation requires a prospective signed source inventory/admission and
bounded real-native qualification, not an operator-only overlay. Do not change
original E14 jobs, manifests, source hashes or accepted training documents.
