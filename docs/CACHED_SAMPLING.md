# Explicit cached Qwen2 sampling

`text-tools-long-kv-v3` is a prospective opt-in harness for Qwen2 models. It
prefills the prompt once, then feeds one sampled token per step with a local KV
cache and computes only the next-token logits during generation. Temperature,
top-p, output limits and per-turn settings remain in the signed harness. It
supports autoregressive sampling only; a cache toggle cannot change an older
harness. Each call starts a fresh cache, including each environment turn.

The existing `text-tools-long-v2` sampler stays unchanged. The new path does not
change full teacher-forced replay: all token probabilities, hidden activations
and TOPLOC proofs are computed with the original full-context computation.
Cached and uncached generation need not yield identical seeded tokens in BF16.
Verification checks the submitted trajectory's model computation, independently
of that sampling equivalence claim.

Four portable CPU controls cover actual cache calls, call isolation, EOS,
architecture/context refusal, legacy dispatch and the explicit GPU-runtime
dispatch on a tiny CPU model. A retained H200 generated eight real successful
MATH trajectories using the learned checkpoint and passed full same-node native,
probability and proof replay. A separate high-temperature trial preserved five
failures before the native grader raised an error; that exception is unscorable,
not a negative training sample. Independent replay on a second H200 has now
passed for one genuine success and one genuine failure from different tasks:
all 152,064-wide F32 log probabilities matched exactly, and every TOPLOC block
had zero mismatch. Token, proof, probability and model-weight mutations were
each rejected as `InvalidSample`; restoring the weights restored honest replay.
Root authenticated the original signed job and actual exit-zero receipt.
A subsequent original generation job produced both a success and a failure for
MATH task 1278 under the same task specification, checkpoint and cached harness
(temperature 1.0, 1,024 output tokens). Independent replay on the second H200
passed for both, with exact full probabilities, zero TOPLOC mismatches and all
eight typed tamper rejections. Root authenticated the original signed generation
and verification jobs, checked every transferred artifact hash and read the
original supervised exit-zero result directly from the verifier machine.
This qualifies the sampled trajectories, without proving sampler-seed provenance.
Prospective source/checkpoint admission on all production roles is still required
before activating the new harness in a public epoch. No existing manifest or
deadline is changed. Keep held-out comparisons on the same versioned sampling
harness before and after training.

The miner now treats the native `TaskError` as an unscorable generation attempt
and keeps searching within the original attempt/deadline limits. It retains
earlier genuine samples and never turns a grader exception into a failure sample.
Other model, proof or assertion errors still propagate. Three controls check a
success/error/failure sequence, refusal to count the error, and unrelated faults.

R2 publication also supports server-side copying from the hash-checked private
frozen key, avoiding an extra upload through the validator host. It never copies
mutable staging bytes. Eight storage controls cover deadline selection,
late overwrites, corruption and publication/copy recovery; a private real R2
copy passed byte-for-byte readback. This does not measure multi-gigabyte throughput.
