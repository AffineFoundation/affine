# Forced-sampling adversarial review

Reviewed on 2026-10-04 for `forced-inverse-cdf-replay-v1`. The scientific
candidate is bundle
`e415623e5a8017133ff7bef3925b724441862f34b6d034ed0deb6b731a26e657`.
This review does not assert deployment, GPU qualification, held-out gains or
absence of every possible exploit.

The admission claim is: each credited trajectory's tokens match the prescribed
sampler applied to the approved checkpoint and authentic environment context.
TOPLOC and full probability checks establish target-model computation on those
tokens. Independent exact sampler replay establishes the token choices and
stopping rule. Environment replay establishes the approved grader's outcome.

No arbitrary teacher-forcing reward bypass was found in the reviewed path.
Rebuilding genuine probabilities and TOPLOC proofs after choosing different
tokens still fails exact sampling replay. The verifier recomputes every audited
turn and token; a matching first turn or prefix cannot authorize the remainder.

## Additional controls

`tests/test_sampling_adversarial_review.py` adds ten CPU controls, using real
small causal-model computations and TOPLOC proofs for the trajectory attacks:

| Attack | Result |
| --- | --- |
| Change the final token of the second turn and rebuild genuine computation evidence | Legacy computation-only verification accepts; forced replay rejects |
| Truncate an authentic suffix and rebuild proofs | Legacy accepts; forced replay rejects |
| Append output after sampler EOS and rebuild proofs | Legacy accepts; forced replay rejects |
| Reseed only the second turn and rebuild proofs | Legacy accepts; forced replay rejects |
| Substitute context and rebuild proofs | Authentic environment context check rejects |
| Substitute checkpoint or epoch in a fresh sampling receipt | Receipt binding rejects |
| Controller-sign a legacy computation-only assurance for a forced epoch | Reward projection rejects |
| Keep correct report assurance but change one credited rollout's attempt | Reward projection rejects |
| Miner-sign its own positive audit | Authority authentication rejects |
| Provide valid sampling receipts for a batch without a completed full audit | Reward projection rejects |

Run these controls from the repository root:

```sh
.venv/bin/python -B -m unittest discover -s tests -p test_sampling_adversarial_review.py -v
```

All ten passed. This is additional local evidence; it does not replace independent
H200 controls or a completed public epoch. The reviewed scientific sampler,
model, GPU runtime, protocol, reward bridge, miner and transport modules matched
the immutable candidate bytes when inspected. The operational historical source
requirements repair is separate from this candidate.

## Trust boundaries and remaining limits

Only fully audited, accepted batches earn points or enter training. Correct
sampling receipts alone do not authorize credit. The live writer additionally
authenticates the original signed job, approved source inventory, approved
worker identity, completed queue record and exact original audit metadata.
The reward projection function is not itself a GPU verifier and must not be
exposed as a miner-report admission endpoint.

Verifier nodes are operator-owned trusted execution services. A compromised
approved verifier signing key could attest false results; these receipts are
not a cryptographic proof against a malicious approved verifier. Independent
trainer re-auditing protects training inputs, but does not by itself protect a
reward already released from a false verifier attestation. Key isolation,
original authenticated job lineage and operator reconciliation remain required.

Miners can select tasks and choose among up to 128 approved deterministic
attempts to obtain one success and one failure. This is deliberate selection
bias. Verified sampler membership does not make the selected training population
an unbiased on-policy policy-gradient sample. Training objectives and evaluation
must account for this distinction.

The contract does not prove who historically executed a trajectory or prevent
miners from sharing an authentic trajectory. Matching tokens and authentic
computation are the enforceable admission claims. Same-epoch collision scoring
continues to zero all observed verified copies. This retains the previously
identified Sybil/collusion cancellation incentive; submission privacy reduces
targeted copying but does not remove it.

Grading proves the pinned environment grader accepted the outcome. It does not
prove every reasoning statement is correct. Grader exploits require separate
environment-specific tests; the sampler cannot repair a defective grader.

## Historical source repair boundary

Historical signed jobs must be checked against their original approved archive's
source requirements, rather than a newer source module list. It is safe to
derive those requirements from an authority-approved, SHA-authenticated archive,
parse only a literal declaration or the exact closed tuple/path generator, and
then check every original job module digest against that same archive. Reject
missing, empty, ambiguous or executable declarations.

Live worker admission must retain its current source requirements by default.
Neither a miner nor a mutable job field may select a weaker source inventory.
Do not import or execute historical archive code merely to discover its list of
required modules.

## Next priority

Complete the first public forced-sampling epoch and independently reconcile its
frozen uploads, actual verifier reports, reward vectors and covered training
inputs. Then measure a predetermined held-out comparison over nonempty epochs.
Only after these gates pass should a new contract reduce full auditing or raise
the batch cap. Any such contract needs explicit public documentation of which
samples are verified, rewarded and admitted to training, plus adversarial tests
for its changed assurance boundary.
