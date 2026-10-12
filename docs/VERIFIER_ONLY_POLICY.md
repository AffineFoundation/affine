# Prospective verifier-only policy

The signed epoch manifest controls activation. The prepared boundary is epoch
170 or a later unopened epoch; deployment and a real signed opening must confirm
it. Publishing this code does not activate it for existing epochs.

The optional `sampling_verification_policy` field selects
`symmetric-prefill-source-preserving-v1`. It binds the verifier's source hashes
separately from the generation source in `source_bundle`. Existing admitted
miners keep their generation source, sampler, checkpoint discovery, task rules,
nonces, eight-rollout quota, proof format and upload flow. No bootstrap refresh
is required solely for this verifier change. Training and optimizer behavior
are unchanged.

## Exact source artifacts

| Purpose | SHA256 |
| --- | --- |
| Miner generation | `f5287f57bdae344e5a3d073309e5c830073f603838b4b52a189f5ec0a269cb24` |
| Qualified verifier | `149e540f5eb7ec40d98331ce8e82956b87b2fd057cc2011164d25524081be4ef` |

The [qualified verifier archive](https://affine.io/sources/149e540f5eb7ec40d98331ce8e82956b87b2fd057cc2011164d25524081be4ef/source.tar.gz)
is exactly 7,242,107 bytes. Its content-addressed bucket key is
`public/sources/149e540f5eb7ec40d98331ce8e82956b87b2fd057cc2011164d25524081be4ef/source.tar.gz`.
Download it and verify the full SHA256 before extracting or running it. The
public URL requires no storage credentials. The archive contains source only,
not operator signing policies, credentials or training submissions.

GitHub main merges the verifier changes with later compatible public changes.
Its checkout is not claimed to reproduce the archive hash or replace the
signed execution inventory. In particular, `current_policy()` describes the
local checkout's files; production executes the separately qualified inventory
bound by its authenticated route. Never substitute a local policy fingerprint
for the manifest's admitted one.

## Verification and evidence

A teacher-forced prefill recomputes the submitted token path's probabilities.
For each prescribed draw, the verifier accepts a token only in the calibrated
interior of its CDF interval. A draw close to either interval boundary is
ambiguous, including a draw just inside the interval. Boundary and support
ambiguity require the prescribed cached reference adjudication. An unavailable
reference or other numerical uncertainty stays unresolved; it must not become
confirmed cheating. An exact reference mismatch remains invalid.

Worker reports bind complete per-rollout, per-turn verification events to the
actual accepted child batch. Missing, duplicated or cross-child events cannot
establish accepted evidence. The verifier policy also enters the reward
evidence cohort fingerprint so results from distinct contracts are not silently
pooled. Epochs without the optional field retain the original fingerprint and
original verification behavior. Existing signed artifacts are never rewritten.

This checks computation and sampling consistency under the admitted contract.
It does not prove a unique historical execution, original authorship or correct
intermediate reasoning. Native grading remains a separate check.

## CPU controls

From a checkout with its development dependencies installed:

```sh
python -m unittest discover -s tests -p 'test_symmetric_verifier*.py' -v
```

The portable controls cover symmetric boundary handling, cached replay and
mutation rejection, neutral unavailable-reference results, complete report
coverage, source-aware backend admission, unchanged generation draws and
receipts, historical cohort compatibility and the generation-source bootstrap
pin. GPU qualification of the exact frozen archive is separate from these CPU
controls; changing a checkout's numerical runtime requires its own qualification.
