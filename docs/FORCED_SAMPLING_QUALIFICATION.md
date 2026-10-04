# Forced sampling qualification

Scientific implementation: `310492f6a87721760630b196e375bd2ba43cbe05`.
Immutable runtime bundle SHA256:
`e415623e5a8017133ff7bef3925b724441862f34b6d034ed0deb6b731a26e657`.
Published code or successful controls alone do not establish deployment. The
signed current manifest and authenticated mining discovery establish which
contract is active. The prospective change waits for the original epoch to
finish under its original contract.

## What is checked

`forced-inverse-cdf-replay-v1` fixes public draws, approved weights, task,
context, attempt budget, temperature/top-p and sampler. An independently loaded
model regenerates every audited output token and its stopping position. The
existing full conditional probability, TOPLOC and native environment checks
remain required. A verifier's authenticated report must attest to the exact
sampling contract before an accepted batch can become new-contract reward or
training evidence. The trainer independently re-audits selected pairs.

This establishes sampler consistency. It does not prove a miner's historical
execution, remove selection bias from success/failure searches, establish
arbitrary hardware portability, or turn unchecked batches into verified data.
The old precommitted held-out evaluator remains a diagnostic and cannot supply
mining evidence.

## Actual H200 controls

Controls used the existing learned Qwen2.5-Math-7B-Instruct checkpoint
`093a61c368dc9aeee4551d87365ad5504cc7b13eb0a9bdcc999a58964e8ee570`
and the unchanged native MATH grader. The signed contract used the uncached
eager `text-tools-long-v2` harness, temperature 0.8, top-p 1 and a 1,024-token
output limit. A Level-5 task produced both required outcomes.

| Control | Tokens | Generation | Independent exact replay |
| --- | ---: | ---: | ---: |
| Successful attempt 0 | 746 | 40.22 s | 32.21 s |
| Unsuccessful attempt 1 | 1,024 | 51.45 s | 53.27 s |

Both genuine outputs passed after an independent model reload: zero observed
honest false rejections. Three attacks were rejected:

- A copied correct answer with freshly computed genuine probabilities and
  TOPLOC fingerprints passed the old verification and failed forced replay.
- An attempt outside the signed range was rejected.
- A missing sampler receipt was rejected.

The control took 420.47 seconds overall. Its complete paired artifact was
404,989,206 bytes, SHA256
`3b3d6300f826090cf7323e3d79fc95f77f37d0f763183c718813fe67c13b5da8`.
These timings are two outputs on one qualified profile, not a fleet throughput
estimate or an upper bound. Strict replay adds generation cost to verification.

A first control exhausted twenty attempts on five easy tasks because every
generated answer succeeded. It did not produce the required negative control
and was retained as an unsuccessful search. Sampling rules were not weakened.

## Independent verification and training

The actual artifact was uploaded to private immutable R2 storage and checked by
full SHA256/size readback. A separate H200 verified it through the production
operator-signed backend job path, accepting one success/failure pair and its
sampling assurance. The trainer H200 then independently re-audited the same
pair and performed one full-model covered-policy optimizer update.

The isolated successor was
`448c0ed86fdfa71cb10ad2619cc9adbaeec5503c004bede0aabe7c5036e505f2`.
This control changed no live epoch, made no chain transaction, and did not
promote its successor into the production checkpoint stream. No learning gain
is claimed. An earlier signed-backend control failed because its test manifest's
harness registry still selected the old harness. That failed original remains
in history; the corrected manifest bound both the primary harness and registry.

All five role hosts passed complete source-file checks and read-only checkpoint,
tokenizer/context and native-grader admission. The source archive and signed
descriptor passed independent complete R2 readback. The local control suite
passed 158 checks spanning sampling, stopping, malformed contracts, legacy
isolation, miner deadlines, source/harness binding, audits, training and reward
attribution. These do not replace the first real public-miner epoch after the
boundary switch.

## Operational handoff

The operator supervises the original controller until its actual successful
process wait completes. The new controller inherits its completed checkpoint,
round and training count; old submissions, reports, earned rewards and writer
cursor remain intact. New source approval, two verifier workers, covered trainer
policy, actual controller identity and public discovery switch together. The
first signed manifest must contain the required sampling contract before mining
can be advertised as open. A scoped one-shot public miner uses an already
registered operator-owned identity; its private signing key stays on the
operator host.

The cutover service is bounded and does not automatically replay a partially
executed transition. An execution failure needs inspection of its original
evidence. Completed local submission downloads are retired automatically after
authenticated report checks and complete durable R2 archive readback. This
retention never deletes bucket history or changes scientific penalties.
