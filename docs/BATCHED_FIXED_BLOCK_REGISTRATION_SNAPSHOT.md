# Fixed-block registration snapshot

Epoch opening and reward snapshot collection authenticate every activated miner
against one immutable chain block. The commitment parser, Ed25519 activation
signature, subnet-owner exclusion, activation height, public key and UID/hotkey
reverse check are unchanged.

`ChainAdapter.registrations()` now materializes subnet `Uids` and `Keys` maps at
that same block instead of requesting both entries separately for every valid
activation commitment. Duplicate map keys, duplicate reverse identities, wrong
types and negative UIDs fail closed as infrastructure errors. A failed map read
raises; it never fabricates an empty authenticated roster. Missing or stale
forward/reverse mappings still exclude the corresponding miner.

A read-only current-SDK control at block 9214687 returned exactly the same 246
registered identities and activation metadata:

| Implementation | Actual seconds | SDK storage calls |
| --- | ---: | --- |
| Original | 101.912 | 1 map plus 696 scalar calls |
| Final candidate | 16.505 | 3 maps |

Repeated valid activation entries explain why scalar calls exceeded twice the
final roster size. Map calls may involve SDK pagination, so three map calls is
not a claim of exactly three network RPCs. The candidate spent 14.94 seconds
reading commitments, 0.80 on UIDs and 0.74 on reverse keys in this observation.
These timings exclude SDK construction/import and are one fixed-block control,
not a guaranteed epoch-speed benchmark. No wallet or chain transaction was used.

The final actual source-bound parity evidence is retained privately at
`state/root-audits/batched-fixed-block-chain-registration-readonly-20261005-v1/ACTUAL-FINAL-SOURCE-FIXED-BLOCK-PARITY.private.json`.
Its candidate file was hashed before execution and independently reread after.
The preliminary candidate timing is not used as final source-binding evidence.

This change is prospective. The running V12 source and original epoch jobs remain
immutable. Production adoption requires the next reviewed source bundle and
manifest boundary; pushing this code alone does not replace a running epoch.
