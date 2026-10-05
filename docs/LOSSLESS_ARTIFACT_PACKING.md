# Lossless pair packing, prospective opt-in

The E11 miner spent about 95–102 seconds packing each full-vocabulary pair.
An existing local genuine math pair (task 4114, original source763/CP28) contains
1,172,782,103 uncompressed bytes and two full-vocabulary FP32 NPY tensors. A
CPU-only benchmark repacked it on Arbos with stable framing and checked the SHA
and size of every complete unpacked member against the unchanged original.
No GPU rollout, remote rental, R2 download or production mutation was involved.

| DEFLATE level | Packing seconds | Container bytes | Relative bytes vs level 6 |
| --- | ---: | ---: | ---: |
| 6, historical default | 145.510 | 438,273,804 | 1.000 |
| 1 | 16.126 | 496,220,332 | 1.132 |
| 3 | 31.066 | 461,101,176 | 1.052 |
| 0 | 3.294 | 1,172,871,915 | 2.676 |

Level 1 was about nine times faster for CPU packing, with 13.2% more upload
bytes. This is a same-host compression comparison, not a measured epoch-speed
or H200-time guarantee. Level 0's much larger upload is a poor initial tradeoff.
Private actual evidence and the exact executed packing source are preserved in
`state/root-audits/lossless-zip-packing-benchmark-20261005-v1`.

The operator can select level 1 in a NEW signed manifest:

```json
"artifact_compression_policy": {"version": "lossless-deflate-v1", "level": 1}
```

This is an exact two-field policy. Levels must be integers from 0 through 9;
booleans, floats, missing or extra fields and unknown versions are rejected.
An absent policy preserves level 6 and exact historical default writer bytes.
The controller permits the new policy only with per-pair commitment transport.
Both owned mining and the official external miner use the same one-pass pair
packer and automatically follow the signed manifest.

The official miner CLI and signed-source bootstrap accept
`--compression-level 1` as an optional assertion. It must match the signed
manifest and cannot override it; omitting it follows the manifest automatically.
Source pinning is checked first, and a mismatch is refused before model or
checkpoint loading. Existing acknowledged prepared slots resume their exact
original bytes without recompression.

All logits, proofs, token records, model computation, samplers, native grading,
FP32 array bytes and verifier rules are unchanged. Container SHA values naturally
change for a different encoding: miners commit and upload the actual new bytes,
and validators still hash those exact bytes. No historical digest is rewritten
or relabeled as equivalent. The selected compression level is a transport
instruction, not a claim that the archive proves any sampling behavior.

Compressed and expanded limits remain unchanged. Independent prepared sizes and
the cumulative hypothetical ZIP are both bounded; cumulative manifest framing
uses the selected DEFLATE level too. A faster level that exceeds a cap is rejected
rather than increasing that cap. Stable names, timestamps, permissions, tensor
shape/dtype checks, one-pass writing, receipt hashes and upload-journal ordering
remain intact. Activation requires a future root-reviewed source boundary.
