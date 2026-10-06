# Prospective selected-token probability artifacts

This is an explicit new artifact contract for review. It has not activated a live epoch or changed any historical full-vocabulary artifact. Source9f and its qualification evidence remain immutable.

The signed epoch manifest can opt in with:

```json
"probability_artifact_policy": {"version": "selected-token-logprobs-v1"}
```

Without this field, generation and verification retain the full-vocabulary format and all existing probability comparisons. Unknown policies are refused. There is no format inference from submitted shapes, compression or file size. The operator must admit a new complete source containing `subnet/probability_artifacts.py`; the backend requires its source pin. The field is included in training computation bindings so an execution amendment cannot remove or change it.

For the new contract, each existing turn NPY file contains an ordinary finite float32 matrix with shape `[len(output_tokens), 1]`. Row `i` is the full-model log probability of that turn's token `output_tokens[i]`, before sampling temperature or top-p filtering. Token IDs stay in the existing authenticated rollout. All existing turn context, text, observations, stopping, reward, task identity and TOPLOC fingerprints remain in the manifest JSON. There are no object arrays, pickle payloads or compressed probability sketches. Stable ZIP framing and the signed lossless compression setting remain compatible.

Generation still executes the full model and builds the same TOPLOC fingerprints. It gathers the selected probabilities before retaining/serializing each rollout, reducing both persisted arrays and the search frontier's retained array memory. This proposal does not accelerate model forward passes.

The verifier independently computes the complete `[output_tokens, vocabulary]` distribution under the authenticated checkpoint and full context. It compares every selected claim with the same existing calibrated probability tolerance, verifies all required TOPLOC fingerprints, and then checks the prescribed random draw at every verified token. Exact replay and calibrated prefill CDF/support-adjudication behavior remain unchanged. The full reference distribution is passed to sampling verification; submitted selected probabilities never determine the sampler or its CDF. Compact claims cannot run with the forced sampling context disabled. A valid TOPLOC proof for synthesized tokens does not establish valid sampling. Missing/nonfinite reference distributions, selected claims, tokens, proof states or receipts are refused.

The new contract deliberately replaces the miner's all-vocabulary probability claims with selected-token claims. It does not claim that an old full-vocabulary proof obligation was satisfied by a compact historical artifact. Legacy verification still rejects a forged unselected vocabulary entry and rejects selected-only matrices. Conversely, compact verification rejects full-vocabulary matrices. Historical submissions are routed to their original complete source and signed manifest.

An original live7f audit object measured through bounded central-directory and NPY-header reads was860,376,975 bytes. It contained two float32 arrays shaped309×152064 and1024×152064, representing1333 output tokens. Their declared raw payload was810,805,248 bytes plus256 bytes of NPY headers. The JSON member was48,187 bytes, including86 TOPLOC fingerprints with29,584 base64 characters. Two new selected arrays for those token counts would contain5,332 probability bytes, a152,064-fold payload reduction; the retained JSON/proofs would dominate the resulting artifact. This is a format-based projection, not an executed conversion or a new audit verdict. The probe read216,975 bytes in four bounded HTTP Range requests and did not rehash the full archive. Member compression/type metadata are untrusted until the original full verifier authenticates and decodes the object.

CPU controls use genuine tiny-model computations and TOPLOC evidence, verify identical tokens/context/proofs across formats, reject changed tokens with fresh genuine proofs, changed seeds with fresh receipts, forged selected probabilities at every position, changed context/task/checkpoint bindings, missing proofs/references, malformed shapes/dtypes and disabled sampling. A real bounded serializer comparison for64×4096 float32 distributions produced913,952 bytes in the legacy ZIP and832 bytes in the selected ZIP, with identical token/proof metadata:1098.5-fold smaller. This tests serialization, not production GPU compatibility or live throughput.

Before activation, ROOT must review and seal a new source, choose the explicit manifest policy for new epochs, admit the complete source on workers and in authority source unions, and perform applicable original native/GPU calibration controls. No old9f source replacement, historical policy rewriting or artificial proof/durability ACK is part of this proposal.
