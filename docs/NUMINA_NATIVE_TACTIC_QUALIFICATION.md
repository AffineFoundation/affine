# Original Numina native resource qualification

A prospective snapshot contains 32 original cached Numina tasks, with 32 distinct
formal statements, 31 distinct provider names, and no formal-statement overlap
between mining indices 0–15 and heldout indices 16–31. The duplicate provider
name is disclosed; this is not a claim of 32 distinct UUIDs. Snapshot SHA256:
`bedd979e813cab4c035870e3f38fa2519503f82145a2bd43bb97cbb9dcd974a3`.
Original dataset revision: `51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94`.

The retained machine already has the original configured Mathlib image cached:
`projectnumina/kimina-lean-server@sha256:588a2cbbd10da509ed13f53ac136f8463fabff02dfe4eca535e7c47ae6e3ffd9`.
No local image pull or new rental is required.

`ops/probe_numina_native_tactics.py` runs bounded mining-side native controls.
Actions read only the sandbox's public starter and replace its placeholder with
automated Lean tactics. They never read the snapshot's hidden reference proof.
The original protected-signature guard, Lean compiler and grading function decide
reward. This is tool-resource qualification, not model generation, TOPLOC proof,
training, or heldout evaluation. Actual positive/negative controls must be read
from the experiment's terminal records; launch alone establishes no success.

The first immutable run is under
`state/numina-native-qualification/1790872228`, with source archive
`f7ad7f89b97bd21a7ce10e5393de9feee9a49f6c778a637ae8032d153cf6de21`.
Its single worker runs the original Docker runtime and compiler bounds, records
errors separately, and closes owned task containers after each native attempt.

The native run completed with exit 0: index 13 received reward 1, with the
protected signature intact and Lean compiler exit 0. All other mining indices
received reward 0; three were explicitly rejected by the original signature
guard. A fresh negative control on index 13 only inspected the working directory
and left the original starter unchanged; the original grader gave reward 0.
Root checked the actual source/snapshot/report bytes and signed a native-only
completion inventory. These controls establish a reachable native K1L1 target;
model/proof generation and full pipeline integration remain unverified.

A second native control tested the exact shared candidate pair intended for model
verification. Both commands read the same public starter and differ only in the
placeholder-count assertion (`n>0` versus `n<0`). Repeating each command through
the full original two-turn environment produced positive and negative outcomes
respectively on index 13. Model-token length equality is checked at runtime;
character-length equality alone is not accepted as proof.

`ops/probe_numina_model_search.py` is a separate bounded prospective proof test.
It authenticates the approved plan, source closure and public-tactic helper,
loads the pinned current model, retains sampled candidate K1L1 traces if found,
and runs an independent model reload for full-probability/TOPLOC/native replay.
It uses at most 16 seeds on index 13. It makes no optimizer or chain call. Its
new source archive is
`cd8b797fbf558548f5e55fdc06f0f8f854f4e47065081d59308d31bc47d4fe16`
and input checkpoint is `aaac517b5a1a39f3fdd78cf2c73adbad62f9f8b94f00793a95f7f8bcf6d3739d`.
The runtime proof test is pending; native controls do not establish its result.
