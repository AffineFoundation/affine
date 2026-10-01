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
