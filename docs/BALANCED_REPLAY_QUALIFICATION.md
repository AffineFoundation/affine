The prospective replay qualification uses authenticated historical FULL audits
from nine original environment families. It does not credit a new miner batch or
submit blockchain weights. Historical chosen actions may have come from curated
public-input candidate policies; replay is not autonomous search.

`ops/build_verified_replay_pool.py` binds frozen ZIP hashes, signed manifests and
audits, exact environment/harness versions, model/tokenizer geometry, and the
complete heldout registry. The v3 pool selects one contribution per environment
before selecting a second. Historical probability rows are evidence, never the
reference distribution for a later checkpoint.

`ops/probe_balanced_replay_optimizer.py` accepts a separately operator-signed
qualification plan and exact pool/current-manifest bindings. It recomputes full
probabilities and TOPLOC proofs at the approved current checkpoint, then replays
each original environment. A single AdamW instance captures every pair's reference
before any update and performs at least one update for every selected family.
Reports retain exact positive/negative hashes and optimizer counters.

Qualification reserves GPU memory and remote disk space before model loading.
It preserves each intermediate checkpoint. This is an isolated qualification,
not yet the continuous controller's replay or cross-epoch optimizer state. The
nine-family v3 pool in `state/verified-replay-pool/currentd8-v3` has authenticated
historical lineage; its preparation alone proves no optimizer behavior or heldout
improvement. Comparable heldout evaluation and a successful reviewed controller
integration remain necessary before learning-effectiveness claims.

The isolated nine-family run completed successfully in
`state/balanced-replay-qualification/1790865379`. Its current input checkpoint was
`d8e047f13278692e0baa0df213b4d5566318e582bb74e47754a63a91d6125383`;
the ninth update produced
`7b2068a580a5134f500e961b5732fd4ed94b214c546c18f5b993a766068d3ba6`.
All nine pairs passed freshly computed probabilities, TOPLOC and native replay.
Updates covered i3math, Logic, Math, PopQA, Reasoning Gym, SciText, Trivia,
Unscramble and Verbatim, with optimizer counters 1 through 9 and gradients for
all 218 parameter tensors. A root audit authenticated the signed completion,
source archive and report bytes. It did not recompute model inference itself.
The successor is now published to R2. Root independently streamed and hashed
all six checkpoint objects, totaling 3,426,302,727 bytes, against the signed
qualification file map. This qualification has paired model-audited heldout
evaluations at both checkpoints: 192 fixed original tasks, 16 per environment,
with identical dataset identities and no evaluation failures. Math reward rose
from 0.375 to 0.5; Trivia fell from 0.5 to 0.4375 and RGym fell from 0.001489437
to 0.001055941. The other nine environments were unchanged. These mixed results
do not establish a general improvement.

Continuous replay integration remains prospective. To reproduce the reviewed
controller changes in a new operator-owned MRCR-v7c source tree, run
`python -m ops.prepare_balanced_replay_source --source APPROVED_MRCR_TREE
--destination NEW_TREE`. The utility verifies exact base, patch and helper
hashes, rejects source symlinks and destinations inside the input, and copies
source without models, state or credentials. It changes no running service.
Cached training retries republish authenticated metrics; consumed historical
targets are recorded in one atomic epoch/count journal only after checkpoint
publication. Replay targets produce optimizer data, not additional miner points.

The root evidence checker independently authenticated both evaluation jobs,
matched all 384 model-verified task outcomes to the 24 aggregate records, checked
identical before/after source inventories and fixed task identities, and found
all 24 records in the actual affine.io HTTPS export. This checks collected
execution evidence; it does not repeat GPU computation on the root machine.
The continuous auditor now accepts historical attribution only with an explicitly
signed replay request, exact current-compatible pool and consumed target hashes,
fresh numerical/native checks, and matching authenticated metrics. Historical
pairs remain outside miner scoring; fresh-only epochs retain their old checks.
