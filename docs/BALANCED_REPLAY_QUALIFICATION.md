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
