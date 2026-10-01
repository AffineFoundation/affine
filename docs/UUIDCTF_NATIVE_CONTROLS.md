# Original UUIDCTF native controls

Two original Affine UUIDCTF tasks now have successful and unsuccessful sandbox
trajectories with fresh native replay. Standard difficulty, the original seeded
index mapping, synthesized corpus and original reward implementation are
preserved. No model generation, TOPLOC, remote miner epoch or training is
established by this prerequisite.

The reviewed stdlib forensic solver reads only public files inside the task
container. It matches the incident's customer to an account, filters evidence by
case, tenant, material type and incident time window, decodes five different UUID
encodings, orders the records and applies the public protocol's hash reducer.
It never receives the hidden expected answer or operator task snapshot. The
original task's grader reads the resulting sandbox answer file and scores it.
The negative control writes an incorrect UUID. A fresh environment reexecutes
all committed commands and compares the complete reset, observations and score;
forged tool observations and rewards are rejected.

Evidence under `state/native-uuidctf-controls-v1` records:

- Two original positive/negative pairs with native rewards 1 and 0.
- Eight rejected mutations across the four trajectories.
- Four fresh independently rerun native replays in
  `root-fresh-native-replay-check.json`.
- Observed native image digest
  `sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b`.
- Removal checks for the fifteen probe-owned container names observed in its log.

The Docker runtime uses the existing `python:3.12-slim` image. Recording this
observed digest does not make a mutable tag an immutable production pin. A
production challenge must approve/pin its native runtime and source bundle
separately. The controls do not prove model provenance or broad model quality.

Reproduce into a new owned state directory using the existing verifier and Docker:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -m ops.probe_native_uuidctf \
  --state state/my-uuidctf-probe --count 2
.venv/bin/python -m unittest discover -s tests -p test_public_uuidctf.py
```

Each original task uses an isolated one-CPU/two-GB Docker runtime and is torn down
after execution. Only the named first two tasks are selected from the original
3,000-index pool; difficulty and task generation are unchanged. The public solver
is a curated control, not an autoregressive model result. Subsequent gates are
model/probability/TOPLOC verification, heldout comparison and common training.
