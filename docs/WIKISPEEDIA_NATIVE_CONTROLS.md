# Original Wikispeedia native replay qualification

Four original Affine Wikispeedia tasks now have successful and unsuccessful
trajectories executed through the shared `EnvironmentSession`. The unchanged
original toolset navigates the real SNAP article graph and the unchanged
original grader scores its resulting trace. No model, TOPLOC proof, training,
remote miner epoch or chain weight submission is claimed by this prerequisite.

The successful control uses breadth-first search over the public article graph,
with the source and target supplied in the original public task. It neither
reads a hidden reward answer nor fabricates the target-reached observation.
The unsuccessful control requests an unavailable article and then finishes.
Each retained trajectory is replayed in a fresh native session and all actions,
observations, task identity and reward must match. Forged tool observations,
rewards, task identities and source hashes are rejected.

The first attempt failed before producing a rollout: the original prompt says
`wiki_click_link`, while this installed verifier exposes `click_link`. The probe
now selects the click function from the actual original tool schema, without
renaming the tool or changing its implementation. The failed v1 attempt remains
under `state/native-wikispeedia-controls-v1`. The successful v2 controls remain
separate; v3 additionally binds the full environment definition digest.

Current operator evidence is under
`state/native-wikispeedia-controls-v3-pinned-definition`:

- Four original positive/negative pairs, with native rewards 1 and 0.
- Thirty-two rejected mutations across those pairs.
- Eight separately rerun native replays in `root-fresh-native-replay-check.json`.
- All 4,610 extracted graph/article files checked byte-for-byte against the two
  recorded original SNAP archives in `root-original-resource-extraction-check.json`.

Original taskset defaults are preserved, including the seeded 4,000-pair pool,
four-to-seven-hop distance band and links-only navigation. Only the first four
original tasks are frozen for this prerequisite. This bounded control is not a
claim of broad model performance or production support.

Reproduce in a new owned state directory using the existing verifier environment:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -m ops.probe_native_wikispeedia \
  --state state/my-wikispeedia-probe --count 4
.venv/bin/python -m unittest discover -s tests -p test_native_wikispeedia.py
```

The cache contains the two original Stanford SNAP tarballs and extracted files.
The probe checks extracted file contents against those archives. A deployed
challenge must separately pin these recorded resource digests and approve its
runtime/source bundle. Native replay is not proof of which model produced the
actions. Model generation, probability/TOPLOC verification, heldout cohorts and
common training remain subsequent gates.

## Shared candidate harness prerequisite

`ops.probe_wikispeedia_candidate_harness` converts each public graph route to
the common `text-tools-v1` candidate dialect, selecting the actual original
click tool schema. At each turn, the proposed choices are the next public link
and an unavailable article; a final ordinary reply terminates unsuccessful
navigation. This uses the existing per-turn harness configuration without
changing the GPU runtime or original environment.

Four original tasks passed both deterministic native control branches through
that dialect in `state/native-wikispeedia-candidate-harness-v1`. Six helper tests
also check directed graph bounds, ambiguous tool schemas, false negative
articles, resource tampering, and forged native outcomes. These are curated
public proposals: model selection, token/context budgets, TOPLOC and probability
verification, remote resource binding and training still require a separate
approved model experiment.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -m ops.probe_wikispeedia_candidate_harness \
  --qualified-state state/native-wikispeedia-controls-v3-pinned-definition \
  --state state/my-wikispeedia-candidate-harness
```
