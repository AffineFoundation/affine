# Public-input candidate controls

`subnet.public_candidates.propose(env_id, messages)` produces bounded, deterministic
candidate text using only the visible user messages. It receives no task object,
gold answer, grading result, private model state, or held-out record. Supported
heuristics currently include rocket lift-off calculations, heat-rate unit
conversions, Campsite constraint solving, and bounded text-ordering controls for
visible ages, dated events, procedural dependencies and recognized narratives. Other prompts return an empty pool. This is deliberately partial
coverage, not a general solver for an entire environment family.

The approved `candidates` harness can consume a proposal pool after it is included
in a new immutable challenge. The model scores/samples from that pool and produces
the ordinary token/log-probability/TOPLOC trajectory. Original environment grading
and independent model verification remain required. A candidate's presence does
not establish success. These are curated, off-policy controls; held-out model
performance must use a separate autoregressive policy and disjoint tasks.

Run the original grader controls without performing model inference or training:

```sh
.venv/bin/python -m ops.probe_public_candidates
.venv/bin/python -m unittest discover -s tests -p test_public_candidates.py
```

The ignored evidence file records public messages, original task/source hashes,
exact proposed texts, native grades, and generator/probe source hashes. On the
existing fixed four-task snapshots, positive and negative proposals were actually
graded for SciText, Logic, and Unscramble indices 0 and 1. The Science heat-transfer
prompt's proposed unit/precision variants all graded negative, and its second
prompt is unsupported. These controls do not establish mined proof batches,
training updates, or broader taskset coverage. Only a completed signed epoch can
establish those later milestones.

Never rewrite a running manifest or relabel an old evaluation curve when adding a
pool. Stage the new source/task artifacts, approve a prospective challenge, and
record a new baseline wherever the taskset or evaluation protocol changes.

The prospective v2 ordering controls were checked against all sixteen Unscramble
training tasks in the wider taskset, without reading held-out tasks or gold fields.
Ten tasks have at least one native positive and one negative proposal; five have
no supported proposal and one has only negatives. This establishes public-input
control coverage, not model sampling, proofs or training on those tasks. The
active first wider worker retains its previously approved pools.

Reproduce the training-only native controls with:

```sh
.venv/bin/python -m ops.probe_public_candidates \
  --spec-directory state/wide-tasksets --spec-pattern '{source}.spec.json' \
  --sources affine_unscramble --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --output state/multi-environment/public-solver-research/unscramble-wide-training-native-grades-v2-final.json
```
