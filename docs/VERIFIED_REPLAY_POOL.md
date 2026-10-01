# Authenticated, balanced historical replay

`subnet/verified_replay_pool.py` admits historical positive/negative pairs only
from operator-authenticated FULL audits and exact frozen submission bytes. It
binds the historical manifest, checkpoint file map, source descriptor, task,
environment, harness and selected rollout hashes. Current model architecture,
configuration, tokenizer files and token geometry must remain compatible.
Sampled audits and auxiliary model roles are excluded from this pilot.

Version 3 requires an explicit held-out registry for **every** current
environment. A missing entry is an error; an explicit empty list has a distinct
meaning. Mining and held-out indices must be unique, integer, bounded by the
signed task count and disjoint. These checks also run when a pool is built, so
signed intermediate admission records cannot bypass the current registry gate.

Selection groups by the authenticated environment ID. The adapter is checked
separately. This matters because all twelve original wide-suite environments use
`prime_v1`: grouping by adapter would collapse them into one family. Selection
takes one available pair from each environment before taking another, with
signed pair/reuse limits and proposed reuse increments. The controller must
persist those increments only with the corresponding completed training job.

This module authenticates historical evidence; it does **not** perform fresh
model or environment execution. Historical log probabilities are never a
current-model reference. A trainer must recompute its immutable reference under
the current approved checkpoint, bind the selected pool and pair targets in its
job, and record optimizer attribution. The preparation helper does not launch
training, migrate a service or submit blockchain weights.

The operator prepared a version-2 pool from real signed manifests, FULL audits
and frozen ZIPs: nine distinct compatible environment/index pairs, selected
across nine environments, using checkpoint `088fdf925b4c0063e9113b1a99aab49ba9f6b749fd77939d1a0cfba3fd474b89`.
Those inputs already declared the complete twelve-environment held-out map.
The historical version-2 evidence remains preserved. Version-3 preparation and
actual pooled optimizer execution are separate gates; the historical preparation
is not a claim that either has run.

Fifteen synthetic signed-lineage controls cover forged audits, changed frozen
bytes and pair hashes, incompatible model/tokenizer bindings, sampled audits,
held-out exclusions and registry omissions, index types/bounds/duplicates,
round-robin selection across twelve environments sharing one adapter, reuse
caps, mutable views and signed pool digests. They are conformance checks rather
than model experiments.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_verified_replay_pool.py'
```


A new version-3 pool has since been prepared and authenticated for current
checkpoint `d8e047f13278692e0baa0df213b4d5566318e582bb74e47754a63a91d6125383`.
It retains nine eligible environment/index pairs across the same nine families,
with the complete twelve-environment held-out map. Its pool SHA256 is
`91f561155a08458161efb914de1718bdf6998f4476e70ea6b78f26f33137b5b7`;
private evidence is in `state/verified-replay-pool/currentd8-v3`. Current-model
reference recomputation and actual pooled optimization are still separate,
uncompleted qualification gates in this report.

`subnet/replay_training.py` is the prospective current-job admission bridge. It
checks the complete live environment/held-out inventory and runtime/checkpoint
compatibility, verifies historical pairs again under the current model, then
merges one pair per family with fresh epoch data preferred. Least-used historical
targets rotate within a family; a fresh contribution shadows its historical
candidate and must not increase historical reuse. Ten controls cover admission,
current recomputation, rejected native verification, shadowing and rotation.
The private controller stage additionally requires enough optimizer steps for
every merged family and journals only consumed replay targets after checkpoint
publication. These source contracts do not establish a completed live replay
epoch or improved held-out performance.
