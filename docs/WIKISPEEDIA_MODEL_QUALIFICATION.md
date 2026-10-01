# Original Wikispeedia model qualification

The prospective probe `ops/probe_wikispeedia_model_search.py` samples the four
mining indices of the native-qualified original twenty-task snapshot. Sixteen
heldouts remain excluded from this search. It uses the common GPU runtime,
full-vocabulary float32 log probabilities, TOPLOC fingerprints, and the
separately versioned `text-tools-window-v1` harness.

Before importing the original provider, a fresh process must set the approved
private `WIKISPEEDIA_CACHE_DIR` and verify both SNAP archives and all 4,610
extracted graph/article files. The common adapter then resets each original task
and checks its public messages and task hash. Candidate tool calls are recomputed
from the public graph and original tool schema; replacing them with a hidden
answer fails admission. A separate resource guard must bind exact source,
resource and operator-file membership before entering this probe. The signed
plan also binds the checkpoint, numerical policy and all source digests.

Each task has its own public navigation choices and bounded turn count. The
model selects among these choices by the existing sum-log-probability policy;
this is curated candidate sampling, not autonomous navigation or an unbiased
sample of the model's unrestricted output distribution. Variable candidate
lengths are allowed and retain that policy's length bias. Complete successful
and unsuccessful trajectories are retained when found; failure to find both is
reported rather than counted as a qualifying K1/L1 batch.

Fresh verification runs as a separate process, reloads the pinned model and
checks frozen batch bytes, checkpoint, index, per-task harness, observed K/L
counts, full log probabilities, TOPLOC and every original native observation.
No optimizer, score or chain-weight operation exists in the probe. The retained
GPU must be idle, and the ongoing wide empty-epoch and normal Numina/Pydantic
training recovery have priority.

Five focused contract controls pass. Root also ran the actual native preflight
in a fresh process over all four mining tasks and original resources, and
confirmed that a deliberately replaced public choice is rejected. Evidence:
`state/native-wikispeedia-model-contract-v1/root-native-model-contract-check.json`.
These checks do not establish GPU proof acceptance or training coverage.

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests \
  -p test_wikispeedia_model_contract.py
```

After a complete resource and model plan is separately approved, run generation
and verification in separate fresh guarded processes. Do not reuse a provider
already imported with a different cache or launch alongside the wide worker.
