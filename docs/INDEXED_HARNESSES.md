# Signed per-index harness selection

The prospective `indexed-harness-v1` wrapper lets an environment declare distinct
approved harnesses for its mining indices. This supports public route choices
without adding environment-specific branches to mining, verification or training.

```json
{
  "version": "indexed-harness-v1",
  "by_index": {
    "0": {"version": "text-tools-window-v1", "policy": "candidates",
          "candidates": ["public choice A", "public choice B"],
          "max_output_tokens": 256, "temperature": 4, "top_p": 1}
  }
}
```

`subnet/sample_harness.py` validates exact coverage of the containing definition's
signed mining indices. Keys use their canonical decimal representation. Extra,
missing, aliased and nested mappings are rejected; there is no fallback. Ordinary
harnesses retain their existing normalization. Resolution returns independent
copies, so changes to one runtime cannot alter the signed manifest or another
runtime's configuration.

Callers must authenticate the manifest and resolve using its definition and the
validated batch/rollout index. An uploaded harness is never authoritative.
Verification caches need the environment, index and resolved harness identity;
training and historical replay must resolve the same configuration. Heldout
suites use their own explicitly approved evaluation harness and are excluded
from mining resolution.

Five tests pass, including mutable candidate isolation, per-index choices,
heldout/boolean-index refusal, missing/extra/aliased indices and silent legacy
policy changes. This helper is prospective: integration into the shared loop
and actual signed multi-index execution remain pending. It does not change the
active v7k source or the separately sealed Wikispeedia probe.

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests \
  -p test_sample_harness.py
```
