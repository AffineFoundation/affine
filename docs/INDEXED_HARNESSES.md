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
harnesses retain their existing normalization; `None` remains `None` so the
runtime selects the historical environment policy. Explicit inactive definitions
may declare no mining indices; their indexed mapping must be empty, and no index
can be resolved from them. Resolution returns independent
copies, so changes to one runtime cannot alter the signed manifest or another
runtime's configuration.

Callers must authenticate the manifest and resolve using its definition and the
validated batch/rollout index. An uploaded harness is never authoritative.
Verification caches need the environment, index and resolved harness identity;
training and historical replay must resolve the same configuration. Heldout
suites use their own explicitly approved evaluation harness and are excluded
from mining resolution.

`project(config, selected_indices, approved_indices)` validates the complete
approved registry before selecting a rotated mining subset for the signed
manifest. It rejects unapproved indices and preserves the selected policies.
The complete replay registry and this projected mining mapping have distinct
scopes; replay admission must bind the resolved configuration for its index.

Eight tests pass, including mutable candidate isolation, per-index choices,
heldout/boolean-index refusal, missing/extra/aliased indices and silent legacy
policy changes, inactive populations, the actual legacy runtime policy and
strict subset projection. This helper is prospective: integration into the shared loop
and actual signed multi-index execution remain pending. It does not change the
active v7k source or the separately sealed Wikispeedia probe.

An additional CPU compatibility check authenticated the actual signed
sixteen-environment recovery manifest and ran the helper beside byte-identical
v7k runtime modules. Every declared policy retained its runtime configuration;
all fifteen inactive definitions refused resolution, and the active Numina
index remained authorized. The receipt is
`state/gpu-wide/root-indexed-legacy-manifest-compatibility.json`, with module and
manifest hashes. This did not load weights, execute inference or deploy.

That check first exposed a mismatch in the root/window-based prototype:
it lacked the live worker's MRCR shell-candidate dispatch and original native MCP
tool-error bridge. The isolated integration now restores them. An independent
CPU control checks exact bytes for five native/model modules, AST identity for
twelve existing harness functions, and identical normalized policies for all
sixteen declared harnesses. Eight additional integration controls pass for
index rotation, inactive definitions, registry tampering, local mining and both
auditors. These checks qualify neither GPU execution nor deployment; complete
replay/optimizer and lifecycle controls remain admission gates. The receipt is
`state/gpu-wide/root-common-policy-preservation-check.json`. Preserve the live
runtime and separately version the combined source rather than altering a
sealed checkpoint cohort.

```sh
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests \
  -p test_sample_harness.py
```
