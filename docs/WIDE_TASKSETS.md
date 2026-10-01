# Prospective wider tasksets

`ops.materialize_tasksets` snapshots original provider tasks into a new signed
identity. It never relabels the old four-task results: training indices0–15 and
fixed held-out indices16–31 are disjoint in each new32-task source. The active
six-family GPU service remains on its original immutable source and dataset until
a completed epoch boundary permits deliberate migration to a new source stage.

The operator has materialized Verbatim, Math, RGym, When2Call, IFEval, Oolong,
SciText, Logic and Unscramble. Oolong's full dataset preparation exceeded local
free space; the retained remote pod completed it and returned only the1.46MB
snapshot. Historical archive and checkpoint evidence were preserved.

Example (in a scoped process with sufficient provider-cache disk space):

```sh
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m ops.materialize_tasksets \
  --config state/gpu-continuous-config.json \
  --sources affine_math affine_rgym \
  --count 32 --training-count 16 --output state/wide-tasksets-next
```

Taskset overrides can select provider-specific sample budgets.
Each worker is bounded by timeout; failures and exact provider errors remain in
the materialization report. Successful snapshots report byteSHA and sourceSHA.

The optional signed `indices_per_environment_per_epoch` rotates within each
approved source's training indices across its group cycles. It avoids spending
all three submission slots on the first source's first indices. It does not
change historical default behavior or select held-out tasks.

Public-prompt proposal controls are distinct from model inference. Logic has
native positive/negative controls on all sixteen training tasks; the current
SciText and Unscramble proposal heuristics cover only two of sixteen. Curated
When2Call and IFEval controls remain index0-specific. The original Oolong
frequency-count algorithm handles one question type and must qualify each new
task before claims of successful search. A wide snapshot alone establishes no
successful mining, training, model quality improvement or full source coverage.
Before migration, candidate class probabilities must be measured with the
approved model: a valid native answer can still have near-zero sampling chance.

## Native tool-error boundary

The first wider miner produced three independently accepted batches, but its
full held-out role aborted on a model-generated unknown tool name. Calling the
private MCP tool manager bypassed the native server’s error-result boundary.
The epoch’s immutable manifest, frozen uploads and successful audits are retained;
it is untrained and its evaluation failure is explicit. No failed job is counted
as a complete baseline.

`subnet/native_tool_errors.py` provides the prospective
`native-mcp-toolerror-observation-v1` policy. Actual native MCP dispatch controls
confirmed identical unknown-tool error text and preserved successful results;
infrastructure exceptions still propagate. An original When2Call control then
finished with the original grader’s negative outcome. This control has no model
proof or training claim.

`ops/patches/native-mcp-tool-errors.patch` adds opt-in wiring, a separate environment
version, helper-byte fingerprinting and an MCP package pin. Apply it only inside
an isolated prospective source stage, refresh signed specs and approve a new
epoch. The live repository adapter remains unchanged so existing CPU pilot
source pins remain valid. Never rewrite a running manifest or substitute a new
worker’s source hashes into an old report.
