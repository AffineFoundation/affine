The prospective Pydantic proposer reads only the original visible request and
its fenced Python schema. It parses AST without executing that code or reading
private task/grader fields. Bounded supported annotations produce a JSON object;
a same-character-length required-key mutation supplies a potential negative.
The original Pydantic grader separately determines whether either proposal works.

Actual native controls covered the sixteen original mining tasks, leaving the
sixteen heldout tasks untouched. Eight indices (0, 1, 6, 8, 10, 11, 12, 15) have
both original reward1 and reward0 controls. Six have two negative outcomes;
indices4 and13 require additional annotation support. This is a limited proposer,
not evidence that those other tasks are impossible.

Six tests cover native Pydantic validation, equal character length, aliases and
nonempty lists, malformed schema rejection, ignored hidden tool fields, and
absence of execution even when visible code contains a side effect. Reproduce
the original controls with:

```bash
python -m ops.probe_public_pydantic \
  --output state/pydantic-tasksets/public-ast-controls.json
```

These controls have no model inference, TOPLOC proof, sampled K/L batch, training
or miner credit. Equal character length does not establish equal tokenizer
length or reachable sampling probabilities. A separately signed current-model
qualification and independent native replay remain necessary before adding a
new common pipeline policy. Existing unrestricted Pydantic negative-only proof
evidence and its source version are preserved.

Root independently reran all sixteen original native controls, matching the
frozen proposer/probe/spec source hashes and every task, candidate and outcome
exactly. Eight K1L1 native controls are confirmed; this still proves no target-model
inference or common-epoch training. The root repeat is private under
`state/pydantic-tasksets/public-ast-controls-root.json`.
