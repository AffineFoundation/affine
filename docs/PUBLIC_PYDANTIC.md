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

A separate model qualification uses `ops/probe_pydantic_public_model_search.py`. The signed plan pins the generator, complete worker source inventory, original task snapshot, checkpoint and strict numerical profile. For each mining index it reconstructs two proposals from that task's public reset messages, records both token lengths and actual candidate sampling probabilities, then performs bounded candidate sampling. This avoids pooling unrelated short and long schemas. The model policy is explicitly a curated public-AST candidate policy; equal character lengths do not imply equal token lengths or balanced class probabilities. A separate process reloads the same approved weights and reconstructs public candidates before verifying every retained full-logit/TOPLOC artifact and original native outcome. No optimizer, miner credit or chain operation occurs in this qualification.

The first run is `state/pydantic-public-model-control/1790874356`, original mining indices 0 and 1, sixteen seed attempts each, checkpoint `aaac517b5a1a39f3fdd78cf2c73adbad62f9f8b94f00793a95f7f8bcf6d3739d`. Its worker uses the continuous controller's adapter source in a separate immutable archive. Generation and a separate fresh full-logit/TOPLOC/native verifier completed, but both indices yielded only positive samples: no K1/L1 pair. Actual candidate negative probabilities were about 0.000509 and 0.000106, with unequal token lengths (49/52 and 86/87). This preserved v1 result motivates a new public-only candidate policy, rather than claiming sampled K/L availability or common training.
