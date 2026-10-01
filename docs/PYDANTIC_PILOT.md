# Original Pydantic environment pilot

Thirty-two original `justus27/pydantic-adherance-test` tasks are materialized without replacing their prompts or schema code. Indices 0–15 are for mining; 16–31 are disjoint held-out evaluation tasks. All thirty-two original schemas loaded, and original-grader negative controls on indices 0 and 16 returned zero. These are environment controls, not model performance or training evidence.

`ops/probe_pydantic_model_search.py` searches original training tasks with unrestricted target-model autoregressive sampling. An operator-signed plan fixes source membership and hashes, model weights, native environment snapshot, harness, numerical policy, and a bounded attempt budget. Held-out indices, candidate-policy substitutions, changed source bytes, and incorrect signing authorities are rejected. A separate process reloads the model and verifies actual batch tokens, full float32 vocabulary probabilities, TOPLOC, and original environment replay.

The retained-worker probe uses the unchanged approved GPU profile and actual free-VRAM checks. A found positive/negative pair is necessary before admitting this family to the common epoch pipeline. The current experiment has not established that pair, shared training, or improvement; evidence remains in the operator's `state/pydantic-tasksets` directory.

The first completed retained-worker search used original training indices 0
and 1, eight seeds per task, a 512-token output budget, and the approved
checkpoint `62aad91e5b66778d283662f750752c6e04aa5a999a08aa7ddb9c700dda211790`.
All sixteen attempts were negative. One frozen negative per task was retained
(actual outputs 448 and 446 tokens); separate fresh model verification passed
full-vocabulary float32 probabilities, TOPLOC and original replay. Independent
inspection authenticated the signed plan and sixty-six frozen source modules,
hashed both real artifacts and checked probability framing. A fresh original
environment replay on the operator machine independently matched both task
hashes, observations, terminal classifications and rewards of zero. This
provides negative-sample verification evidence; no qualifying K1/L1 batch,
common Pydantic training epoch, or improvement is claimed.
