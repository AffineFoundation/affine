# Budget forcing: a planned change to the scoring

**Status: planned. The rule is off.** The box scores as before while `params.budget_forcing_tokens` is absent from `https://affine.io/mining.json`. The turn-on is not decided yet: it needs a separate decision. This document promises no turn-on and no date.

<!-- EARLIEST-TURN-ON: the one place in this repository that states the date or the round. Edit only the next line. -->
**Earliest turn-on (planned):** no earlier than 7 days after the announcement of this change in the Affine Discord, at a round boundary.

When the rule is on, `https://affine.io/llms.txt` section 6 has the full procedure with the values in force. The box writes that text from its params. If this document and llms.txt differ, llms.txt is right.

## In short

- **Today:** a reply that reaches the token cap gets the outcome code `C`. The grader never reads it. It counts as not passed.
- **With the rule:** the box closes such a reply, the model writes a few more tokens, and the grader reads the whole text. A pass gets the new outcome code `p`.
- `p` is a pass. It is also still a cap hit.
- mining.json gets one new parameter, `params.budget_forcing_tokens` (the planned value is 32), and one new entry, `contract.outcome_codes.token_cap_pass`.
- **Update your copy of the code before a turn-on** (section 6). The old miner tool stops on a mining.json that has the new parameter.
- Your batch file, its rules and the upload do not change.

## Words used here

- **Box**: the validator's GPU machine. It trains and scores every round.
- **Head (M)**: the current model. **Null M**: the head with no LoRA, scored as the baseline.
- **Round**: one run of the box: it trains one LoRA for each batch, scores each LoRA and null M on the same tasks, and decides the winner.
- **Token cap**: `max_output_tokens` of the params, 65,536 tokens today. A reply that reaches it is cut there.
- **Rollout**: one try of one model at one task.
- **Chunk**: a group of tasks that the box sends to the models at one time. A full round has several chunks.
- **Look**: a point in the scoring of a round at which the box judges the scores so far. A full round has several looks. The last look is at the end of the tasks.
- **Stop time**: the time at which the box stops the scoring of a round, some minutes before the time limit of the round (llms.txt section 3).
- **Outcome code**: one character for one rollout. `P` pass, `F` fail, `C` hit the token cap, `E` an error of the box, `-` not run.
- **Forced close**: the text that the box puts after a cut reply, so that the model gives its final answer.
- **Greedy**: the model takes the most likely token at each step (temperature 0). It makes no random draw.

## 1. Why

Today a reply that reaches the token cap counts as not passed, and the grader never reads it. The reasoning in such a reply can be right: the model only did not write its answer before the cap. The forced close lets the model write that answer. Then the score counts a right answer as right, for null M and for each LoRA in the same way.

## 2. What the box does when the rule is on

The rule is on when `params.budget_forcing_tokens` is above 0. The texts below show the planned value, 32.

1. The first request of a rollout is the same as today, with one more key: `"return_prompt_text": true`. The server then returns `prompt_text`: the prompt as it made it from the chat template.
2. A reply that ended by itself (`finish_reason` `stop`) is graded as today. Nothing changes for it.
3. A reply that reached the token cap (`finish_reason` `length`) gets one more request. The text to continue has these parts, in this order, with nothing between them:
   - the `prompt_text` of the first reply;
   - the text of the reply;
   - only when that text has no `</think>`: a line feed and `</think>`;
   - two line feeds and `Final answer: \boxed{`.
4. The box sends that text in one POST to `/v1/completions` on the same server, for the same model:

```json
{
 "model": "<null or the lora_hash>",
 "prompt": "<the text to continue>",
 "max_tokens": 32,
 "temperature": 0.0,
 "skip_special_tokens": false,
 "add_special_tokens": false
}
```

5. `max_tokens` is the value of `budget_forcing_tokens`: the model writes at most that many tokens. This request is greedy. It has no seed. A final `<|im_end|>` is removed.
6. The grader grades one text: the reply, then the forced close, then the new tokens.
7. When the grader passes that text, the outcome is `p`. With each other grade the outcome is `C`, as today.

What happens when the second request fails:

- The second request has the tries and the time limits of the first request.
- When it has no usable reply, the outcome is `C`.
- When the stop time of the round passes during the second request, the rollout has no reply. Then the box does not use that chunk, as today for a first request that the stop time cut. The round is then judged as each round that the stop time cuts (llms.txt section 3). When the first look is complete, the win rule runs on the tasks that the round scored before that chunk. When it is not, each LoRA is `not_scored`: you are not charged.
- When the server returns no `prompt_text` with a reply at the cap, the box does not score that chunk, and it tries the chunk again. When each try fails, the round has no result: no LoRA is judged, and nobody is charged. This is an error of the box, not an outcome of a model. The box does not write `C` in place of the rule.

The time: the second requests run inside the scoring time of the round. So the scoring of a round with the rule on can take more time. The second requests cannot run past the stop time: a second request gets only the time that is left to the stop time.

A round is scored with one set of params. The validator takes no new scoring parameter while a round is open. So no round is scored partly with the rule.

## 3. How `p` counts

| Code | Meaning | A pass for delta and z | A cap hit for the cap-hit rule |
|---|---|---|---|
| `P` | The reply ended and it is right. | yes | no |
| `F` | The reply ended and it is wrong. | no | no |
| `C` | The reply hit the token cap. With the rule: it is also not right after the forced close, or the second request had no usable reply. | no | yes |
| `p` | The reply hit the token cap, and it is right after the forced close. | yes | yes |

- **Delta and z**: the pass rate of a model on a task is the share of its samples with `P` or `p`.
- **The cap-hit rule**: the count of a model is its count of `C` and `p`. The limit does not change: max(`cap_hit_floor`, the cap-hit share of null M + `cap_hit_margin`). A LoRA whose share is above that limit is still not eligible, even when its forced closes pass.
- **The IRT snapshot** (the task difficulties that the draw uses) counts `p` as a pass.
- `E` and `-` do not change.

## 4. What changes in the public files

All of this is from the first publish after a turn-on. Before it, the files have none of it.

- **mining.json**
  - `params.budget_forcing_tokens` appears, with the value in force. While the rule is off, the key is absent (it is not written as 0).
  - `contract.outcome_codes` gets the entry `"token_cap_pass": "p"`.
  - `params_hash` changes, because the params change.
- **llms.txt**: section 6 gets the forced close and the code `p`, and section 11 gets the row `budget_forcing_tokens`. The box writes this by itself.
- **Round files** (`public/rounds/ROUND_ID/round.json`): the outcome strings in `null_outcomes` and `winner_outcomes` can hold `p`. A round that closed before a turn-on has no `p`.

## 5. What does not change

- The batch file: its format, each rule of the file check, and the rule codes.
- An output in your file that has exactly `max_output_tokens` tokens stopped at the cap, and its official grade is still 0. The forced close is a step of the round scoring only. The file check does not make it.
- The upload: the sealed link and the HTTP PUT.
- The training: each training value of the recipe, and what you may set under `train`. Only the digest of the GPU code changes (section 7, step 2).
- The first request of a rollout, but for the key `return_prompt_text`: the same seed, the same settings, the same token cap.
- The other numbers of the win rule: `z_bar`, `cap_hit_floor`, `cap_hit_margin`, `min_scored_tasks`, the looks.

## 6. What you must do

### Update your copy of the code

Do this before a turn-on. You can do it at once: the new tool also reads the mining.json of today.

```
cd affine
git checkout v2
git pull
```

If your install is editable (`pip install -e` or `uv pip install -e`, as llms.txt and `REPLICATE.md` say), that is all: the update adds no package. If you installed without `-e`, install again.

### Check which code you have

Run this in the Python environment in which you run the miner tool:

```
python -P -c "from sn120_v2.params import Params; print(hasattr(Params, 'budget_forcing_tokens'))"
```

Keep the option `-P`. With it, Python reads the installed copy of the code, the copy that `miner/affine_miner.py` uses. Without it, in the directory `v2/`, Python reads the files of the checkout, and the check can print `True` while the installed copy is old. (`python -m sn120_v2.replicate`, run in `v2/`, also reads the files of the checkout.)

- `True`: you have the new code. It reads mining.json with the rule off and with the rule on.
- `False`: you have the old code. Update it. If you pulled already, install again.
- The error `No module named 'sn120_v2'`: no copy is installed in this environment. Install it.

The miner tool has no version number. This check is the way to tell.

### What the old code does after a turn-on

Every command of `v2/miner/affine_miner.py` (`bounds`, `check`, `upload`, `reproduce`) and every command of `python -m sn120_v2.replicate` (`compare`, `fetch`, `run`) reads the params of mining.json first. The old code does not know the new field. It prints this and stops with exit code 2:

```
{
 "error": "ContractError",
 "reason": "Params: unknown field(s) ['budget_forcing_tokens'] for Params"
}
```

- `upload` then sends nothing. Your file in the store stays as it was.
- An upload that does not use the tool (a plain HTTP PUT to your opened link) works as before.
- The old code also does not know the code `p`. Its check of outcome strings refuses a string that holds `p`, and where it counts passes without that check, it does not count `p`.

### If you have your own scripts

- Accept the key `params.budget_forcing_tokens` and the entry `contract.outcome_codes.token_cap_pass` in mining.json.
- Accept the code `p` in outcome strings. Count it as a pass, and as a cap hit.

### See if the rule is on

```
curl -s https://affine.io/mining.json | python -c "import json, sys; print(json.load(sys.stdin)['params'].get('budget_forcing_tokens', 'absent: the rule is off'))"
```

A number above 0 means that the rule is on, with that many tokens.

## 7. The planned steps

1. **The new code is in the branch `v2`**, with this document. From then on you can update.
2. **The box takes the new code, with the rule still off.** This step is planned. Its date is not set yet. It can come before or after step 1. It changes no score. mining.json then shows a new `params.recipe.code_digest`, so `recipe_id` and `params_hash` change. The old code still reads everything.
3. **The turn-on.** It is planned, and it is not decided yet. The earliest time is the line at the top of this document. A turn-on can only be between two rounds. The rule then counts from the first round that opens after it.

`python -m sn120_v2.gpu.plan` prints the code digest of your copy. While it is not `params.recipe.code_digest` of mining.json, `python -m sn120_v2.replicate compare` lists the difference `code`. That is expected in two cases: you updated, and the box did not take the new code yet; or the box took the new code, and you did not update. No command stops because of it.

## 8. At home, with `python -m sn120_v2.replicate`

- `run` scores with the code of the box. So with the rule on, the new code makes the forced close on your machine too, with the value of mining.json.
- The forced close needs a vLLM server that returns `prompt_text`. The `gpu` extra pins vLLM 0.32.0. A test run of this scoring code on a rented machine with 8 GPUs (2026-11-15; not the box) got `prompt_text` for each reply at the cap with that version.
- The report counts `p` as section 3 says: `score.pass_rate` and `score.null_pass_rate` count `P` and `p`; `score.cap_rate` and `score.null_cap_rate` count `C` and `p`.
- The second request is greedy, but its prompt holds the reply of the first request, and that text is not the same from run to run. So a `p` is equal to the box only in distribution, as each other outcome code (`REPLICATE.md` section 5).
