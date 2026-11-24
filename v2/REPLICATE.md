# Train and score your batch at home

`python -m sn120_v2.replicate` takes your `batch.json`, checks it as the validator does, trains the LoRA from the current head M, scores the LoRA and null M on the tasks of a round, and judges the result with the win rule.

It has no training code and no scoring code of its own. It runs the code of the validator's GPU machine (the **box**) on your GPUs. Where your machine cannot equal the box, the tool changes that one value, and the report names each change under `differences`.

This document says what you need, the commands, what is the same as on the box, what differs, what you can repeat exactly, and what has been verified.

Nothing here touches the subnet. The tool reads public files. It writes only below the work directory you name. It uploads nothing and holds no key.

## Words used here

- **Box**: the validator's GPU machine. It trains and scores every round.
- **Head (M)**: the current model. **Null M**: the head with no LoRA, scored as the baseline.
- **mining.json**: the public contract file, `https://affine.io/mining.json`. It holds the params, the head with the sha256 of each file, and the open round.
- **Draw**: the tasks of a round, in the order in which they are scored, each with a weight.
- **Install digest**: a hash of the Python version and of every installed package with its version. **Code digest**: a hash of the files `sn120_v2/gpu/*.py`. The params pin both for the box.

## 1. What you need

- Linux with the NVIDIA driver (`nvidia-smi` works) and at least 1 GPU. The tool uses GPUs 0 to N-1, where N is `--gpus`.
- Python 3.12 and `uv`.
- Free local disk for the base files of the head (18,849,863,399 bytes for the head of 2026-11-14) and for the caches of vLLM.
- Free ports 8100 to 8100+N-1 on `127.0.0.1`. One vLLM server listens on each.

Which GPU is large enough is **not measured**. The code has run only on the GPUs of the box. Each GPU loads the whole model in `bfloat16`, serves a context of `max_context_tokens` tokens, and in training holds the longest row of your file.

Install, from a checkout of the branch `v2`:

```
git clone https://github.com/AffineFoundation/affine.git && cd affine
git checkout v2
cd v2
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[gpu,envs,miner]"
```

This installs `boto3` with the `miner` extra as well as the tokenizer, chat-template and grading dependencies.

Run every command below from `v2/`, inside the virtual environment.

Every command takes `--mining-json`: the address `https://affine.io/mining.json`, or a file you saved from it.

## 2. The commands

### compare: what differs between your machine and the box

```
python -m sn120_v2.replicate compare --mining-json https://affine.io/mining.json --gpus 1
```

It starts nothing and needs no GPU. It prints one JSON object:

- `box`: `params_hash`, `recipe_id`, the pinned `image_digests`, the pinned `code_digest`, `n_gpus`.
- `here`: your `image_digest` (the install digest), your `code_digest`, `gpus_found` (the GPUs that `nvidia-smi` lists; `null` when it does not answer), `n_gpus` (the value of `--gpus`).
- `differences`: a list of `{"what", "text"}`. Section 5 explains each `what`.

### fetch: the base files of the head

```
python -m sn120_v2.replicate fetch --mining-json https://affine.io/mining.json --work /data/affine
```

It downloads each base file of the head through the signed links of the links file that mining.json names (`contract.read_links`), and checks the size and the sha256 of each file against the head manifest in mining.json. The links expire; the tool reads the links file anew at each start. If a download fails, run the command again: complete files are kept. `--read-links FILE_OR_URL` names another copy of the links file.

If you have the files already, name their directory with `--from`. Then nothing is downloaded:

```
python -m sn120_v2.replicate fetch --mining-json https://affine.io/mining.json --work /data/affine --from /path/to/model
```

Each file is linked (or copied, when a link is not possible) and its sha256 is checked. Other files in that directory are ignored. While the head is the genesis model, llms.txt section 1 names the Hugging Face repository and revision that hold these files.

The files are collected in `<work>/fetch/<base_hash>/` and moved to `<work>/cache/<base_hash>/` when the set is complete. The command prints `{"base_hash", "dir", "files", "fetched"}`. Use the printed `dir` as the head tokenizer directory. For `affine_miner.py check` and `reproduce`, `--tokenizer <work>/cache` also works: when it contains the current contract's `<base_hash>/` directory, the tool resolves that inner directory and logs the selected path.

**Do not add or change a file in `<work>/cache/`.** The box code verifies a directory there against its id and deletes a directory that does not match. `run` does the same. That is why `fetch` exists: a download tool that writes its own files into the model directory would make the set invalid.

When the head changes (a winner is folded), `base_hash` changes. Run `fetch` again. The run removes the directory of the old base from `<work>/cache/`.

### The current head's missing processor files

Some already-sealed heads omit `preprocessor_config.json`, `processor_config.json` and `video_preprocessor_config.json`. Their `base_files` and `base_hash` must not change. Do **not** put the missing files into `<work>/cache/` or edit mining.json.

For local **evaluation only**, make a separate serving directory, leaving the verified cache untouched:

```bash
export HEAD=/data/affine/cache/<base_hash>
export SERVING=/data/affine/serving/<base_hash>
mkdir -p "$SERVING"
cp -al "$HEAD"/. "$SERVING"/
for name in preprocessor_config.json processor_config.json video_preprocessor_config.json; do
  if [ ! -f "$HEAD/$name" ]; then
    curl -fL "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/resolve/2367e865d009c13ac81713a2878291d33ab28177/$name" -o "$SERVING/$name"
  fi
done
```

The head has a **VL-shaped architecture**. Disable image/video inputs in standalone vLLM serving, evaluation and miner sampling with this exact option:

```
--limit-mm-per-prompt '{"image":0,"video":0}'
```

The official evaluator and `miner/generator_search.py` already pass that option; use it for standalone serving and any other miner sampling server too. This serving copy is an evaluation workaround, not a new head and not a substitute for its sealed manifest. If another small processor or tokenizer file from the parent or that pinned genesis revision is needed, add it only to the separate serving copy. Future folds/published heads carry these small files automatically. `run` still reads the original verified head; the standalone serving copy does not alter its canonical training or head hashes.

### run: check, train, score, judge

```
python -m sn120_v2.replicate run --mining-json https://affine.io/mining.json --hotkey <ss58> --work /data/affine --gpus 1 batch.json
```

`--hotkey` is the hotkey named in the file. No signature is needed.

Which round is repeated:

| Argument | The seed and the draw |
|---|---|
| none | The round that mining.json shows as open. If none is open, the last closed round in `history`. |
| `--round ROUND_ID` | That round of mining.json (`round` or `history`). |
| `--round-file FILE_OR_URL` | The public file of a closed round, `public/rounds/ROUND_ID/round.json`. Its seed and snapshot are used, and the report gets `box_round` (section 4). |
| `--round-seed SEED --irt-snapshot FILE_OR_URL` | A seed of your choice with a snapshot you name. No real round. |

The IRT snapshot comes from the public address `public/irt/SNAPSHOT_ID.json`, or from `--irt-snapshot`. The tool checks that the id of the snapshot is the hash of its content, and that it is the snapshot the round names.

Other arguments:

- `--gpus N`: use GPUs 0 to N-1. The default is `n_gpus` of the params, the count of the box. The command stops at once if `nvidia-smi` lists fewer.
- `--pool auto|current|v2.0`: `auto` (the default) uses the archived v2.0 pool when the recorded draw names a task that is not in the current pool. `current` refuses such a draw; `v2.0` forces the archive. The archived pool is `sn120_v2/envs/archive/math_pool_v2.jsonl`, byte-for-byte the original pool. See [POOL_CHANGES.md](POOL_CHANGES.md).
- `--first-tasks N`: score only the first N tasks of the draw. These are the tasks the box scores first, in the same order with the same weights. It is not a new draw of N tasks. The looks are shares of N. With fewer than `min_scored_tasks` tasks no LoRA can win.
- `--round-clock`: keep the time limits of a round (section 5, `round-clock`).
- `--strict`: stop unless your machine has the pinned install, the pinned code and the GPU count of the box.
- `--out FILE`: also write the report to this file.

Progress goes to the standard error stream. The report is one JSON object on the standard output.

Exit code 0: a report. Exit code 1: the file breaks a rule; the output is `{"valid": false, "rule", "reason"}`. Exit code 2: the run could not be done; the output is `{"error", "reason"}`.

Where things are, below `--work`:

- `cache/<base_hash>/`: the base files.
- `store/pending/rounds/local/local-l0/loras/000/`: the LoRA of the last run (`adapter_config.json`, `adapter_model.safetensors`, `train_report.json`). The next run overwrites it.
- `logs/vllm-<gpu>.log`: the output of each vLLM server. `logs/train-<job>.log`: the output of a training job that failed. `logs/phases.jsonl`: times and sizes of each phase.

## 3. Building the batch file

With 2 rollouts per prompt, one official-head sample of 960 prompts produced 97 mixed-reward groups, a yield of about 10%; plan on about 5–8 times as many prompts as the mixed-reward groups you need, and the file needs 80 mixed groups.

A mixed-reward group has rollouts with different rewards; its nonzero advantages train. A file in which no rollout trains is invalid. Check the actual file and its token/reward rules before uploading:

```
python miner/affine_miner.py check --mining-json mining.json --hotkey <ss58> --tokenizer /data/affine/cache --grade batch.json
```

Only an output with exactly `params.max_output_tokens` tokens (currently 65,536) is an official cap-hit. A shorter local limit such as 8192 must not be labeled an official cap-hit: below 65,536, a correct boxed answer can grade 1.0 rather than 0. At exactly the official cap the grade is 0 even if the answer is correct. The checker's reward-mismatch message explains both cases.

For direct grading, `sn120_v2.envs.math_grader.grade(task, output, capped)` accepts a `sn120_v2.envs.math_pool.Task` or a mapping with string `task_id`, `slice`, `problem` and `answer` fields. A task JSON object therefore needs no separate conversion; malformed mappings and unrelated objects raise a clear `TypeError`. Set `capped=True` only for a stop at the official output cap, not for a shorter local serving cap.

## 4. The report of `run`


| Field | Meaning |
|---|---|
| `valid`, `format`, `batch_sha256`, `n_prompts`, `n_rollouts`, `dropped_groups` | The result of the file check. |
| `rules_run` | The rule families that ran. Always `format`, `tokens` and `grader`: the validator's full check. |
| `train_applied` | The value the trainer used for each key that a miner may set. |
| `head_version_id`, `base_hash` | The head of mining.json that was trained from and scored. |
| `round` | `round_id` (`null` for a seed of your choice), `round_seed`, `irt_snapshot_id`, `n_tasks` (the size of the draw), `scored_tasks` (smaller with `--first-tasks`). |
| `box` | `params_hash`, `recipe_id` and `n_gpus` of mining.json. |
| `here` | `image_digest`, `code_digest` and `n_gpus` of this run. |
| `differences` | Every difference between this run and a round on the box (section 5). |
| `status`, `reason` | `winner`, `lost`, `no_decision`, `skip`, `not_trained` or `not_scored`, with the same meaning as in a real round, and the reason in words. |
| `eliminated_at_look` | The look at which the LoRA was dropped, or `null`. |
| `score` | `n_tasks`, `delta`, `se`, `z`, `pass_rate`, `null_pass_rate`, `cap_rate`, `null_cap_rate`, `cover`, `eligible`, `reasons`. `null` when nothing was scored. |
| `null_outcomes`, `outcomes` | One string per sample index: one outcome code per task, in the order of the draw. Null M has `null_samples_per_task` strings (defaulting to `samples_per_task`); your LoRA has `samples_per_task` strings. Scoring compares each LoRA pass with the mean of the king's samples on that task. |
| `lora` | `lora_hash`, `dir` (the LoRA files on your disk), `train_report`. Absent when the training failed. |
| `phase_s` | Wall seconds: `training` and `evaluation`. |
| `stopped_at_limit` | `true` only with `--round-clock`, when the time limit of the round ended the scoring. |
| `box_round` | Only with `--round-file`: `round_id`, `same_head` (the round ran on the head of this run), `same_draw` (the draw rebuilt here equals the draw in the file, over the tasks scored here), and `null_m`. |

`box_round.null_m` is present when the head and the draw are the same and one box scored the round. It compares null M here with null M on the box, task by task: `rollouts_compared`, `same_outcome`, `passed_here`, `passed_on_box`. This is the measurement of how close your machine is to the box. It is the only one: the box publishes the outcomes of null M and of the winner, not of a LoRA that lost.

## 5. What is the same as on the box, and what differs

### The same

`run` builds these objects and calls them. No line of them is copied or changed:

| Step | Code of the box |
|---|---|
| File check | `sn120_v2/admission.py` `check_batch`, with the tokenizer of the head, the public task pool and the official grader in worker processes: at most 8 workers and 20 seconds for one row, as on the validator. |
| Training | `sn120_v2/gpu/trainer.py` `LoraTrainer` starts `python -m sn120_v2.gpu.train_job`. The recipe is the one of mining.json with your `train` values. The seed is `train_seed(round_seed, batch_sha256)`. |
| Scoring | `sn120_v2/gpu/evaluator.py` `VllmEvaluator`: vLLM servers with the arguments of the box, the request body of the box, the seed `rollout_seed(round_seed, task_id, sample)` for null M and for the LoRA, and the official grader. |
| The round | `sn120_v2/worker.py` `LocalRunner`: training, then the looks in chunks of `eval_chunk_tasks` tasks, with the early drop of a LoRA that can no longer win. |
| The decision | `sn120_v2/judging.py` `decide`: delta, standard error, z, the cap-hit and cover rules, the status. With `crown_gate` = `argmax_symmetric` and `z_bar` > 0, eligibility needs delta > `TIE_TOLERANCE` AND z strictly above `z_bar` (with se > 0), including `winner` here. |
| The draw | `sn120_v2/irt.py` `draw_from_snapshot`. |
| The model cache | `sn120_v2/gpu/resident.py` `Resident`: every file set is hashed before it is used. |

### "The GPU backend is not published"

That sentence is no longer true, and it was imprecise. The GPU code of the box has always been in this repository: it is `v2/sn120_v2/gpu/`. mining.json pins its digest as `params.recipe.code_digest`, and `compare` shows that digest next to the digest of your checkout. What was missing is a function that builds this code for a machine that is not the box. `sn120_v2.replicate` is that function.

What is still **not** published is the exact list of the packages installed on the box. The install line of the box is public (`uv pip install -e ".[dev,gpu,r2,chain,envs,miner]"` on Python 3.12, see `SMOKE.md` section 1), but the repository has no lock file, so the versions that this line resolves to depend on the day of the install. The install digest covers the patch version of Python and every installed package. So your install digest differs from the pinned one, even with the same install line. This is the difference `install` below.

The code digest covers `sn120_v2/gpu/*.py` only. The file check, the round runner and the judging are outside it. The tool cannot tell you whether your checkout has the box's version of those files. Use the tip of the branch `v2`.

### The differences

Each entry of `differences` in a report has one of these names in `what`. The text in the report has the numbers of the params at that moment.

| `what` | When | What differs |
|---|---|---|
| `install` | Your install digest is not in `recipe.image_digests`. In practice always. | The versions of torch, vLLM and every other package can differ from the box. The box code refuses an install that is not pinned, so the run uses a copy of the params that pins your install digest. |
| `code` | Your `sn120_v2/gpu/` has another digest than `recipe.code_digest`. | You run other GPU code than the box. The run pins your digest. Check out the commit the box runs to remove this difference. |
| `gpus` | `--gpus` is not `n_gpus` of the params. | One vLLM server runs on each GPU, and task i of a chunk goes to server i mod the number of GPUs. With fewer GPUs each server gets more of the tasks. The training job splits the rows over its GPUs and adds the gradients: the same sum in exact arithmetic, another rounding. |
| `round-clock` | Always, unless `--round-clock`. | No time limit of the round applies. On the box the training ends `train_time_budget_s` after the start of the round, and the scoring stops at `round_time_limit_s - promote_reserve_s`; what is scored by then is the whole test. A slower machine would hit these limits and score fewer tasks. The limit of one reply, `eval_time_budget_s`, is the same here. |
| `one-batch` | Always. | The round has one batch: yours. The result says whether your LoRA beats null M by the win rule. In a real round the eligible LoRA with the highest delta wins, so `winner` here does not mean a win there. The training job has all GPUs of the run; on the box a job has floor(`n_gpus` / batches of the round) GPUs, at least 1. On the box all LoRAs of a round are loaded into the same servers and scored together, and the LoRA table of the servers (`--max-loras`) is the number of batches of the round; here it is 1. |
| `no-master` | Always. | The fp32 master of the head is not checked and not needed. The box checks it before it trains or scores, because the box also folds a winner. This tool never folds. |
| `free-gpus` | Always. | Before the scoring starts, only GPUs 0 to N-1 must be free. The box waits for all of its GPUs. |
| `first-tasks` | With `--first-tasks`. | Only the first N tasks of the draw are scored. |
| `round-head` | The round you repeat ran on another head than the head of mining.json. | The run uses the head of mining.json. Null M is then another model than in that round. |
| `round-params` | The round you repeat ran with other params than the params of mining.json. | The run uses the params of mining.json. |

The changed values are these and no others: `recipe.image_digests`, `recipe.code_digest`, `n_gpus`, and, without `--round-clock`, `round_time_limit_s`, `lease_timeout_s`, `promote_reserve_s` and `train_time_budget_s`. Every training value, every scoring value and every rule of the decision is the one of mining.json.

`--strict` stops the run when `install`, `code` or `gpus` would be in the list.

Two more facts that are not in the list, because they are about the machine and not about the run:

- The type of the GPU. Another GPU type computes with another rounding.
- The box runs rounds all day; its vLLM compile caches are warm. Times here say nothing about times there.

## 6. What you can repeat exactly, and what is equal only in distribution

**Exactly the same, on any machine:**

- The draw of a round: the task ids, their order and their weights. It is a pure function of the public IRT snapshot and the public round seed. `--round-file` checks this against the draw that the box published (`same_draw`).
- The seed of each rollout: `sha256("<round_seed>|<task_id>|<sample>")` as a 63-bit integer. Null M and every LoRA share the same seed for sample 0 on the same task. When the king has more samples, its later samples are king-only and use their own sample indices.
- The seed of the training job: `sha256("<round_seed>|train|<batch_sha256>")` as a 63-bit integer. The A matrices of the LoRA start from this seed, so they start the same.
- The prompt of each task and every request setting (`temperature`, `top_p`, `top_k`, `max_output_tokens`, thinking on).
- The rows that train, their advantages, and the tokens of each row.
- The grade of a given reply text. The grader refuses to start unless its packages have exactly the pinned versions. One exception: a grade that takes longer than the grader's time limit is an error, and a slower CPU reaches that limit sooner.
- The decision for given outcome strings: delta, standard error, z, the status.

**Equal only in distribution:**

- The text of a rollout, and so its outcome code. The seed is the same, but the sampled tokens depend on floating-point results, and these depend on the GPU type, the package versions, and which requests the server computes together in one batch. Do not expect the same text for the same seed, on your machine or on the box. Expect the same pass rate within sampling noise.
- The values of the trained LoRA after the first step, and so `lora_hash`. The gradient depends on floating-point results in the same way.
- So `delta` and `z` of your LoRA differ from run to run and from the box.

**Not known before the round opens:**

- The draw of a future round. The round seed is a hash of the round id, the head and a value that the validator cannot choose. mining.json shows it when the round opens. A seed of your choice gives a draw from the same distribution, when the snapshot is the current one (`irt_snapshot_id` in mining.json).
- The other LoRAs of the round.

**Not measured:** how often the same seed gives the same text on the box, and how far the pass rate of null M on another GPU type is from the box. `--round-file` gives you the second number for your machine.

## 7. With `affine_miner.py reproduce`

The miner tool takes a backend function. This module has one:

```
python v2/miner/affine_miner.py reproduce --mining-json mining.json --hotkey <ss58> --irt-snapshot irt.json --round-seed <seed> --backend sn120_v2.replicate:backend batch.json
```

That command runs with the params of mining.json as they are. So the box code refuses the run unless your machine has the pinned install, the pinned code and `n_gpus` GPUs. In practice that is the box only. Use `python -m sn120_v2.replicate run` on every other machine. The backend needs the base files in the cache of the work directory. So pass `--work DIR` to the miner tool, and run the `fetch` command above with the same `--work DIR` first. Without `--work` the miner tool uses a new temporary directory, which has no base files.

## 8. What has been verified

On real GPUs:

- The box trains and scores its rounds with the code that `run` calls (`sn120_v2/gpu/`, the round runner, the judging). When this was written (2026-11-14), round `r000001` was open and no round had closed. So no public round file existed yet, and `--round-file` has not been used on a real one.
- **Nothing of `sn120_v2/replicate.py` has run on a GPU.** No run of `run` on a miner's machine exists yet. Which GPU is large enough, how long a run takes, and whether an install of another day works are not known.

On CPU, by the tests in `tests/test_replicate.py` (the real trainer, evaluator, runner and judging of the box, with a fake training process, fake vLLM servers and a fake grader):

- `run` trains with the seed, the rows and the recipe of the box, and scores every task of the draw in its order with the seeds of the round, for null M and the LoRA.
- The copy of the params changes only the values listed in section 5.
- `fetch` checks every file, moves only a complete set into the cache, and shows no part of a signed link in an error.
- `--first-tasks` scores a prefix of the draw. `--round-file` compares with the file of the box.

Against the live public files, on a machine without a GPU, on 2026-11-14 (params `f777b949…`, head `9fa81073…`):

- `compare` reads mining.json and prints the differences.
- The signed links work: the 8 small base files were downloaded, each with the right size and sha256. The 4 weight files (18.8 GB) were not downloaded.
- The draw of round `r000001` was rebuilt from the public snapshot and seed: 2000 tasks, all in the public pool. It could not be compared with the draw of the box, because the file of that round is public only after the round closes.
- The file check ran with the real tokenizer of the head and the real grader on a small valid file.

## 9. For the operator

This change was prepared and tested without a production host. It changes files under `sn120_v2/gpu/` and `pyproject.toml`; because the code digest covers `sn120_v2/gpu/*.py`, a box adopting this commit will have a new code digest and must qualify it under its normal rollout procedure. The public-contract loader does not alter strict journal decoding or replay. Processor/tokenizer supplementation happens in a separate prepared fold-input directory; it never changes a sealed parent cache or manifest. This PR does not deploy the change, restart a box, or modify an existing checkpoint or optimizer lineage.

One finding for the public site, with no action in this change: `https://affine.io/public/...` answers 403 to the default client name of Python's `urllib`, while `mining.json` and `llms.txt` answer 200. This tool sends its own client name (`sn120-v2-replicate/1`) and is not affected. A miner script that reads `public/irt/...` or `public/read_links.json` with plain `urllib` is.

One step would remove the difference `install` for miners. It is not part of this change and it needs the box: publish the package list of the box (`uv pip freeze` and the Python version) next to mining.json. A miner who installs exactly that list on the same Python gets the pinned install digest.
