# Solo GPU scheduling and serving changes

These changes are relative to `v2` at `7bcb24bc` (PR 274), including the early-training and cutoff behavior seen in `r000059`. They are for the later redeployment together with PR 274, not the rolled-back `4a0c03c1` deployment.

## Idle GPUs remain available for eval

A GPU with no training assignment starts its vLLM server and claims null rollouts immediately, even before any LoRA has finished training. When no rollout is ready, it waits for a LoRA, a completed result, a cut, or a failure. It exits only when training has finished and no live LoRA or null work remains, or on a cut or failure. This applies whether there are fewer, as many, or more training batches than GPUs.

The last LoRA is added before the scheduler marks training complete, so a worker cannot observe training completion while its final artifact is not yet available. Work-selection order, task/sample indexing, rollout seeds, grading, scoring, and early-look decisions are unchanged.

The regression test in `tests/test_solo_round.py` runs eight GPU scheduler workers with six LoRAs still in training. It requires every GPU to serve null before the LoRAs are added, then requires every GPU to serve rollouts, GPUs 6 and 7 to serve LoRA work, and every LoRA and null outcome to complete. A direct scheduler test also covers exhausted null work with one LoRA and one GPU.

## Deadline keeps the completed paired prefix

At the scoring cutoff, Solo now returns the complete `eval_chunk_tasks` block prefix shared by null and every still-live LoRA, padded with `NOT_RUN` for the unfinished tail, and marks `stopped_at_limit=True`. This restores the paired-prefix behavior of the streaming evaluator: completed work is not discarded merely because the round limit interrupted the rest. Only an empty shared prefix raises `InfraError`. The public round text consequently reports `(stopped at the time limit)`.

The tests cover a partial shared prefix with extra null results, the existing round-text wording, and the empty-prefix failure. They do not change the judging rule, early looks, or seeds.

## FP8 KV cache and 512 concurrent requests

Evaluation vLLM servers use `--kv-cache-dtype fp8` and `--max-num-seqs 512`. The normal HTTP client pool matches that setting with `clients_per_server = 512`, so each server is fed 512 requests concurrently. The existing file-descriptor budget can still reduce the Solo client count together with the corresponding per-server sequence limit.

The following fields are part of `Params` and therefore `params_hash`:

| Parameter | Value |
| --- | --- |
| `kv_cache_dtype` | `"fp8"` |
| `max_num_seqs` | `512` |

The default `params_hash`, with an empty `recipe.code_digest` and no image digests, is `cf2da207288c978d9452fc5594d156efef6ab46f283cf45018c032073bbf81b1`. Pinning a deployment's code or image digests produces a different hash through the normal `set-params` procedure. The `dtype` remains `bfloat16`: FP8 applies to the KV cache, not the model weights.

The generated `llms.txt` server command and request-feed description read these same parameters. The idle scheduling change does not add parameters. The GPU `code_digest` changes because the scheduler and serving implementation changed.

## Throughput evidence

Operator-supplied offline measurements on 8x B300:

| KV cache | Sequences per server | Throughput per GPU | Preemptions |
| --- | ---: | ---: | --- |
| bf16 | 256 | 6,632 tok/s | Not supplied |
| fp8 | 512 | 10,904 tok/s | 0 |

These are offline measurements, not results from a new live round. This change has not been run on B300 hardware as part of this PR.
