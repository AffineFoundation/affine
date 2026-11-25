# Solo GPU scheduling and serving changes

These changes are relative to `v2` at `7bcb24bc` (PR 274), including the early-training and cutoff behavior seen in `r000059`. They are for the later redeployment together with PR 274, not the rolled-back `4a0c03c1` deployment.

## Idle GPUs remain available for eval

A GPU with no training assignment starts its vLLM server and claims null rollouts immediately, even before any LoRA has finished training. When no rollout is ready, it waits for a LoRA, a completed result, a cut, or a failure. It exits only when training has finished and no live LoRA or null work remains, or on a cut or failure. This applies whether there are fewer, as many, or more training batches than GPUs.

The last LoRA is added before the scheduler marks training complete, so a worker cannot observe training completion while its final artifact is not yet available. Work-selection order, task/sample indexing, rollout seeds, grading, scoring, and early-look decisions are unchanged.

The regression test in `tests/test_solo_round.py` runs eight GPU scheduler workers with six LoRAs still in training. It requires every GPU to serve null before the LoRAs are added, then requires every GPU to serve rollouts, GPUs 6 and 7 to serve LoRA work, and every LoRA and null outcome to complete. A direct scheduler test also covers exhausted null work with one LoRA and one GPU. With no live LoRA, the early-exit guard requires `training_done and not alive`; the separate exhausted-work check still lets workers exit after every job is claimed.

## Concurrent startup and adapter loading

Each GPU has its own startup lock, so repeated callers reuse one server. Spawning and the health wait do not hold the shared `_solo_lock`; only the lifecycle dictionary updates use it. Thus one GPU's vLLM startup does not serialize its peers or block already-loaded LoRA claims. `_load_one` also checks its per-GPU loaded set before taking the shared lock and repeats the check after acquiring it. The tests force two GPU spawns and two health waits to overlap and reject any shared-lock acquisition for an already-loaded adapter.

## Deadline keeps the completed paired prefix

At the scoring cutoff, Solo now returns the complete `eval_chunk_tasks` block prefix shared by null and every still-live LoRA, padded with `NOT_RUN` for the unfinished tail, and marks `stopped_at_limit=True`. This restores the paired-prefix behavior of the streaming evaluator: completed work is not discarded merely because the round limit interrupted the rest. Only an empty shared prefix raises `InfraError`. The public round text consequently reports `(stopped at the time limit)`.

The tests cover a partial shared prefix with extra null results, the existing round-text wording, and the empty-prefix failure. A `_Deadline` observed by a worker marks the scheduler cut directly, and the runner also checks the deadline after the pool drains. This covers all workers returning before the runner's clock poll, the path that previously left `scheduler.cut` false and discarded completed work. A runner-level regression verifies the shared prefix and `stopped_at_limit=True` in that case. They do not change the judging rule, early looks, or seeds.

## Opt-in FP8 KV cache and 512 concurrent requests

The old behavior remains the default: `kv_cache_dtype = "auto"` and `max_num_seqs = 256`. Both fields are listed in `Params.LEFT_OUT_AT_DEFAULT`, so old `Params` JSON and tickets retain their exact JSON, `params_hash`, and ticket digest. The default `params_hash` remains `10a121ec5a4ba830057e5510665fee1be7aad34dae259d3427e4547c393c2cf1`, unchanged from v2 at `7bcb24bc`.

Production enables the speedup explicitly through `set-params`:

| Parameter | Opt-in value |
| --- | --- |
| `kv_cache_dtype` | `"fp8"` |
| `max_num_seqs` | `512` |

With those explicit values, the default `Params` hash is `cf2da207288c978d9452fc5594d156efef6ab46f283cf45018c032073bbf81b1`. The hash changes only when the parameters are explicitly selected (or another pinned value, such as a deployment code/image digest, changes). The `dtype` remains `bfloat16`: FP8 applies to the KV cache, not the model weights.

`params.max_num_seqs` drives both the HTTP client pool and vLLM `--max-num-seqs`; the existing file-descriptor budget can reduce both together. At the default `auto` setting, `--kv-cache-dtype` is omitted and the server command remains byte-identical to before. The generated `llms.txt` server command and request-feed description read these same parameters. `ServeSettings.clients_per_server` remains an optional local override, not a separate production 512 constant.

While any training job is still running, each Solo worker caps its in-flight requests at 256 via the execution-only `SoloSettings.clients_while_training`. Once `scheduler.training_finished()` is true, the worker uses its full configured client count (512 for the fp8/512 profile). This keeps the server's fp8/512 settings while reducing eval pressure on GPUs that are still training. It does not affect hashed Params, task/sample pairing, or seeds.

The GPU `code_digest` changes because the scheduler and serving implementation changed. `v2/ops/compat/params_fields_compat.patch` is a compat-only patch for an older checkout: it adds only these two fields and their `LEFT_OUT_AT_DEFAULT` entries. It lets pre-PR code read journals containing explicit fp8/512 parameters and is not part of v2's scoring or serving implementation.

## Throughput evidence

Operator-supplied offline measurements on 8x B300:

| KV cache | Sequences per server | Throughput per GPU | Preemptions |
| --- | ---: | ---: | --- |
| bf16 | 256 | 6,632 tok/s | Not supplied |
| fp8 | 512 | 10,904 tok/s | 0 |

Affine also measured the scheduler and concurrent-startup fixes in a shadow test on 8x B300 using the `r000057` inputs: 5,562 s eval wall versus 9,114 s live (**39% lower**), 94% average utilization across all eight GPUs, and the same decision as live. The referenced `PART-A.md` notes were not available in this checkout; these figures are supplied by Affine, not reproduced by this PR's local tests.

These are offline or shadow-test measurements, not results from a new live round. This change has not been run on B300 hardware as part of this PR.
