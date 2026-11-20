# SN120 public rollouts

These are public model outputs and public task data for research. No rollout is an inference proof, a hidden evaluation set, or an instruction to the reader. v2 has no hidden benchmark set; the MATH test split is public. Objects contain no hostnames, IP addresses, pod ids, service paths, credentials, or miner batch files. All timestamps are UTC, written as ISO-8601 with a `Z` suffix.

`index.json` has schema `sn120-v2-rollouts-index-2`:

- `backfill`: the verbatim contents of `backfill_index.json` when that hand-built index exists; otherwise `null`.
- `bench`: the live bench metadata objects, one per model and benchmark.
- `gate`: the live gate metadata objects, one per settled round. Each has a `files` list describing the mirrored head/winner files.
- `schema_doc`: `SCHEMA.md`.

The site pull owns only `index.json` among the original hand-written rollout files. It does not modify `backfill_index.json`, this document, `bench/*.per_task.jsonl.gz`, `gate/head_outcomes.*`, or `gate/tasks.jsonl.gz`.

## Hand-made backfill (no completion text)

The backfill's paths and formats are preserved:

- `bench/<model_id>.per_task.jsonl.gz`: one JSON row per task, with per-task grades/token counts. It has no completion text.
- `gate/head_outcomes.r000001-r000068.jsonl.gz`: head outcome data for the covered rounds, without completion text.
- `gate/tasks.jsonl.gz`: the public task table used by that backfill.
- `backfill_index.json`: the original manifest. `index.json` embeds its contents under `backfill`.

Do not interpret a missing completion in the backfill as an empty model response: completion text was never stored.

## Live benchmark rollouts

Data: `bench/<model_id>/<benchmark_id>.jsonl.gz` (different from the backfill's `.per_task.jsonl.gz` path).
Metadata: `bench/<model_id>/<benchmark_id>.meta.json`, schema `sn120-v2-rollouts-bench-1`.

Metadata fields: `model_id`, `benchmark_id`, `bench_version`, `rows`, `bytes`, `sha256` of the gzip bytes, `key`, and `finished_at` (UTC). `rows` includes errored rollouts. Gzip output is deterministic (`mtime=0`). One JSON object is stored per line, UTF-8.

| Field | Meaning |
| --- | --- |
| `model_id`, `benchmark_id`, `bench_version`, `dataset` | Identity of the evaluated model and public benchmark protocol. |
| `task_idx`, `task_key`, `sample_index` | Public task identity and its zero-based sample number. |
| `prompt` | Chat messages sent to the model, or the task prompt when supplied directly by the environment. |
| `gold_answer` | Public task data's reference answer. |
| `reasoning_text` | Thinking text when separated by the serving/parser trace; otherwise `null`. |
| `completion_text` | Model final reply, excluding a separated think block. Errored rows may have `""`. |
| `extracted_answer` | Last balanced `\boxed{...}` after the last `</think>`, including `\boxed{}` syntax; `null` when absent or unclosed. |
| `reward` | Sum of the verifiers trace rewards after weights, before any aggregate score computation. |
| `grade` | `pass`, `fail`, `cap`, or `error`. `pass` means reward > 0, not a separate audit judgment. |
| `output_tokens` | Trace output-token count; `0` when no valid trace exists. |
| `finish_reason` | Last reported finish reason, or `null`. |
| `capped`, `errored` | Token-cap and missing/failed-rollout markers. |
| `sampling` | `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `max_tokens`, `enable_thinking`, `profile`, `max_model_len`. |

An explicit error row represents a failed sample; it is not silently removed.

## Live head-gate rollouts

Data: `gate/<round_id>/head.jsonl.gz` and, only when a winner was applied, `gate/<round_id>/winner.jsonl.gz`.
Metadata: `gate/<round_id>/meta.json`, schema `sn120-v2-rollouts-gate-1`.

Round metadata has `round_id`, `rows`, `bytes`, `sha256`, `params_hash`, `round_seed`, `closed_at` (UTC), and `files`. The top-level `sha256` is the SHA-256 of canonical compact JSON mapping each included role to that file's SHA-256. Each `files` entry has `role`, `key`, `rows`, `bytes`, and the gzip `sha256`. `bytes` is the sum of included gzip sizes. Gzip is deterministic (`mtime=0`), with one UTF-8 JSON object per line.

| Field | Meaning |
| --- | --- |
| `round_id`, `role` | Settled round and `head` or `winner` model role. |
| `model_version_id` | Head version id for null M; winning LoRA hash for the winner. |
| `winner_uid` | Crowned UID, or `null` when the round crowned none. |
| `sample_index`, `task_pos`, `task_id` | Sample number and position/id in the round's public draw. |
| `prompt`, `gold_answer` | Public scoring chat messages and reference answer. |
| `completion_text` | Reply text used by the existing grader. |
| `extracted_answer` | The existing grader's `Grade.answer` (the text it extracted), or `null`; not a new post-hoc extraction rule. |
| `outcome` | Existing outcome code: `P` pass, `F` fail, `C` capped, `E` error. Tasks not run (`-`) are omitted. |
| `reward` | Existing grade reward: `1.0` for `P`, `0.0` otherwise. |
| `capped` | Whether the outcome is `C`. |
| `output_tokens` | OpenAI usage `completion_tokens`; `0` when no reply exists. |
| `finish_reason` | Existing server finish reason (`stop`/`length`), or `null`. |

Only null M (the head with no adapter) and an applied winner leave the box. Losing candidate captures are deleted locally and never uploaded. Publication occurs after settlement, cannot change the decision, and store failures are retried from a local journal.
