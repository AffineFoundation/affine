# Notice text — wvk 25: teacher swap + 262k + scoring bundle (directive 2026-09-26 09:09 UTC; post 2026-09-26)

## Discord (announcements) — post today

**Upcoming fork: weight_version_key 24 → 25, effective Wednesday 2026-09-30 14:00 UTC (at the first duel boundary after that time)**

What changes
- **Teacher.** The frozen model the duel scores against moves from `Qwen/Qwen3.8-27B` to `zai-org/GLM-5.3-Flash` (320B MoE, 18B active, MIT). Every anchor (μ, σ) is the teacher's own leave-one-out statistic, so the rule re-baselines itself; scores are not comparable across the fork.
- **Context.** Serving window 131,072 → 262,144 tokens on the eval pod and the teacher swarm. Prefix cap in D 110,000 → 255,744 tokens, measured with both the teacher and the genesis tokenizer. Deep-trajectory turns dropped at the old cap enter D over the following folds.
- **Scoring bundle (same fork):**
  1. *Miner empty-thought rule.* A miner turn whose thought has fewer than 10 content tokens scores `min(z_R, z_A)` — the typicality leg is dropped — instead of the −6 floor, the same rule wvk 24 applies to the teacher's own references. Forfeits (no parseable action) keep the −6 floor. Admission gate: a side whose share of such turns exceeds 2× the teacher's own share on the slice has those turns scored at the floor. Why: since wvk 22 most of every crown margin came from the king having more empty-thought turns than the challenger — a channel noise patches could move; nothing about turn quality had to be better.
  2. *Sequential stopping.* The paired margin is checked every 100 scored turns; a duel stops with a crown when `margin − 2.6·SE > δ` on two consecutive checks, stops for futility when even a 2.6·SE upward move cannot reach δ, and otherwise runs to 1,000 turns. Offline: 29/30 verdicts agree with the full slice, ~2× verdicts per day. The first five wvk-25 duels also score the full slice in shadow and publish both decisions.
  3. *R cap.* `z_R := min(z_R, 0)` — a thought earns no credit for predicting the teacher's action better than the teacher's own alternative thoughts do. (Ships unless the pre-flip probe on the final code shows a problem.)
  4. *Control.* The fully matched teacher-vs-king control (king scored on the same k−1 references as the held-out reference) is published on every verdict and is the rollback signal.
- **Admission rule (like the architecture pin, same fork): 256k context required.** A submission's `config.json` must declare an effective context window of at least 262,144 tokens — `max_position_embeddings` (or the equivalent key), times the rope-scaling factor for `linear` / `dynamic` / `yarn` scaling, as vLLM derives `max_model_len`. Checked before download at dispatch and prefetch and at R2 intake; rejection reason `rejected_context_too_short`; `submit.py check` reports it before you upload. The genesis family is native 262,144 and passes unchanged; a config that shortens the window is rejected. Challengers are served and probed at `max_model_len 262144` from T0.

What does NOT change
- δ = 0.2 sd, k_sigma = 2, forfeit floor −6, miner caps (thought 4,096 with the 1.25× teacher-relative rule, action 768), ref cap 4,864, the architecture pin, the B licence, the thought-length floor.
- **Reign 21 stands.** Forward-only; no re-verdicts; `min_submission_block` unchanged.

Pre-flight for miners
- Your `config.json` must declare ≥ 262,144 tokens of context (see the admission rule above); `python affine/scripts/submit.py check <repo>` shows the derived window.
- `vllm serve <your checkpoint> --max-model-len 262144 --tensor-parallel-size 2` must load and answer `/v1/completions` with finite logprobs on an echo request. The genesis family is native 262k; do not shorten rope in your config.
- Nothing else changes in what you submit.

Numbers behind it (stored verdicts + a 404-turn GLM shadow): fully matched teacher−king control +0.39 sd under Qwen → +0.80 under GLM (typicality +0.40 → +1.68, action +0.29 → +0.53, R ≈ 0 under both); GLM reference yield 2.9 of 3 per turn.

Timeline
- 2026-09-26: this notice; datagen teacher seat moves to GLM (data event).
- 2026-09-29: GLM teacher swarm pre-warmed next to the Qwen one.
- **2026-09-30 14:00 UTC: flip** at the first duel boundary; the first wvk-25 verdict stamps `teacher.repo = zai-org/GLM-5.3-Flash`, `max_model_len 262144`, the new knobs; a live line follows here.

## Addendum (2026-09-26 17:40 UTC, post as a follow-up line under the notice) — tool-call format at T0

- **Tool-call turns change format at T0.** About a third of D (the `tool_call` / `text` / `terminus` turns from tool-using harnesses) is stored as the teacher's own chat-template text. From T0 those prefixes are re-baked under GLM-5.3-Flash's template: tool schemas in a `<tools>…</tools>` system block, tool calls as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, tool results as `<tool_response>…</tool_response>` in a user turn, and GLM's `<|system|>Reasoning Effort: Max` preamble as literal text at the top of the system prompt. Bash / boxed prefixes are byte-identical.
- The action parser is unchanged and accepts any `<tool_call>…</tool_call>` block, but the teacher's reference actions are in GLM's form and the action leg (A) and the B licence are scored by the GLM teacher — a Qwen-style call (`<function=…>` / JSON) on a tool-call turn is scored as an unlikely action. **Emit GLM's form on tool-call turns.**
- Unaffected: the protocol probe (renders through your own template with `tools=`), the `</think>` think-close rule, the `text` fallback at tool turns, the thought-length floor.

## llms.txt — "Upcoming fork: wvk 25" (until T0), then "Fork history: wvk 25"

wvk 25 (effective 2026-09-30 14:00 UTC): teacher `Qwen/Qwen3.8-27B` → `zai-org/GLM-5.3-Flash`; serving window 131,072 → 262,144 tokens (prefix cap 255,744, both tokenizers); miner empty-thought rule (< 10 content tokens → `min(z_R, z_A)`, admission gate 2× the teacher's share); sequential stopping (looks every 100 turns, `margin − 2.6·SE > δ` on two consecutive looks, futility stop, else 1,000); `z_R` capped at 0; fully matched control published; admission rule `[submission].min_context_tokens = 262144` (effective context window from `config.json`, rope scaling applied as vLLM derives `max_model_len`; genesis passes; `rejected_context_too_short`). Tool-call turns of D re-derived under the new teacher's chat template (`<tools>` system block, `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`, `<tool_response>` results, GLM preamble as literal system text) — emit GLM's tool-call form on those turns; the parser accepts any `<tool_call>` block but A and B are scored by the GLM teacher. bash / boxed turns byte-identical. Forward-only; reign 21 stands; `min_submission_block` unchanged.

## Dashboard banner (`#fork-notice`, remove at T0)

Upcoming fork wvk 25 — Wed 2026-09-30 14:00 UTC: teacher → GLM-5.3-Flash, 262k context, miner empty-thought rule, sequential stopping, R cap. Reign 21 stands. Miners: your config must declare ≥ 262,144 tokens of context and serve at that window.

## AMENDMENT (draft 2026-09-26 23:30 UTC — post only on Jacob's decision) — teacher swap deferred; wvk 25 ships as the scoring bundle + 262k on the current teacher

**Amendment to the wvk 25 notice.** During pre-flip testing we found that GLM-5.3-Flash's teacher echo log-probabilities are not run-to-run reproducible on our serving stack at duel lengths (sparse-attention top-k selection and fp8-MoE batch variance; dense mode fixes it only at a 64k window, which would cap the dataset). We will not put a non-reproducible teacher into the contract. Therefore:

- **wvk 25 on Wednesday 2026-09-30 14:00 UTC ships without the teacher change.** Teacher stays `Qwen/Qwen3.8-27B` (echoes bit-exact today).
- **Everything else in the notice stands:** the serving window 131,072 → 262,144 tokens (prefix cap 255,744), the 256k-context admission rule (`config.json` effective window ≥ 262,144; `rejected_context_too_short`; `submit.py check`), the miner empty-thought rule + admission gate, sequential stopping (looks every 100 turns, `margin − 2.6·SE > δ` on two consecutive looks), the R cap, the fully matched control on every verdict. Reign 22 (the sitting king) stands; forward-only; `min_submission_block` unchanged.
- **The tool-call format change does NOT happen at T0.** D stays baked under the Qwen template; keep emitting Qwen-style tool calls. Ignore the 09-26 addendum about GLM's `<tool_call>name<arg_key>…` form until a future teacher notice.
- The teacher swap moves to its own fork with its own ≥ 48-hour notice once a deterministic echo path exists (dense prefill at 262k in vLLM, or a candidate whose echoes are exact under load). Nothing about that fork is decided.

## Alternative amendment (option B′ — only if Jacob accepts the noise)

- Teacher → GLM-5.3-Flash proceeds as noticed. **Known property:** the teacher's echo log-probabilities on our serving stack are not run-to-run reproducible above 2,048 prompt tokens (sparse-attention top-k / fp8-MoE batch variance). Measured on 200 stored turns re-echoed twice: turn-score repeat sd ≈ 0.4–0.6 sd; on a 1,000-turn paired margin ≈ 0.016 sd — about 0.45 of a verdict's SE. **What this means for you:** the stored echo values on the verdict are the contract and the scoring rule replays exactly from them; if you re-run the echoes yourself you will get different per-turn numbers and a margin within ≈ 0.02 sd of the published one; a verdict whose margin sits within that of the bar could have gone the other way on a re-run — we publish the measured `teacher_echo_repeat_sd` on every verdict and re-measure it weekly. No verdict is re-judged on a re-echo.
