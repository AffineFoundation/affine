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

What does NOT change
- δ = 0.2 sd, k_sigma = 2, forfeit floor −6, miner caps (thought 4,096 with the 1.25× teacher-relative rule, action 768), ref cap 4,864, the architecture pin, the B licence, the thought-length floor.
- **Reign 21 stands.** Forward-only; no re-verdicts; `min_submission_block` unchanged.

Pre-flight for miners
- `vllm serve <your checkpoint> --max-model-len 262144 --tensor-parallel-size 2` must load and answer `/v1/completions` with finite logprobs on an echo request. The genesis family is native 262k; do not shorten rope in your config.
- Nothing else changes in what you submit.

Numbers behind it (stored verdicts + a 404-turn GLM shadow): fully matched teacher−king control +0.39 sd under Qwen → +0.80 under GLM (typicality +0.40 → +1.68, action +0.29 → +0.53, R ≈ 0 under both); GLM reference yield 2.9 of 3 per turn.

Timeline
- 2026-09-26: this notice; datagen teacher seat moves to GLM (data event).
- 2026-09-29: GLM teacher swarm pre-warmed next to the Qwen one.
- **2026-09-30 14:00 UTC: flip** at the first duel boundary; the first wvk-25 verdict stamps `teacher.repo = zai-org/GLM-5.3-Flash`, `max_model_len 262144`, the new knobs; a live line follows here.

## llms.txt — "Upcoming fork: wvk 25" (until T0), then "Fork history: wvk 25"

wvk 25 (effective 2026-09-30 14:00 UTC): teacher `Qwen/Qwen3.8-27B` → `zai-org/GLM-5.3-Flash`; serving window 131,072 → 262,144 tokens (prefix cap 255,744, both tokenizers); miner empty-thought rule (< 10 content tokens → `min(z_R, z_A)`, admission gate 2× the teacher's share); sequential stopping (looks every 100 turns, `margin − 2.6·SE > δ` on two consecutive looks, futility stop, else 1,000); `z_R` capped at 0; fully matched control published. Tool-call turns of D re-derived under the new teacher's chat template; bash / text turns byte-identical. Forward-only; reign 21 stands; `min_submission_block` unchanged.

## Dashboard banner (`#fork-notice`, remove at T0)

Upcoming fork wvk 25 — Wed 2026-09-30 14:00 UTC: teacher → GLM-5.3-Flash, 262k context, miner empty-thought rule, sequential stopping, R cap. Reign 21 stands. Miners: serve 262,144 tokens.
