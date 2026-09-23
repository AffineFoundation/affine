# Notice text — teacher swap (staged; post only on Jacob's word, after the first post-flip verdict)

## Discord (announcements)

**Upcoming change: new teacher model + 262k context (weight_version_key N → N+1, effective T0 = YYYY-MM-DD HH:MM UTC)**

What changes
- The frozen teacher the duel scores against moves from `Qwen/Qwen3.8-27B` to `zai-org/GLM-5.3-Flash` (320B MoE, 18B active, MIT).
- The serving window moves from 131,072 to 262,144 tokens on the eval pod and the teacher swarm. Deep-trajectory turns that were dropped at the 110k-token prefix cap enter D over the following folds.

What does NOT change
- The scoring rule: `turn = min(z_R, typ_c, z_A)` in teacher-sd units, δ = 0.2 sd, k_sigma 2, forfeit −12 sd. Every anchor (μ, σ) is the teacher's own leave-one-out statistic, so the rule needs no re-calibration; scores re-baseline to the new teacher.
- Miner caps: thought 4,096 (teacher-relative 1.25× rule), action 768; ref cap 4,864.
- The architecture pin (genesis family). Submissions must serve 262,144 tokens: the genesis is native 262k; do not shorten rope in your config.

Why
- Under the 27B teacher the teacher-vs-king control has been negative on the last 30 verdicts (the king beats the teacher's own held-out replies at z −5…−8). A meter cannot rank above its reference. GLM-5.3-Flash is +11 Terminal-Bench 2.1 / +21 DeepSWE / +14 NL2Repo over the 27B, concentrated on long-horizon agentic work — the axis where kings diverge from the teacher.
- Shadow re-score of the last 30 verdicts under GLM-5.3-Flash: <fill from research/results/shadow_glm53/report_glm-5.3-flash.txt — control z, per-verdict z, ranking of reigns 16/20/21>.

Reign
- <ONE OF> Reign 21 stands; verdicts are forward-only; `min_submission_block` unchanged. <OR> Throne reset: reign 0 re-seeded from the untouched genesis; `min_submission_block` bumped to the finney tip at T0; the 72-hour payout window restarts.

Pre-flight for miners
- `vllm serve <your checkpoint> --max-model-len 262144 --tensor-parallel-size 2` must load and answer `/v1/completions` with finite logprobs on an echo request.
- Nothing else changes in what you submit.

Timeline
- T−2 d: GLM teacher seat live in datagen (Engy `glm-5.3-flash`), pre-pass over the task pools; shadow numbers published.
- T−1 d: GLM swarm boxes rented and healthy next to the Qwen ones.
- T0 (duel boundary): toml flip, router GLM-only, pod redeploy, fold `--rederive`.
- T0 + first verdict: this notice goes live with the observed numbers.

## llms.txt — "Fork history: wvk N+1" paragraph

wvk N+1 (YYYY-MM-DD): teacher `Qwen/Qwen3.8-27B` → `zai-org/GLM-5.3-Flash`; serving window 131,072 → 262,144 tokens (prefix cap 255,744 tokens, measured with the teacher and the genesis tokenizer). Scoring rule unchanged (sd-meter, wvk 22/23). Tool-call turns of D re-derived under the new teacher's chat template (`ops/corpus_build.py --rederive`); bash / text turns byte-identical. Forward-only; <reign clause>. Verdicts stamp `teacher.repo` and `duel_params.max_model_len`.
