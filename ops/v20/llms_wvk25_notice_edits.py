"""llms.txt: add "Upcoming fork: wvk 25" (notice 2026-09-26, T0 2026-09-30 14:00 UTC). Idempotent.
At T0 the cutover deploy turns it into "Fork history: wvk 25" (lead's script)."""
from __future__ import annotations
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
def sub(s, old, new):
    if s.count(old) != 1: raise SystemExit(f"anchor: {old[:70]!r}")
    return s.replace(old, new)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--builder", type=Path, default=REPO / "affine" / "scripts" / "build_llms_txt.py"); a = ap.parse_args()
    s = a.builder.read_text()
    if "WVK25_NOTICE" in s: print("already"); return 0
    s = sub(s, 'WVK24_EFFECTIVE = ', 'WVK25_NOTICE = "2026-09-26"\nWVK25_T0 = "2026-09-30 14:00 UTC"\nWVK24_EFFECTIVE = ')
    s = sub(s, '        "{WVK24_EFFECTIVE}": WVK24_EFFECTIVE,\n', '        "{WVK24_EFFECTIVE}": WVK24_EFFECTIVE,\n        "{WVK25_NOTICE}": WVK25_NOTICE,\n        "{WVK25_T0}": WVK25_T0,\n')
    s = sub(s, "- **Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})** \\\n",
            """- **Upcoming fork: wvk 25 — teacher → GLM-5.3-Flash, 262k context, miner \\
empty-thought rule, sequential stopping, R cap (notice {WVK25_NOTICE}, effective \\
{WVK25_T0} at the first duel boundary after that time)** — the frozen teacher moves \\
from `Qwen/Qwen3.8-27B` to `zai-org/GLM-5.3-Flash`; serving window 131,072 → \\
262,144 tokens (miners: serve `--max-model-len 262144`); a miner thought with < 10 \\
content tokens scores `min(z_R, z_A)` with an admission gate at 2× the teacher's \\
share; the paired margin is checked every 100 turns (crown when `margin − 2.6·SE > \\
δ` on two consecutive looks, futility stop, else 1,000); `z_R` capped at 0; fully \\
matched control published. Reign 21 stands; forward-only
- **Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})** \\
""")
    s = sub(s, "`code/affine/model_store.py`).\n\n**Step 2 — pre-flight the checkpoint directory (offline, free).**",
            """`code/affine/model_store.py`). **Context rule (from the wvk-25 fork, \\
{WVK25_T0}):** `config.json` must also declare an effective context window of \\
at least 262,144 tokens (`[submission].min_context_tokens`; `validate_repo_context` \\
— see the upcoming-fork section for the derivation). "Rope stays free" means \\
theta and type; scaling the window below 262,144 is a rejection.

**Step 2 — pre-flight the checkpoint directory (offline, free).**""")
    s = sub(s, "## Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})\n",
            """## Upcoming fork: wvk 25 — teacher → GLM-5.3-Flash, 262k context, scoring bundle (notice {WVK25_NOTICE}, effective {WVK25_T0})

**Notice {WVK25_NOTICE} (explicit dated operator directive, Jacob Steeves \\
2026-09-26 09:09 UTC "lets do this switch"). Effective {WVK25_T0}, at the first \\
duel boundary after that time. `weight_version_key` 24 → 25. Forward-only — \\
reign 21 stands, no re-verdicts, `min_submission_block` unchanged.** This \\
section becomes "Fork history: wvk 25" at the flip. Plan: the cutover sheet \\
published with the notice; live line on Discord after the first wvk-25 verdict.

**What changes.**
- **Teacher.** The frozen model the duel scores against moves from \\
`Qwen/Qwen3.8-27B` to `zai-org/GLM-5.3-Flash` (320B MoE, 18B active, MIT). Every \\
anchor (μ, σ) is the teacher's own leave-one-out statistic, so the rule \\
re-baselines itself; scores are not comparable across the fork.
- **Context.** Serving window 131,072 → 262,144 tokens on the eval pod and the \\
teacher swarm; prefix cap in D 110,000 → 255,744 tokens, measured with both the \\
teacher and the genesis tokenizer. Deep-trajectory turns dropped at the old cap \\
enter D over the following folds. Tool-call turns of D are re-derived under the \\
new teacher's chat template; bash / text turns are byte-identical.
- **Miner empty-thought rule** (`[duel.sd_meter].miner_empty_rule = "drop_typ"`). A \\
miner turn whose thought has fewer than 10 content tokens scores `min(z_R, z_A)` — \\
the typicality leg is dropped — instead of the −6 floor, the same rule wvk 24 \\
applies to the teacher's own references. Forfeits (no parseable action) keep the \\
−6 floor. **Admission gate** (`empty_gate_ratio = 2.0`): a side whose share of \\
such turns exceeds 2× the teacher's own share on the slice has those turns scored \\
at the floor. Why: since wvk 22 most of every crown margin came from the king \\
having more empty-thought turns than the challenger — a channel noise patches \\
could move; nothing about turn quality had to be better.
- **Sequential stopping** (`[duel].seq_enabled = true`, `seq_look_every = 100`, \\
`seq_k = 2.6`, `seq_consecutive = 2`). The paired margin is checked every 100 \\
scored turns in slice order; a duel stops with a crown when `margin − 2.6·SE > δ` \\
on two consecutive checks, stops for futility when even a 2.6·SE upward move \\
cannot reach δ, and otherwise runs to 1,000 turns and applies the standard rule. \\
Offline: 29/30 verdicts agree with the full slice, about twice the verdicts per \\
day. The verdict stamps every look (`sequential.looks`) and the stop.
- **R cap** (`r_cap_teacher = true`). `z_R := min(z_R, 0)` — a thought earns no \\
credit for predicting the teacher's action better than the teacher's own \\
alternative thoughts do. Ships unless the pre-flip probe on the final code shows \\
a problem.
- **Control.** The fully matched teacher-vs-king control (`control_matched`: king \\
scored on the same k−1 references as the held-out reference) is published on \\
every verdict and is the rollback signal.
- **Admission rule — 256k context** (`[submission].min_context_tokens = 262144`; \\
operator directive 2026-09-26 09:18 UTC "the new models should be required to \\
have a 256k sequence length"). A submission's `config.json` must declare an \\
effective context window of at least 262,144 tokens, derived the way vLLM \\
derives `max_model_len`: the smallest of `max_position_embeddings` / \\
`n_positions` / `seq_length` / `max_seq_len` / `max_sequence_length` / \\
`model_max_length` present (`text_config` when the root has none), multiplied by \\
`rope_scaling.factor` for `linear` / `dynamic` / `yarn` (yarn: \\
`original_max_position_embeddings × factor`); no multiplier for `llama3`, `su`, \\
`longrope`, `default` or no rope scaling. The genesis declares \\
`text_config.max_position_embeddings = 262144` with default rope and passes; a \\
config that shortens the window (e.g. 131,072) or scales it below 262,144 is \\
rejected before any download — at the R2 intake (`affine2|ready`), at dispatch \\
and at prefetch — with the decision `rejected_context_too_short`. Admission rule, \\
not a scoring change; `python affine/scripts/submit.py check <dir>` prints the \\
derived window. Verdicts stamp `duel_params.min_context_tokens`.

**What does NOT change.** δ = 0.2 sd, k_sigma = 2, forfeit floor −6, miner caps \\
(thought 4,096 with the 1.25× teacher-relative rule, action 768), reference cap \\
4,864, as-generated rendering, the architecture pin, the B licence, the \\
thought-length floor, the protocol probe.

**Pre-flight for miners.** `vllm serve <your checkpoint> --max-model-len 262144 \\
--tensor-parallel-size 2` must load and answer `/v1/completions` with finite \\
logprobs on an echo request. The genesis family is native 262k; do not shorten \\
rope in your config — `config.json` must derive to ≥ 262,144 tokens or the \\
submission is refused at intake. Nothing else changes in what you submit.

**Numbers behind it** (stored verdicts + a 404-turn GLM shadow): fully matched \\
teacher−king control +0.39 sd under Qwen → +0.80 under GLM (typicality +0.40 → \\
+1.68, action +0.29 → +0.53, R ≈ 0 under both); GLM reference yield 2.9 of 3 per \\
turn. Miner empty-thought rule on the wvk-22/23 crowns: reigns 17–21 all fall \\
under δ, reign 16 keeps its crown.

**Timeline.** {WVK25_NOTICE}: this notice; the datagen teacher seat moves to GLM \\
(data event). 2026-09-29: GLM teacher swarm pre-warmed next to the Qwen one. \\
**{WVK25_T0}: flip** at the first duel boundary; the first wvk-25 verdict stamps \\
`teacher.repo = zai-org/GLM-5.3-Flash`, `max_model_len 262144` and the new knobs.

---

## Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})
""")
    a.builder.write_text(s); print("patched", a.builder); return 0
if __name__ == "__main__":
    raise SystemExit(main())
