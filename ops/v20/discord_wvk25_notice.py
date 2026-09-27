#!/usr/bin/env python3
"""Post the wvk-25 fork notice (lead's text, ops/v19/discord_teacher_swap_post.md on the
cutover branch) — public channel in two messages + private ops line. Links → discord_wvk25.links."""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
GUILD = "799672011265015819"; PUBLIC_CH = "1381987595881414656"; PRIVATE_CH = "1510910974498967613"
REPO = Path(__file__).resolve().parents[2]

PUBLIC_1 = """**Upcoming fork: weight_version_key 24 → 25, effective Wednesday 2026-09-30 14:00 UTC (at the first duel boundary after that time).** Explicit operator directive 2026-09-26.

**What changes**
- **Teacher.** The frozen model the duel scores against moves from `Qwen/Qwen3.8-27B` to `zai-org/GLM-5.3-Flash` (320B MoE, 18B active, MIT). Every anchor (μ, σ) is the teacher's own leave-one-out statistic, so the rule re-baselines itself; scores are not comparable across the fork.
- **Context.** Serving window 131,072 → 262,144 tokens on the eval pod and the teacher swarm. Prefix cap in D 110,000 → 255,744 tokens, measured with both the teacher and the genesis tokenizer. Deep-trajectory turns dropped at the old cap enter D over the following folds.
- **Scoring bundle (same fork):**
 1. *Miner empty-thought rule.* A miner turn whose thought has fewer than 10 content tokens scores `min(z_R, z_A)` — the typicality leg is dropped — instead of the −6 floor, the same rule wvk 24 applies to the teacher's own references. Forfeits (no parseable action) keep the −6 floor. Admission gate: a side whose share of such turns exceeds 2× the teacher's own share on the slice has those turns scored at the floor. Why: since wvk 22 most of every crown margin came from the king having more empty-thought turns than the challenger — a channel noise patches could move; nothing about turn quality had to be better.
 2. *Sequential stopping.* The paired margin is checked every 100 scored turns; a duel stops with a crown when `margin − 2.6·SE > δ` on two consecutive checks, stops for futility when even a 2.6·SE upward move cannot reach δ, and otherwise runs to 1,000 turns. Offline: 29/30 verdicts agree with the full slice, ~2× verdicts per day.
"""

PUBLIC_2 = """**Scoring bundle, continued (wvk 25)**
 3. *R cap.* `z_R := min(z_R, 0)` — a thought earns no credit for predicting the teacher's action better than the teacher's own alternative thoughts do. (Ships unless the pre-flip probe on the final code shows a problem.)
 4. *Control.* The fully matched teacher-vs-king control (king scored on the same k−1 references as the held-out reference) is published on every verdict and is the rollback signal.

**What does NOT change**
- δ = 0.2 sd, k_sigma = 2, forfeit floor −6, miner caps (thought 4,096 with the 1.25× teacher-relative rule, action 768), ref cap 4,864, the architecture pin, the B licence, the thought-length floor.
- **Reign 21 stands.** Forward-only; no re-verdicts; `min_submission_block` unchanged.

**Pre-flight for miners**
- `vllm serve <your checkpoint> --max-model-len 262144 --tensor-parallel-size 2` must load and answer `/v1/completions` with finite logprobs on an echo request. The genesis family is native 262k; do not shorten rope in your config.
- Nothing else changes in what you submit.

**Numbers behind it** (stored verdicts + a 404-turn GLM shadow): fully matched teacher−king control +0.39 sd under Qwen → +0.80 under GLM (typicality +0.40 → +1.68, action +0.29 → +0.53, R ≈ 0 under both); GLM reference yield 2.9 of 3 per turn.

**Timeline**
- 2026-09-26: this notice; datagen teacher seat moves to GLM (data event).
- 2026-09-29: GLM teacher swarm pre-warmed next to the Qwen one.
- **2026-09-30 14:00 UTC: flip** at the first duel boundary; the first wvk-25 verdict stamps `teacher.repo = zai-org/GLM-5.3-Flash`, `max_model_len 262144`, the new knobs; a live line follows here.

Full text and knobs: https://affine.io/llms.txt → "Upcoming fork: wvk 25"."""

PRIVATE = ("wvk 25 notice posted (public, 2 messages) + llms.txt 'Upcoming fork: wvk 25' + dashboard #fork-notice banner (remove at T0: `python ops/v20/banner_wvk25.py --remove`). "
           "T0 Wed 2026-09-30 14:00 UTC; lead bc-3979ad9e (teacher swap / 262k / deploy_teacher_swap.sh), fork worker bc-ced503c8 (scoring knobs staged on box main: [duel.sd_meter] miner_empty_rule / empty_gate_ratio / r_cap_teacher, [duel] seq_*; all off until T0). "
           "Rollback signal: control_matched sign flip (typicality or A) in the first 20 wvk-25 verdicts.")

def token():
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t: return t
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="): return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")

def post(tok, ch, text):
    assert len(text) <= 2000, (ch, len(text))
    req = urllib.request.Request(f"https://discord.com/api/v10/channels/{ch}/messages",
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json", "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return f"https://discord.com/channels/{GUILD}/{ch}/{json.loads(r.read())['id']}"

def main():
    for n, t in (("public_1", PUBLIC_1), ("public_2", PUBLIC_2), ("private", PRIVATE)): print(n, len(t)); assert len(t) <= 2000, n
    if "--dry" in sys.argv: return 0
    tok = token(); links = [("public_1", post(tok, PUBLIC_CH, PUBLIC_1)), ("public_2", post(tok, PUBLIC_CH, PUBLIC_2)), ("private", post(tok, PRIVATE_CH, PRIVATE))]
    (Path(__file__).resolve().parent / "discord_wvk25.links").write_text("".join(f"{k} {v}\n" for k, v in links))
    for k, v in links: print(k, v)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
