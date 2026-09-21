#!/usr/bin/env python3
"""Post the wvk-22 notice: public SN120 channel (2 messages) + private Arbos
ops channel (1 message). Jacob 2026-09-18 10:40 UTC. Token from the live
validator env (~/.affine-validator.env: DISCORD_BOT_TOKEN_ARBOS_BITTENSOR).
Writes the links to ops/v17/discord_wvk22_notice.links.

  python ops/v17/discord_wvk22_notice.py --dry    # print, do not post
  python ops/v17/discord_wvk22_notice.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"   # 120・ⴷffine・ⴷ
PRIVATE_CH = "1510910974498967613"  # Arbos ops
HERE = Path(__file__).resolve().parent

PUBLIC_1 = """**Notice — wvk 22: the scoring rule becomes the sd-meter `min(z_R, typ_c, z_A)`, and slices go 1,300 → 1,000 turns.** Explicit operator directive 2026-09-18. **Effective at the first duel boundary after today's shadow validation — projected within ~2–4 h. The queue is empty, so no submitted model is affected mid-flight.** Forward-only: reign 14 stands, no re-verdicts, `min_submission_block` unchanged.

**Definition.** The teacher's joint over a turn has three factors: how it writes a *thought* for the task, how much that thought predicts its *action*, and how much its own thinking licenses an *action*. Your reply is compared with the teacher's own 3 samples on the same turn on each factor, in units of the teacher's own sample-to-sample spread (teacher-sd). **Turn score = minus the largest standardised deviation across the three.** Thought typicality is measured on *content* tokens only — the tokens whose teacher log-prob the task moves by more than 1 nat (|lpC(tok|x) − lpC(tok|∅)| > 1). A reply the teacher could have written scores ≈ 0; one that is off on any factor scores that deficit.

**The three legs (derivation).**
```
a_i = lpC(y_C^i|z_A) − lpC(y_C^i|∅);  R = τ·log mean_i exp(a_i/τ) − mean_i a_i   (centred Reason, unchanged, τ = 0.03)
b_i = [lpC(y_A|z_C^i) − lpC(y_A|∅)]·bytes(y_A);  A = τ·log mean_i exp(b_i/τ)     (action leg, summed nats)
m_c = mean lpC(tok|x) over your thought's content tokens; μ_c = same over the 3 teacher thoughts
z_R = (R−μ_R)/σ_R   z_A = (A−μ_A)/σ_A   typ_c = 2 − |m_c−μ_c|/σ_c
turn = min(z_R, typ_c, z_A); forfeit (no action / no </think>) = a fixed negative in sd; < 10 content tokens → typ_c at the floor
```
μ per turn = the teacher's own value (each reference scored as the miner against the other two, leave-one-out); σ = pooled within-turn spread of the references per dialect over the duel. Crown iff paired mean > max(k_sigma·SE, δ_sd), plus the unchanged thought-length floor and B gate."""

PUBLIC_2 = """**What changes (old → new).** `score_mode` min_rg → `sd_min_rga`; `n_turns` 1,300 → **1,000** (verdicts ~25 % faster, SE × 1.14); `band_c` / `band_floor` retired (content-token typicality replaces the band); δ, k_sigma and the forfeit floor re-expressed in sd units — **values calibrated to reproduce the current crown rate (δ ≈ 1.5× the 2σ bar; the −0.1 floor ≈ 2.4 sd below the mean valid turn); final numbers stamped in llms.txt at the flip.** Everything else stays: `</think>` required, prose at tool turns, teacher-relative thought cap, reference cap, protocol probe, admission rules.

**What you must do differently.** Act like the teacher, not just sound like it. Content matters, style does not: filler, restating the prompt, or the teacher's phrasing without its reasoning no longer earn typicality. Your action is now scored too — the teacher, thinking its own thought, must find your action likely. It no longer pays to sit at the edge of the old band: the score is continuous in your distance from the teacher's own samples. Nothing changes in what you emit (same prompt, dialects, caps, `</think>`).

**The honest line.** This improves the meter's robustness — it defeats every synthetic thought attack we tried (filler, generic, restated prompt, style skeleton, thinking-off) and ranks the teacher's own held-out replies first — but it is a better *distillation* meter, not a benchmark of coding or chat ability.

**Shadow now.** Since ~10:45 UTC every verdict carries `verdict.shadow.sd_meter` (the new score next to the live one: per-side mean, margin, SE, z, bind fractions, would-crown, echo cost). Full text + formulas: https://affine.io/llms.txt → "Upcoming change: wvk 22"."""

PRIVATE = """wvk 22 notice posted (Jacob 10:40 UTC: announce now, queue empty). Rule: sd-meter `min(z_R, typ_c[lift>1 nat], z_A)` in teacher-sd (LOO anchors, σ pooled per dialect, summed A), n_turns 1300→1000, band_c/band_floor retired, δ/k_sigma/forfeit in sd (provisional δ_sd 0.10, k_sigma 2, forfeit −2.4 sd; final from the shadow). Shadow live on the eval pod since 10:41 UTC (box `59ccb6c`, PR #36): every verdict stamps `shadow.sd_meter` (loo + frozen anchors, teacher-vs-king control, echo cost by tag). Gate to flip: calibration replay on the last 40 verdicts + 1–2 sane shadow duels → report → Jacob's go → `ops/v17/deploy_wvk22.sh` at a boundary. Rollback `ops/v17/rollback_wvk22.sh` if the first 3 verdicts: SE > 2× shadow expectation, teacher-vs-king control missing, or any leg binds > 80 %. Plan: docs/wvk22-plan.md."""


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t:
        return t
    for line in (Path.home() / ".affine-validator.env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")


def post(tok: str, channel: str, content: str) -> str:
    assert len(content) <= 2000, (channel, len(content))
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{channel}/messages",
        data=json.dumps({"content": content, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        mid = json.loads(r.read())["id"]
    return f"https://discord.com/channels/{GUILD}/{channel}/{mid}"


def main() -> int:
    dry = "--dry" in sys.argv
    for name, txt in (("public_1", PUBLIC_1), ("public_2", PUBLIC_2), ("private", PRIVATE)):
        print(f"{name}: {len(txt)} chars")
        assert len(txt) <= 2000, name
    if dry:
        print(PUBLIC_1, "\n---\n", PUBLIC_2, "\n---\n", PRIVATE)
        return 0
    tok = token()
    links = []
    if "--private-only" in sys.argv:
        # Public messages already posted: recover their links from the channel.
        req = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{PUBLIC_CH}/messages?limit=5",
            headers={"Authorization": f"Bot {tok}", "User-Agent": "affine-notice/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            msgs = json.loads(r.read())
        for m in reversed(msgs):
            for name, txt in (("public_1", PUBLIC_1), ("public_2", PUBLIC_2)):
                if m["content"][:80] == txt[:80]:
                    links.append((name, f"https://discord.com/channels/{GUILD}/{PUBLIC_CH}/{m['id']}"))
    else:
        links.append(("public_1", post(tok, PUBLIC_CH, PUBLIC_1)))
        links.append(("public_2", post(tok, PUBLIC_CH, PUBLIC_2)))
    links.append(("private", post(tok, PRIVATE_CH, PRIVATE)))
    out = HERE / "discord_wvk22_notice.links"
    out.write_text("\n".join(f"{k} {v}" for k, v in links) + "\n")
    for k, v in links:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
