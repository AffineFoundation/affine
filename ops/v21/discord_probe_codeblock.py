#!/usr/bin/env python3
"""Discord note: protocol-probe code-fence rule ENFORCED (admission rule, no wvk).

  python ops/v21/discord_probe_codeblock.py --flip-time "HH:MM UTC" [--first chal-xxxxx --first-block '...']
Links → ops/v21/discord_probe_codeblock.links
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"
PRIVATE_CH = "1510910974498967613"
REPO = Path(__file__).resolve().parents[2]


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t:
        return t
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")


def post(tok: str, ch: str, text: str) -> str:
    assert len(text) <= 2000, len(text)
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{ch}/messages",
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return f"https://discord.com/channels/{GUILD}/{ch}/{json.loads(r.read())['id']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-time", required=True)
    ap.add_argument("--first", default="")
    ap.add_argument("--first-block", default="")
    ap.add_argument("--private-only", action="store_true")
    a = ap.parse_args()
    first = f"\n\nFirst verdict under the rule: `{a.first}` — {a.first_block}" if a.first else ""
    public = (
        f"**Admission rule update ({a.flip_time}): the chat-protocol probe now checks code fences.** No scoring change, no `weight_version_key` change "
        f"(same class as the architecture pin).\n\n"
        f"**The rule, plainly:** when a prompt asks for *only the code*, your reply must not carry a malformed fence. **Bare code passes. Exactly one balanced "
        f"fenced block with a language tag (```` ```python … ``` ````) passes.** A lone closing ```` ``` ```` with no opening, an unbalanced fence, an untagged "
        f"fence, or a second block fails.\n\n"
        f"**How it is measured:** five public HumanEval-shaped prompts (\"your response should only contain the code for this function\"; HumanEval/0, 70, 134, 69, 103) "
        f"run through your own chat template with thinking on, 4 completions each at T = 0.7, `max_tokens` 2,048. They join the ten existing probe prompts; "
        f"**pass rate ≥ 0.90 pooled over 40 replies or the model is rejected** before scoring (`rejection_reason = \"protocol:…\"`, fence reasons named: "
        f"`close_only_fence`, `unbalanced_fence`, `fence_no_language`, `multiple_code_blocks`).\n\n"
        f"**Why:** reign 22's HumanEval fell 74.4 → 57.9 on replies that end with a lone closing fence and no opening — a habit that grew along the lineage "
        f"(close-only replies: genesis 0 → reign 20: 4 → reign 21: 35 → reign 22: 61 of 164; the code inside was correct). The scoring meter is blind to a few "
        f"fence bytes, so this is an admission check. Shadow read on 11 verdicts (2026-09-27/28): fence-clean models 0.95–1.00, close-only models 0.05–0.35.\n\n"
        f"**Pre-flight before you submit:** `python -m evalsrv.protocol_probe --base-url http://YOUR_VLLM/v1 --model YOUR_MODEL --enable-thinking --code-n-samples 4` "
        f"(code at https://affine.io/code/evalsrv/protocol_probe.py). Spec: https://affine.io/llms.txt → chat-protocol probe.{first}"
    )
    private = (
        f"[ops] protocol probe: five code_only cases ENFORCED at {a.flip_time} (operator 2026-09-28 12:15 UTC after the 11-verdict shadow read); "
        f"IDE strict case stays shadow; code_n_samples 4, code_max_tokens 2048 (chal-00721's 7 no_think_close at 1024 were budget, not fences); "
        f"pooled bar 0.90 over 40. Simulation on the 11 shadow verdicts: 7 pass / 4 reject (00714 0.35, 00715 0.05, 00718 0.80, 00719 0.75 — all close-only). "
        f"Reign 22 itself sits at 0.90 live (36/40) — a reign-22-like resubmission is on the edge. Deploy: ops/v21/deploy_probe_codeblock.sh "
        f"(pod redeploy at a boundary, no wvk). Rollback = put the five ids back into [protocol_probe].shadow_ids and redeploy.{first}"
    )
    tok = token()
    links = []
    if not a.private_only:
        links.append(post(tok, PUBLIC_CH, public))
    links.append(post(tok, PRIVATE_CH, private))
    (REPO / "ops/v21/discord_probe_codeblock.links").write_text("\n".join(links) + "\n")
    print("\n".join(links))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
