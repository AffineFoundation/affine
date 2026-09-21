"""Add the wvk-18 section to affine/scripts/build_llms_txt.py (idempotent).

Substitutions from the toml: {MAX_THOUGHT}, {MAX_ACTION}, {MINER_CAP} (already
exists), {TEXT_FALLBACK}, {WVK18_EFFECTIVE}. The wvk-17 section's "miners'
cap" wording becomes the historical literal 1792 so it stays true after the
flip; the wvk-12 "what to do" cap reads the live value.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK18_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK17_EFFECTIVE = "2026-09-14"\n',
    'WVK17_EFFECTIVE = "2026-09-14"\nWVK18_EFFECTIVE = "2026-09-15"\n')
sub('        "{WVK17_EFFECTIVE}": WVK17_EFFECTIVE,\n',
    '        "{WVK17_EFFECTIVE}": WVK17_EFFECTIVE,\n'
    '        "{WVK18_EFFECTIVE}": WVK18_EFFECTIVE,\n'
    '        "{MAX_THOUGHT}": str(int(d["max_thought_tokens"])),\n'
    '        "{MAX_ACTION}": str(int(d["max_action_tokens"])),\n'
    '        "{TEXT_FALLBACK}": str(bool(d.get("text_fallback_at_tool_turns", False))).lower(),\n')

# wvk-17 section: historical literals
sub("them under the same cap as miners, {MINER_CAP} tokens (thought + action). \\\n",
    "them under the same cap as miners, 1792 tokens (thought + action). \\\n")
sub("""`max_thought_tokens = 1024` / `max_action_tokens = 768` are the same; a \\
reply longer than that is cut exactly as before.
""", """`max_thought_tokens = 1024` / `max_action_tokens = 768` were the same; a \\
reply longer than that was cut exactly as before (the thought cap rose to \\
2048 with wvk 18, see below).
""")
sub("""teacher's references may run to {REF_MAX_TOKENS} tokens (miners stay at \\
{MINER_CAP}); nothing changes in what miners emit
""", """teacher's references may run to {REF_MAX_TOKENS} tokens (miners stayed at \\
1792); nothing changes in what miners emit
""")
# wvk-12 guidance: live cap
sub("inside the token cap (`max_thought_tokens + max_action_tokens = 1792`). A \\\n",
    "inside the token cap (`max_thought_tokens + max_action_tokens = {MINER_CAP}`). A \\\n")

# TOC
sub("""- **Fork history: wvk 17 — wider grounding band + teacher reference cap \\
""", """- **Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool \\
turns (effective {WVK18_EFFECTIVE})** — you may think up to {MAX_THOUGHT} \\
tokens (was 1024; action cap {MAX_ACTION} unchanged); at a tool-call turn a \\
visible prose reply with no tool call is scored as a `text` action instead \\
of forfeiting — for the teacher's references too
- **Fork history: wvk 17 — wider grounding band + teacher reference cap \\
""")

# section
sub("""## Fork history: wvk 17 — wider grounding band + teacher reference cap (effective {WVK17_EFFECTIVE})
""", """## Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool turns (effective {WVK18_EFFECTIVE})

**Effective {WVK18_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-15 20:12 UTC).** \\
`weight_version_key = 18`; `[duel].max_thought_tokens = {MAX_THOUGHT}` (was \\
1024; `max_action_tokens = {MAX_ACTION}` and the teacher's `ref_max_tokens = \\
{REF_MAX_TOKENS}` unchanged); new `[duel].text_fallback_at_tool_turns = \\
{TEXT_FALLBACK}`. The per-turn score min(R, G), the crown bar `margin > \\
max(2·SE, 0.002)`, the thought-length floor, the B gate, \\
`require_think_close` and the reign chain are unchanged.

**1. You may think up to {MAX_THOUGHT} tokens.** A reply is cut at \\
`max_thought_tokens + max_action_tokens = {MINER_CAP}` tokens; a reply that \\
is cut before its action forfeits the turn (−0.1). At the old 1,024-token \\
thought cap that happened to careful thinkers for no reason the meter \\
cares about: on fresh samples the plain teacher's own forfeits fall from \\
21% at 1,024 to 7% at 2,048, and a coached (longer-thinking) teacher's from \\
29% to 16%. The G leg still judges your thought against the teacher's own \\
reference thoughts, so a longer thought earns nothing by being long — it \\
just stops being cut off. Expected cost on our side: a few percent of \\
verdict time (the wvk-17 reference-cap raise cost ≈ +5%).

**2. A prose answer at a tool-call turn no longer forfeits — when it is \\
what the teacher would do too.** Turns whose dialect is `tool_call` expect \\
a tool call as the action. Sometimes the right move is to answer in words \\
(report the result, say the task is done, ask for the missing piece), and \\
the teacher itself does exactly that on about 14% of its own samples at \\
such turns. Until now such a reply was a dropped reference for the teacher \\
and a forfeit for you. From wvk 18, at a `tool_call` turn a reply that \\
**closed `</think>`**, contains **no tool call** and has a **non-empty \\
visible reply** is scored as a `text` action — the whole visible reply — \\
exactly like a `text` turn: your prose is scored against the teacher's \\
references, prose or tool call alike, and the teacher's prose references \\
count. Over the last 20 verdicts about 69 reference slots per verdict were \\
empty at tool-call turns and 17 (king) / 23 (challenger) miner turns per \\
verdict forfeited there; the prose share of those converts. **What stays a \\
forfeit:** an empty visible reply, and a reply that never closes `</think>` \\
(`require_think_close`): reasoning-only output still scores −0.1, so the \\
wvk-13 hole stays closed. Nothing changes at `bash`, `boxed`, `text` or \\
`terminus_json` turns. Verdicts publish `n_text_fallback` per side and for \\
the teacher (how many samples took this path).

**What changes for you.** You may think longer, and you may answer in prose \\
at a tool turn when that is the right answer. Nothing else. Forward-only: \\
reign 13 stands; no re-verdicts; `min_submission_block` unchanged. Verdicts \\
stamp `duel_params.max_thought_tokens` and \\
`duel_params.text_fallback_at_tool_turns`; wvk ≤ 17 verdicts carry their own \\
stamps (1024 / absent = false) and replay unchanged.

---

## Fork history: wvk 17 — wider grounding band + teacher reference cap (effective {WVK17_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
