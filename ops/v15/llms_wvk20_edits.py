"""Add the wvk-20 section to affine/scripts/build_llms_txt.py (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK20_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK19_EFFECTIVE = "2026-09-16"\n',
    'WVK19_EFFECTIVE = "2026-09-16"\nWVK20_EFFECTIVE = "2026-09-16"\n')
sub('        "{WVK19_EFFECTIVE}": WVK19_EFFECTIVE,\n',
    '        "{WVK19_EFFECTIVE}": WVK19_EFFECTIVE,\n'
    '        "{WVK20_EFFECTIVE}": WVK20_EFFECTIVE,\n'
    '        "{CAP_RATIO}": f"{float(d.get(\'thought_cap_ratio\', 0.0)):g}",\n'
    '        "{CAP_RULE}": (f" Per turn the thought cap is `max({int(d[\'max_thought_tokens\'])}, "\n'
    '                       f"floor({float(d.get(\'thought_cap_ratio\', 0.0)):g} × L_T))`, L_T = the longest "\n'
    '                       "valid teacher reference thought on that turn in teacher tokens (wvk 20)."\n'
    '                       if float(d.get("thought_cap_ratio", 0.0)) > 0 else ""),\n')

sub("""- **Fork history: wvk 19 — a crown must win twice (effective \\
""", """- **Fork history: wvk 20 — teacher-relative thought cap (effective \\
{WVK20_EFFECTIVE})** — per turn you may think up to {MAX_THOUGHT} tokens or \\
{CAP_RATIO}× the teacher's longest reference thought on that turn, whichever \\
is larger; nothing else changes
- **Fork history: wvk 19 — a crown must win twice (effective \\
""")

sub("""## Fork history: wvk 19 — a crown must win twice (effective {WVK19_EFFECTIVE})
""", """## Fork history: wvk 20 — teacher-relative thought cap (effective {WVK20_EFFECTIVE})

**Effective {WVK20_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-16 14:31 UTC).** \\
`weight_version_key = 20`; new `[duel].thought_cap_ratio = {CAP_RATIO}`. \\
`max_thought_tokens = {MAX_THOUGHT}`, `max_action_tokens = {MAX_ACTION}`, the \\
teacher's `ref_max_tokens = {REF_MAX_TOKENS}`, the score min(R, G), the crown \\
bar, the confirmation slice and the reign chain are unchanged.

**What changes.** Until now your reply was cut at a fixed \\
`{MAX_THOUGHT} + {MAX_ACTION}` tokens on every turn. From wvk 20 the thought \\
budget follows the teacher: on each turn the validator first samples the \\
teacher's k = 3 reference rollouts (as before), measures the **longest valid \\
reference thought** L_T in tokens of the teacher's own tokenizer \\
(`Qwen/Qwen3.8-27B`; a reference with no parseable action does not count), \\
and sets your thought cap for that turn to \\
**`cap_T = max({MAX_THOUGHT}, floor({CAP_RATIO} × L_T))`**. In words: you may \\
think up to {MAX_THOUGHT} tokens, or {CAP_RATIO}× as long as the teacher's \\
longest reference thought on that turn, whichever is larger. The action cap \\
({MAX_ACTION}) is unchanged. King and challenger read the same references, so \\
both sides get the same cap on every turn.

**Why.** The fixed cap cuts exactly where the task is hard enough that the \\
teacher itself thinks long. On the stored wvk-18 duels the rule relaxes the \\
cap on ~11% of turns (those where the teacher thinks > 1,640 tokens) and \\
there frees 45–47% of the remaining forfeits; it never lowers the cap. \\
Re-scoring those duels: 0 decision flips, the largest margin move +0.0003. \\
The largest possible cap is `floor({CAP_RATIO} × {REF_MAX_TOKENS})` tokens, \\
which fits the serving window with the 110k-token prefix cap.

**What you see.** `duel_params.thought_cap_rule = "max(fixed, {CAP_RATIO}*L_T)"`, \\
`duel_params.thought_cap_ratio`, `duel_params.thought_cap_tokenizer`; every \\
turn row in the duel record carries `cap_tokens` (the cap that side sampled \\
under) and `ref_thought_tokens` (L_T), so a replay is exact; per side \\
`n_turns_cap_raised`, `mean_cap_tokens`, `max_cap_tokens`. Pre-wvk-20 \\
verdicts have no `thought_cap_ratio` (= fixed cap) and replay unchanged.

**What changes for you.** You may think longer where the teacher does. Nothing \\
else. Forward-only: reign 13 stands; no re-verdicts; `min_submission_block` \\
unchanged.

---

## Fork history: wvk 19 — a crown must win twice (effective {WVK19_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
