"""AGENTS.md §4 contract snapshot: current key = 21 with the 16–21 history in
front of the existing wvk-15 bullet (idempotent)."""
from pathlib import Path

p = Path("/home/const/subnet120/AGENTS.md")
s = p.read_text()
if "`weight_version_key = 21`" in s:
    print("already updated")
    raise SystemExit(0)
old = "- `weight_version_key = 15` (2026-09-12 ~17:01 UTC, explicit operator directive\n"
assert s.count(old) == 1, "anchor"
new = ("- `weight_version_key = 21` (2026-09-17 10:52 UTC, explicit dated operator\n"
       "  directive 10:07 UTC \"Remove the double eval on kings. This is too difficult.\n"
       "  Lets crown if any model passes 2 sigma like before … Feel free to crown the\n"
       "  last model which passed but failed the crown\": `confirmation_required = false`\n"
       "  — one 1,300-turn slice, `margin > max(2·SE, 0.002)` + gates crowns at once;\n"
       "  `chal-00556` (uid 175, `0f4029fd…`, slice-1 z 3.13) crowned retroactively as\n"
       "  reign 14 from its stored verdict, `via = retroactive_wvk21`; forward-only.\n"
       "  20 = 2026-09-16 14:57 UTC teacher-relative thought cap `thought_cap_ratio =\n"
       "  1.25` (cap_T = max(2048, ⌊1.25·L_T⌋), L_T = longest valid teacher reference\n"
       "  thought in teacher tokens); 19 = 2026-09-16 11:56 UTC confirmation slice\n"
       "  (`confirmation_required`, per-duel rule; retired by 21); 18 = 2026-09-15\n"
       "  21:13 UTC `max_thought_tokens` 1024→2048 + `text_fallback_at_tool_turns`;\n"
       "  17 = 2026-09-14 10:41 UTC `band_c` 2→4 + teacher-only `ref_max_tokens = 4096`;\n"
       "  16 = 2026-09-13 13:01 UTC per-duel crown restored (`crown_mode = \"duel\"`,\n"
       "  `near_miss_enabled = false`) after the reign-13 byte-copy under the window\n"
       "  rule — reign 13 and reign 12 uncrowned by operator directive; tensor-level\n"
       "  copy gate staged, not applied (PR #20). Older history follows:\n"
       "- `weight_version_key = 15` (2026-09-12 ~17:01 UTC, explicit operator directive\n")
p.write_text(s.replace(old, new))
print("AGENTS.md updated")
