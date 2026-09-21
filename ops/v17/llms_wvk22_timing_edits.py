"""Timing update to the wvk-22 "Upcoming change" section (Jacob's go 2026-09-18
15:11 UTC: flip only after the queue as of 15:11 has been judged under wvk 21)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"
s = BUILDER.read_text()
if "through `chal-00588`" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub("""slices (notice {WVK22_NOTICE}; effective at the first duel boundary after \\
today's shadow validation, projected within ~2–4 h)** — the turn score \\
""", """slices (notice {WVK22_NOTICE}; go given 15:11 UTC; effective after the queue \\
as of 15:11 UTC — through `chal-00588` — has been judged under wvk 21, \\
projected ~23:00 UTC)** — the turn score \\
""")
sub("""**Notice {WVK22_NOTICE} (explicit operator directive, 2026-09-18 10:04 / 10:25 / \\
10:40 UTC). Effective at the first duel boundary after today's shadow \\
validation — projected within ~2–4 h of this notice. The queue is empty, so \\
no submitted model is affected mid-flight; a model submitted from now on is \\
judged under the rule in force when its duel runs.** `weight_version_key` \\
""", """**Notice {WVK22_NOTICE} (explicit operator directive, 2026-09-18 10:04 / 10:25 / \\
10:40 UTC; go 15:11 UTC with the condition "only when the current queued \\
models have run"). Effective at the duel boundary right after the last entry \\
queued as of 15:11 UTC — `chal-00582` … `chal-00588` — has been judged under \\
wvk 21 (projected ~23:00 UTC). Submissions after 15:11 UTC are judged under \\
wvk 22. Reign 15 (`chal-00581`, crowned 14:59 UTC under wvk 21) stands.** `weight_version_key` \\
""")
BUILDER.write_text(s)
print("patched", BUILDER)
