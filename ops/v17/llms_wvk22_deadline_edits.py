"""Max-timer wording for the wvk-22 notice (Jacob 15:18 UTC: 8 h max)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"
s = BUILDER.read_text()
if "23:11 UTC" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub("""as of 15:11 UTC — through `chal-00588` — has been judged under wvk 21, \\
projected ~23:00 UTC)** — the turn score \\
""", """as of 15:11 UTC — through `chal-00588` — has been judged under wvk 21, and no \\
later than the first duel boundary after 23:11 UTC; projected ≈ 23:45 UTC)** — the turn score \\
""")
sub("""queued as of 15:11 UTC — `chal-00582` … `chal-00588` — has been judged under \\
wvk 21 (projected ~23:00 UTC). Submissions after 15:11 UTC are judged under \\
""", """queued as of 15:11 UTC — `chal-00582` … `chal-00588` — has been judged under \\
wvk 21, and no later than the first duel boundary at or after 23:11 UTC (15:11 + \\
8 h, Jacob 15:18 UTC; a cutoff entry still queued then is judged under wvk 22, \\
the in-flight duel finishes under wvk 21). Projected ≈ 23:45 UTC. Submissions after 15:11 UTC are judged under \\
""")
BUILDER.write_text(s)
print("patched", BUILDER)
