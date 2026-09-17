"""Add the wvk-21 section to affine/scripts/build_llms_txt.py (idempotent);
mark the wvk-19 confirmation section retired."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK21_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK20_EFFECTIVE = "2026-09-16"\n',
    'WVK20_EFFECTIVE = "2026-09-16"\nWVK21_EFFECTIVE = "2026-09-17"\n')
sub('        "{WVK20_EFFECTIVE}": WVK20_EFFECTIVE,\n',
    '        "{WVK20_EFFECTIVE}": WVK20_EFFECTIVE,\n        "{WVK21_EFFECTIVE}": WVK21_EFFECTIVE,\n')

sub("""- **Fork history: wvk 20 — teacher-relative thought cap (effective \\
""", """- **Fork history: wvk 21 — double evaluation removed (effective \\
{WVK21_EFFECTIVE})** — the wvk-19 confirmation slice is gone: a challenger \\
crowns on ONE 1,300-turn slice when its margin clears `max(2·SE, 0.002)`, as \\
under wvk 3–18; `chal-00556`, which passed that bar but failed the \\
confirmation, was crowned retroactively
- **Fork history: wvk 20 — teacher-relative thought cap (effective \\
""")
sub("""- **Fork history: wvk 19 — a crown must win twice (effective \\
{WVK19_EFFECTIVE})** — a duel that clears the bar is confirmed on a second \\
""", """- Fork history: wvk 19 — a crown must win twice (effective \\
{WVK19_EFFECTIVE}, **retired {WVK21_EFFECTIVE}**) — a duel that cleared the bar was confirmed on a second \\
""")

sub("""## Fork history: wvk 20 — teacher-relative thought cap (effective {WVK20_EFFECTIVE})
""", """## Fork history: wvk 21 — double evaluation removed (effective {WVK21_EFFECTIVE})

**Effective {WVK21_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-17 10:07 UTC: \\
"Remove the double eval on kings. This is too difficult. Lets crown if any \\
model passes 2 sigma like before.").** `weight_version_key = 21`; \\
`[duel].confirmation_required = false`.

**What changes.** The confirmation slice introduced by wvk 19 is removed \\
because it made crowning too hard. The crown rule is again the one of \\
wvk 3–18: you dethrone the king when your paired margin `mean(turn_c − \\
turn_k)` over **one** 1,300-turn slice clears **`max(k_sigma·SE, δ) = \\
max(2·SE, 0.002)`**, plus the thought-length floor and the B gate. No second \\
slice, no pooled test. Everything from wvk 20 stays: the teacher-relative \\
thought cap, the caps, min(R, G), the reign chain.

**Retroactive crown.** One duel was rejected by the confirmation slice alone: \\
`chal-00556` (uid 175), whose first slice cleared the bar (margin +0.0022, \\
z 3.13) and whose confirmation slice fell short (pooled +0.0014 < δ). Under \\
the same directive it is crowned from its stored slice-1 verdict — no \\
re-duel — as the next reign, with its payout window starting at the crown. \\
The original `verdict` row stays in the history; the `crowned` row carries \\
`via = "retroactive_wvk21"` and the confirmation numbers for audit. No other \\
verdict since wvk 19 was rejected on the confirmation alone.

**What you see.** `duel_params.confirmation_required = false`; `challenger_wins` \\
decides the crown again on the first slice. Forward-only otherwise; \\
`min_submission_block` unchanged; wvk-19/20 verdicts keep their stamps and \\
replay unchanged.

---

## Fork history: wvk 20 — teacher-relative thought cap (effective {WVK20_EFFECTIVE})
""")
sub("""## Fork history: wvk 19 — a crown must win twice (effective {WVK19_EFFECTIVE})
""", """## Fork history: wvk 19 — a crown must win twice (effective {WVK19_EFFECTIVE}, retired {WVK21_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
