"""Add the wvk-19 section to affine/scripts/build_llms_txt.py (idempotent).

Substitutions from the toml: {CONFIRMATION} ("true"/"false"), {WVK19_EFFECTIVE};
the crown sentence and formula block gain the confirmation clause when the
knob is on (read from the toml so the text cannot drift).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK19_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK18_EFFECTIVE = "2026-09-15"\n',
    'WVK18_EFFECTIVE = "2026-09-15"\nWVK19_EFFECTIVE = "2026-09-16"\n')
sub('        "{WVK18_EFFECTIVE}": WVK18_EFFECTIVE,\n',
    '        "{WVK18_EFFECTIVE}": WVK18_EFFECTIVE,\n'
    '        "{WVK19_EFFECTIVE}": WVK19_EFFECTIVE,\n'
    '        "{CONFIRMATION}": str(bool(d.get("confirmation_required", False))).lower(),\n'
    '        "{CONFIRM_CLAUSE}": (" **A pass is then confirmed on a second independent "\n'
    '                             "1,300-turn slice: that slice\'s own margin must be > 0 and "\n'
    '                             "the pooled margin over both slices must clear "\n'
    '                             "`max(k_sigma·SE_pooled, δ)`; otherwise the verdict is "\n'
    '                             "`confirmation_failed` and the king stands (wvk 19).**"\n'
    '                             if d.get("confirmation_required", False) else ""),\n'
    '        "{CONFIRM_FORMULA}": ("\\n                        THEN confirmation slice (wvk 19): margin_2 > 0"\n'
    '                              "\\n                        AND pooled margin > max(k_sigma·SE_pooled, δ)"\n'
    '                              if d.get("confirmation_required", False) else ""),\n')

sub("(`B = lpC(y_A|z_A) − lpC(y_A|∅) ≥ 0.02`, no leakage). No lpA gates.{CROWN_RULE}\n",
    "(`B = lpC(y_A|z_A) − lpC(y_A|∅) ≥ 0.02`, no leakage). No lpA gates.{CONFIRM_CLAUSE}{CROWN_RULE}\n")
sub("                        AND B pass rate ≥ causality_gamma{MIN_Z_FORMULA}\n",
    "                        AND B pass rate ≥ causality_gamma{MIN_Z_FORMULA}{CONFIRM_FORMULA}\n")

sub("""- **Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool \\
""", """- **Fork history: wvk 19 — a crown must win twice (effective \\
{WVK19_EFFECTIVE})** — a duel that clears the bar is confirmed on a second \\
independent 1,300-turn slice (own margin > 0, pooled margin over the bar) \\
before it crowns; a failed confirmation is a loss; noise crowns fall from \\
≈0.5% to ≈0.01% per attempt; honest improvers wait ~40 min more
- **Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool \\
""")

sub("""## Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool turns (effective {WVK18_EFFECTIVE})
""", """## Fork history: wvk 19 — a crown must win twice (effective {WVK19_EFFECTIVE})

**Effective {WVK19_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-16 11:10 UTC).** \\
`weight_version_key = 19`; new `[duel].confirmation_required = \\
{CONFIRMATION}`. The per-duel rule is unchanged: you beat the king on a \\
1,300-turn slice iff your paired margin `mean(turn_c − turn_k)` clears \\
`max(k_sigma·SE, δ)` with `k_sigma = 2`, `δ = 0.002`, plus the thought-length \\
floor and the B gate. What changes is what happens next.

**A pass is a candidate, not a crown.** When your first slice clears the \\
bar, the validator immediately scores a **second, independent slice** of \\
1,300 turns against the same king: seed `blake2b(block_hash ‖ hotkey ‖ \\
"|slice1")` (derived from your reveal block, so nobody can pick it), turns \\
disjoint from the first slice, fresh teacher references, the same engines \\
(already loaded, so about 40 more minutes). You are crowned only if **(a)** \\
the second slice's own paired margin is **> 0** and **(b)** the **pooled** \\
margin over both slices (exact pooling of the two samples' n / mean / SE) \\
clears **`max(k_sigma·SE_pooled, δ)`** — the same bar, now over 2,600 turns. \\
If either fails, the verdict is recorded as `confirmation_failed`: a loss \\
like any other, the king stands, your hotkey's slot is consumed, nothing is \\
re-queued. You may submit again from another hotkey as before.

**Why.** Every one of the 15 crowns since the wvk-10 reset rested on a \\
single slice. The only two crowns that ever got a second slice — under the \\
retired 12-hour-window rule — saw it come back at or below zero, and both \\
were later revoked. A single slice at `δ = 0.002 ≈ 2.6·SE` lets a challenger \\
that is exactly as good as the king crown by luck on about **0.5% of \\
attempts**; with the confirmation that falls to about **0.01%** (Monte \\
Carlo at today's SE 0.00078). A real improver loses nothing but ~40 \\
minutes: a genuine +0.003 margin clears the pooled bar with z ≈ 5. First \\
slices cleared the bar on 15 of 447 scored verdicts (3.4%), so the \\
confirmation runs a few times a week, not on every duel.

**What you see.** `duel_params.confirmation_required = true`; on a duel whose \\
first slice passed, `verdict.confirmation = {seed, n, margin, se, z, \\
pooled_n, pooled_margin, pooled_se, pooled_z, bar, passed, …}`; the \\
confirmation slice's full record is `evals/<challenge_id>-confirm.json.gz`. \\
A crowned row carries `challenger_wins = true` and `confirmation.passed = \\
true`; a failed confirmation carries `rejection_reason = \\
"confirmation_failed"`.

**What changes for you.** Nothing in what you emit. A winner must win twice; \\
honest improvers wait ~40 minutes longer; noise crowns stop. Forward-only: \\
reign 13 stands; no re-verdicts; `min_submission_block` unchanged. Pre-wvk-19 \\
verdicts carry no `confirmation_required` stamp (= false) and replay \\
unchanged.

---

## Fork history: wvk 18 — miner thought cap 2,048 + prose answers at tool turns (effective {WVK18_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
