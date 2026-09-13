"""Add the wvk-16 section to affine/scripts/build_llms_txt.py (idempotent).

Edits, all anchored on exact existing text (refuses if an anchor is missing
or the section is already present):
  1. WVK16_EFFECTIVE constant + substitution key.
  2. TOC: a wvk-16 line above the wvk-15 line; the near-miss TOC line says
     it is off since wvk 16.
  3. The "Since wvk 15 ... decides." sentence in the δ paragraph is
     replaced by the wvk-16 fact.
  4. Near-miss section: one paragraph saying it is off since wvk 16.
  5. New section "## Fork history: wvk 16 — per-duel crown restored" above
     the wvk-15 section.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK16_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK15_EFFECTIVE = "2026-09-12"\n',
    'WVK15_EFFECTIVE = "2026-09-12"\nWVK16_EFFECTIVE = "2026-09-13"\n')
sub('        "{WVK15_EFFECTIVE}": WVK15_EFFECTIVE,\n',
    '        "{WVK15_EFFECTIVE}": WVK15_EFFECTIVE,\n        "{WVK16_EFFECTIVE}": WVK16_EFFECTIVE,\n')

sub("""- Sequential near-miss (2026-09-11, no fork) — a first-slice margin in \\
the near-miss window draws a second seeded slice; the crown is decided on the \\
pooled 2 × `n_turns`
- **Fork history: wvk 15 — window-best crown (effective \\
{WVK15_EFFECTIVE})** — the king is frozen per {CROWN_WH} h window; the best \\
""", """- Sequential near-miss (2026-09-11, no fork; OFF since wvk 16) — a \\
first-slice margin in the near-miss window drew a second seeded slice; one \\
seeded slice decides again
- **Fork history: wvk 16 — per-duel crown restored (effective \\
{WVK16_EFFECTIVE})** — the 12 h window rule is retired after one day: reign \\
13 was a re-upload of reign 12's weights that won its window on noise; a \\
challenger crowns iff `margin > max(2·SE, 0.002)` on one slice, as under \\
wvk 3–14; reign 13 uncrowned, reign 12 stands
- Fork history: wvk 15 — window-best crown (effective \\
{WVK15_EFFECTIVE}, retired {WVK16_EFFECTIVE}) — the king was frozen per {CROWN_WH} h window; the best \\
""")

sub("""the revert is forward-only. Since wvk 15 ({WVK15_EFFECTIVE}) the crown is \\
decided per window (Fork history: wvk 15 below); the δ bar is still \\
computed and stamped as `duel_rule_wins`, but the window's best positive \\
margin decides.
""", """the revert is forward-only. For one day (wvk 15, {WVK15_EFFECTIVE} → \\
{WVK16_EFFECTIVE}) the crown was decided per 12 h window and the δ bar was \\
only stamped (`duel_rule_wins`); since wvk 16 ({WVK16_EFFECTIVE}) the bar \\
above decides again (Fork history: wvk 16 below).
""")

sub("""## Sequential near-miss (2026-09-11, no fork)

**A sampling-size rule, not a scoring change.** """, """## Sequential near-miss (2026-09-11, no fork; OFF since wvk 16)

**Off since wvk 16 ({WVK16_EFFECTIVE}):** `[duel].near_miss_enabled = false` \\
— one seeded `n_turns` slice decides the duel, as before 2026-09-11. The \\
text below describes the rule as it ran 2026-09-11 → {WVK16_EFFECTIVE}; \\
verdicts of that period stamp `near_miss.enabled = true` and replay \\
bit-identically.

**A sampling-size rule, not a scoring change.** """)

sub("""## Fork history: wvk 15 — window-best crown (effective {WVK15_EFFECTIVE})
""", """## Fork history: wvk 16 — per-duel crown restored (effective {WVK16_EFFECTIVE})

**Effective {WVK16_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-13 12:10 UTC: \\
"go back to the original delta threshold, the original system we had more \\
than a day ago; uncrown the copied model").** `weight_version_key = 16`; \\
`[duel].crown_mode = "duel"`, `near_miss_enabled = false`. `k_sigma = 2.0`, \\
`min_margin = 0.002`, `min_margin_mode = "fixed"`, `min_z = 0`, the per-turn \\
score min(R, G), the thought-length floor, the B gate and the reign chain / \\
payouts are unchanged.

**What changes.** The crown is decided per duel again, exactly as under \\
wvk 3–14: a challenger crowns iff its paired `mean(turn_c − turn_k)` over \\
ONE seeded `n_turns = 1300` slice is **> max(k_sigma·SE, δ) = max(2·SE, \\
0.002)**, its median stripped thought length is ≥ 80 characters and its B \\
pass rate is ≥ 0.30. No 12 h windows, no frozen king, no best-of-window, no \\
pooled confirmation slice, no near-miss second slice. Verdicts no longer \\
carry `crown_mode` / `window_id` / `duel_rule_wins`; `challenger_wins` \\
decides the crown again and `ranking_formula` no longer has the near-miss \\
suffix.

**Why.** Reign 13 (`chal-00461`, crowned 2026-09-13 09:59 UTC as the best \\
positive margin of window 2515) was **reign 12's weights re-uploaded**: \\
every one of the 1,026 tensors byte-identical (70,214,363,872 bytes), only \\
the shard split differed (16 files → 2), so every file hash and the \\
`model_digest` changed and the file-level copy check passed it. Its duel \\
margin was +0.00074 (z = 0.93); its confirmation slice was −0.00065 \\
(z = −0.91); the pooled margin was +0.000045 (z = 0.08) and the window rule \\
crowned on `pooled margin > 0`. The δ bar would have refused it (0.00074 < \\
0.002 and < 2·SE = 0.0016). A second entry in the next window was reign 12 \\
with 26 single-element edits. Under the restored bar a copy's noise margin \\
crowns with probability ≈ 1e-4 per duel; under the window rule ≈ 0.75 once \\
it was the window's best positive margin. The window / near-miss / \\
decaying-δ code stays in the tree behind its knobs so wvk-15 verdicts replay.

**Reign 13 is uncrowned; reign 12 stands.** The `crowned` history row of \\
`chal-00461` is rewritten to `event = "crown_revoked"` (fields kept, plus \\
`revoked_at` / `revoked_reason`), a `failed` row `rejected_model_copy` is \\
appended, and window 2516 (open at the flip, king = the same weights) is \\
recorded as `window_close` with `outcome = "king_stays_fork_wvk16"` — no \\
candidate of it was confirmed or crowned. Weights point at reign 12's hotkey \\
from the next weight sweep. No other verdict is re-decided. \\
`min_submission_block` unchanged. Identical-weights re-uploads of a crowned \\
model will not crown: they sit at the noise floor, below δ.

---

## Fork history: wvk 15 — window-best crown (effective {WVK15_EFFECTIVE}, retired {WVK16_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
