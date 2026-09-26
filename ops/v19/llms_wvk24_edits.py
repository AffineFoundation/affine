"""Add "Fork history: wvk 24" to affine/scripts/build_llms_txt.py (idempotent).
  python ops/v19/llms_wvk24_edits.py --date 2026-09-22 [--rollback DATE]"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

def sub(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:80]!r}")
    return s.replace(old, new)

def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date"); g.add_argument("--rollback", metavar="DATE"); ap.add_argument("--builder", type=Path, default=BUILDER)
    a = ap.parse_args(); s = a.builder.read_text()
    if a.rollback:
        if "WVK24_ROLLED_BACK" in s: print("already"); return 0
        s = sub(s, 'WVK24_EFFECTIVE = ', f'WVK24_ROLLED_BACK = "{a.rollback}"\nWVK24_EFFECTIVE = ')
        s = sub(s, '        "{WVK24_EFFECTIVE}": WVK24_EFFECTIVE,\n', '        "{WVK24_EFFECTIVE}": WVK24_EFFECTIVE,\n        "{WVK24_ROLLED_BACK}": WVK24_ROLLED_BACK,\n')
        s = sub(s, "## Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})\n",
                "## Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE}, **ROLLED BACK {WVK24_ROLLED_BACK}** to −12; wvk-24 verdicts stand)\n")
        a.builder.write_text(s); print("patched (rollback)"); return 0
    if "WVK24_EFFECTIVE" in s: print("already patched"); return 0
    s = sub(s, 'WVK23_EFFECTIVE = ', f'WVK24_EFFECTIVE = "{a.date}"\nWVK23_EFFECTIVE = ')
    s = sub(s, '        "{WVK23_EFFECTIVE}": WVK23_EFFECTIVE,\n', '        "{WVK23_EFFECTIVE}": WVK23_EFFECTIVE,\n        "{WVK24_EFFECTIVE}": WVK24_EFFECTIVE,\n')
    s = sub(s, "- **Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the \\\n",
            """- **Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})** \\
— a turn with no parseable action (or a thought with fewer than 10 content \\
tokens) scores −6 sd instead of −12; still strictly below the 1st percentile \\
of valid turns, so skipping a turn never pays; verdict SE falls ~11 %; nothing \\
else changes; forward-only, reign 21 stands
- **Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the \\
""")
    s = sub(s, "## Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the long end (effective {WVK23_EFFECTIVE})\n",
            """## Fork history: wvk 24 — forfeit floor −12 → −6 sd (effective {WVK24_EFFECTIVE})

**Effective {WVK24_EFFECTIVE} at the first duel dispatched after the eval pod \\
redeploy (explicit dated operator directive, Jacob Steeves 2026-09-23 20:17 \\
UTC: "Lets do this").** `weight_version_key = 24`; `[duel.sd_meter].forfeit_sd` \\
**−12 → −6**. Nothing else changes (δ = 0.2 sd, k_sigma = 2, 1,000-turn slices, \\
caps 4,096 / 4,864, as-generated rendering, `content_prefix = refs_max`, gates). \\
Forward-only — reign 21 stands, no re-verdicts, `min_submission_block` unchanged.

**Why.** At −12 the 2 % of turns that forfeit carried 48 % of the per-turn \\
score variance: a handful of forfeits dominated a verdict's standard error and \\
a miner's training signal. −6 is still strictly worse than any honest turn in \\
practice — the 1st percentile of valid turn scores is −4.6 sd (0.5th: −5.5), \\
only 0.3 % of valid turns score below −6, and even a miner that could predict \\
those turns perfectly and forfeited them would gain 0.007 sd per turn (3.5 % of \\
δ) — so skipping a turn still never pays, which is what the floor exists for. \\
Counterfactual on the last 30 verdicts: no decision changes, SE × 0.89 (median; \\
× 0.78 at best), z shifts within ± 0.5. A 2 % forfeit gap now costs ≈ 0.09 sd \\
(half a δ; it was one δ).

**What you must do.** Nothing new. Answer every turn with a parseable action \\
and close `</think>`; a forfeit costs −6 sd, worse than 99 % of valid turns. \\
The same value floors the typicality leg for a thought with fewer than 10 \\
content tokens.

**What you see.** `duel_params.sd_meter.forfeit_sd = -6`; verdict SE about 10 % \\
smaller for the same slice.

---

## Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the long end (effective {WVK23_EFFECTIVE})
""")
    a.builder.write_text(s); print("patched", a.builder); return 0

if __name__ == "__main__":
    raise SystemExit(main())
