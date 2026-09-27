"""llms.txt builder edits for the wvk 25 δ fork (2026-09-27): add "Fork history: wvk 25 — δ 0.20 → 0.10 sd"
(TOC bullet + section) and renumber the noticed GLM / 262k / scoring bundle from wvk 25 to wvk 26
(display text only; the WVK25_* placeholder names stay so the lead's llms step keeps its anchors).
Idempotent (guard: WVK25_DELTA_EFFECTIVE).

    python ops/v21/llms_wvk25_delta_edits.py --date 2026-09-27 --time "HH:MM UTC" [--builder PATH]
"""
from __future__ import annotations
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]


def sub(s, old, new, count=1):
    if s.count(old) != count:
        raise SystemExit(f"anchor {old[:80]!r}: expected {count}, got {s.count(old)}")
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", type=Path, default=REPO / "affine" / "scripts" / "build_llms_txt.py")
    ap.add_argument("--date"); ap.add_argument("--time")
    ap.add_argument("--rollback", metavar="DATE", help="append the rollback note (δ back to 0.20, wvk integer back) to the wvk-25 section")
    a = ap.parse_args()
    s = a.builder.read_text()
    if a.rollback:
        if "ROLLED BACK" in s:
            print("rollback note already present"); return 0
        s = sub(s, "**What you see.** `duel_params.sd_meter.min_margin_sd = 0.1`, verdict `min_margin` \\\n",
                f"**ROLLED BACK {a.rollback}** (explicit operator-directed revert after crown churn): \\\n"
                "`min_margin_sd` back to 0.2 and `weight_version_key` back to 24 at a duel boundary; \\\n"
                "verdicts judged under wvk 25 stand.\n\n"
                "**What you saw under wvk 25.** `duel_params.sd_meter.min_margin_sd = 0.1`, verdict `min_margin` \\\n")
        a.builder.write_text(s); print("rollback note added", a.builder); return 0
    if not (a.date and a.time):
        raise SystemExit("--date and --time are required for the flip edit")
    if "WVK25_DELTA_EFFECTIVE" in s:
        print("already"); return 0
    # constants + placeholder
    s = sub(s, 'WVK25_NOTICE = "2026-09-26"\n',
            f'WVK25_NOTICE = "2026-09-26"\nWVK25_DELTA_EFFECTIVE = "{a.date} {a.time}"  # δ 0.20 → 0.10 sd (wvk 24→25)\n')
    s = sub(s, '        "{WVK25_NOTICE}": WVK25_NOTICE,\n',
            '        "{WVK25_NOTICE}": WVK25_NOTICE,\n        "{WVK25_DELTA_EFFECTIVE}": WVK25_DELTA_EFFECTIVE,\n')
    # renumber the noticed bundle 25 -> 26 (display text)
    s = sub(s, "- **Upcoming fork: wvk 25 — teacher → GLM-5.3-Flash, 262k context, miner \\\n",
            "- **Upcoming fork: wvk 26 (was announced as 25) — teacher → GLM-5.3-Flash, 262k context, miner \\\n")
    s = sub(s, "**Context rule (from the wvk-25 fork, \\\n", "**Context rule (from the wvk-26 fork, \\\n")
    s = sub(s, "here on runs against reign 22 under the unchanged wvk-24 rule until the wvk-25 \\\n",
            "here on runs against reign 22 under the same scoring rule (δ 0.10 since wvk 25) until the wvk-26 \\\n")
    s = sub(s, "## Upcoming fork: wvk 25 — teacher → GLM-5.3-Flash, 262k context, scoring bundle (notice {WVK25_NOTICE}, effective {WVK25_T0})\n",
            "## Upcoming fork: wvk 26 — teacher → GLM-5.3-Flash, 262k context, scoring bundle (notice {WVK25_NOTICE} as \"wvk 25\", renumbered {WVK25_DELTA_EFFECTIVE}; effective {WVK25_T0})\n")
    s = sub(s, "duel boundary after that time. `weight_version_key` 24 → 25. Forward-only — \\\n",
            "duel boundary after that time. `weight_version_key` 25 → 26 (this bundle was noticed \\\nas wvk 25; the δ fork of {WVK25_DELTA_EFFECTIVE} took that number — nothing else about \\\nthe bundle changed). Forward-only — \\\n")
    s = sub(s, 'section becomes "Fork history: wvk 25" at the flip. Plan: the cutover sheet \\\n',
            'section becomes "Fork history: wvk 26" at the flip. Plan: the cutover sheet \\\n')
    s = sub(s, "published with the notice; live line on Discord after the first wvk-25 verdict.\n",
            "published with the notice; live line on Discord after the first wvk-26 verdict.\n")
    s = sub(s, "**{WVK25_T0}: flip** at the first duel boundary; the first wvk-25 verdict stamps \\\n",
            "**{WVK25_T0}: flip** at the first duel boundary; the first wvk-26 verdict stamps \\\n")
    # TOC bullet (above the operator-crown bullet) + section (above the operator-crown section)
    s = sub(s, "- **Operator crown {OPCROWN_DATE} — reign 22 (`chal-00687`, uid 62)** — crowned \\\n",
            """- **Fork history: wvk 25 — crown floor δ 0.20 → 0.10 sd (effective \\
{WVK25_DELTA_EFFECTIVE})** — explicit dated operator directive 2026-09-27 08:22 UTC \\
("Lower the validator margin to 0.1"); a challenger crowns when its paired margin \\
clears `max(2·SE, 0.10)` sd; nothing else changes; reign 22 stands; the noticed \\
GLM / 262k / scoring bundle is renumbered wvk 26
- **Operator crown {OPCROWN_DATE} — reign 22 (`chal-00687`, uid 62)** — crowned \\
""")
    s = sub(s, "## Operator crown {OPCROWN_DATE} — reign 22 (`chal-00687`, uid 62)\n",
            """## Fork history: wvk 25 — crown floor δ 0.20 → 0.10 sd (effective {WVK25_DELTA_EFFECTIVE})

**Explicit dated operator directive, Jacob Steeves 2026-09-27 08:22 UTC: "Lower \\
the validator margin to 0.1."** `weight_version_key` 24 → 25, flipped at the first \\
duel boundary after the directive. Forward-only — reign 22 stands, no \\
re-verdicts, `min_submission_block` unchanged. The GLM-5.3-Flash / 262k / scoring \\
bundle noticed on 2026-09-26 as "wvk 25" keeps its date (2026-09-30 14:00 UTC) and \\
content and becomes **wvk 26**.

**What changes.** One number: `[duel.sd_meter].min_margin_sd` 0.2 → **0.1**. A \\
challenger is crowned when its paired mean margin over the slice clears \\
`max(k_sigma·SE, δ)` = `max(2·SE, 0.10 sd)` and the gates pass (thought-length \\
floor, B licence, protocol probe). Scores, legs, anchors, the forfeit floor (−6), \\
the caps (4,096 / 768 / 4,864), the rendering, the empty-reference rule and the \\
1,000-turn slice are unchanged; a stored verdict replays to the same margin, SE \\
and z — only the crown decision moves.

**Counterfactual (108 sd-meter verdicts since wvk 22, each against its own \\
then-king).** 8 more crowns: `chal-00613` +0.197 (z 2.55), `00631` +0.154 (3.32), \\
`00643` +0.157 (3.56), `00649` +0.135 (3.20), `00651` +0.166 (3.45), `00652` \\
+0.174 (4.07), `00653` +0.186 (3.55), `00655` +0.112 (2.46); the 7 real crowns \\
stand; `chal-00687` (+0.073, reign 22 by operator crown) stays under. 7 → 15 \\
crowns in 9 days is an upper bound (each crown changes the next king).

**Known risk, accepted by the operator.** The slice SE at n ≈ 1000 is median \\
0.047 sd (p10 0.030, p90 0.110), so δ = 0.10 is ≈ 2.1 SE at the median (0.9–3.3 \\
across duels; δ = 0.20 was ≈ 4.2 SE) and binds on 59 of 108 verdicts instead of \\
94 — for the noisier half of duels the crown is decided by the 2σ test alone \\
(false-crown ≈ 2.3 % per duel for a zero-edge model, and an ε-copy of the king \\
crowns on noise at that rate). This is the shape of the 2026-08-21/22 experiment \\
(δ lowered to the v4 noise floor, wvk 7→8): 4 near-noise crowns in 18 h with the \\
winners' measured score drifting down across the chain — the winner's curse — \\
reverted the next day (wvk 8→9). δ = 0.20 was set on 2026-09-18 at ≈ 4 SE for that \\
reason. If churn returns, the revert is its own fork (`ops/v21/wvk25_delta_toml_edits.py \\
--revert`). Under the wvk-26 sequential rule the first look (n = 100) has SE ≈ 0.15, \\
so an early crown still needs `margin − 2.6·SE > 0.10` (≈ +0.49 at n = 100, +0.27 \\
at n = 500); δ = 0.10 bites at the full slice.

**What you see.** `duel_params.sd_meter.min_margin_sd = 0.1`, verdict `min_margin` \\
= 0.1, `weight_version_key = 25`, `ranking_formula` ends in `max(2·SE, 0.1)`.

## Operator crown {OPCROWN_DATE} — reign 22 (`chal-00687`, uid 62)
""")
    a.builder.write_text(s)
    print("patched", a.builder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
