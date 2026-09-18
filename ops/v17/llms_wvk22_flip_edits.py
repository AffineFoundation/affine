"""At the wvk-22 flip: turn the "Upcoming change: wvk 22" notice in
affine/scripts/build_llms_txt.py into "Fork history: wvk 22 (effective DATE)"
and stamp the final calibrated numbers (idempotent).

  python ops/v17/llms_wvk22_flip_edits.py --date 2026-09-18 --delta-sd 0.10 --forfeit-sd -2.4
  python ops/v17/llms_wvk22_flip_edits.py --rollback 2026-09-18      # add the rolled-back line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"


def sub(s: str, old: str, new: str) -> str:
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:80]!r}")
    return s.replace(old, new)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date")
    g.add_argument("--rollback", metavar="DATE")
    ap.add_argument("--delta-sd", type=float)
    ap.add_argument("--forfeit-sd", type=float)
    a = ap.parse_args()
    s = BUILDER.read_text()
    if a.rollback:
        if "WVK22_ROLLED_BACK" in s:
            print("already patched (rollback)")
            return 0
        s = sub(s, 'WVK22_NOTICE = "2026-09-18"\n',
                f'WVK22_NOTICE = "2026-09-18"\nWVK22_ROLLED_BACK = "{a.rollback}"\n')
        s = sub(s, '        "{WVK22_NOTICE}": WVK22_NOTICE,\n',
                '        "{WVK22_NOTICE}": WVK22_NOTICE,\n        "{WVK22_ROLLED_BACK}": WVK22_ROLLED_BACK,\n')
        s = sub(s, "## Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (effective {WVK22_EFFECTIVE})\n",
                "## Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (effective {WVK22_EFFECTIVE}, **ROLLED BACK {WVK22_ROLLED_BACK}** to the wvk-21 rule: min(R, G), 1,300 turns; wvk-22 verdicts stand)\n")
        BUILDER.write_text(s)
        print("patched (rollback)", BUILDER)
        return 0
    if "WVK22_EFFECTIVE" in s:
        print("already patched")
        return 0
    if a.delta_sd is None or a.forfeit_sd is None:
        raise SystemExit("--delta-sd and --forfeit-sd required")
    d, f = f"{a.delta_sd:g}", f"{a.forfeit_sd:g}"
    s = sub(s, 'WVK22_NOTICE = "2026-09-18"\n',
            f'WVK22_NOTICE = "2026-09-18"\nWVK22_EFFECTIVE = "{a.date}"\n')
    s = sub(s, '        "{WVK22_NOTICE}": WVK22_NOTICE,\n',
            '        "{WVK22_NOTICE}": WVK22_NOTICE,\n        "{WVK22_EFFECTIVE}": WVK22_EFFECTIVE,\n')
    # TOC bullet
    s = sub(s, """- **Upcoming change: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn \\
slices (notice {WVK22_NOTICE}; effective at the first duel boundary after \\
today's shadow validation, projected within ~2–4 h)** — the turn score \\
""", """- **Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn \\
slices (notice {WVK22_NOTICE}, effective {WVK22_EFFECTIVE})** — the turn score \\
""")
    # Section header + effective paragraph
    s = sub(s, "## Upcoming change: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (notice {WVK22_NOTICE})\n",
            "## Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (effective {WVK22_EFFECTIVE})\n")
    s = sub(s, """**Notice {WVK22_NOTICE} (explicit operator directive, 2026-09-18 10:04 / 10:25 / \\
10:40 UTC). Effective at the first duel boundary after today's shadow \\
validation — projected within ~2–4 h of this notice. The queue is empty, so \\
no submitted model is affected mid-flight; a model submitted from now on is \\
judged under the rule in force when its duel runs.** `weight_version_key` \\
21 → 22 at the flip; forward-only — reign 14 stands, no re-verdicts, \\
`min_submission_block` unchanged. This section becomes "Fork history: wvk 22" \\
with the final stamped numbers when the flip lands.
""", f"""**Effective {{WVK22_EFFECTIVE}} at the first duel dispatched after the eval pod \\
redeploy (notice {{WVK22_NOTICE}} 10:46 UTC; explicit dated operator directive \\
2026-09-18 10:04 / 10:25 UTC, go to flip relayed by the coordinator).** \\
`weight_version_key = 22`; `[duel].score_mode = "sd_min_rga"`, `n_turns = 1000`; \\
`[duel.sd_meter]`: `min_margin_sd = {d}` (δ, teacher-sd), `k_sigma = 2.0`, \\
`forfeit_sd = {f}`, `content_lift_nats = 1.0`, `content_min_tokens = 10`, \\
`typicality_width = 2.0`, `a_norm_bytes = 1.0`, `anchor = "loo"`. Forward-only — \\
reign 14 stands, no re-verdicts, `min_submission_block` unchanged. Every \\
verdict stamps these under `duel_params.sd_meter`; the deciding numbers are \\
the verdict's `margin / se / z` (now in sd units) with the full breakdown \\
under `shadow.sd_meter` (`role = "rule"`).
""")
    s = sub(s, """units — values calibrated to reproduce the current crown rate (last 40 \\
verdicts: δ ≈ 1.5× the 2σ bar; the live −0.1 forfeit floor sits ≈ 2.4 sd \\
below the mean valid turn) — **final numbers are stamped here at the flip**. \\
""", f"""units — **δ = {d} sd, k_sigma = 2.0, forfeit = {f} sd**, calibrated on the \\
shadow read to reproduce the current crown rate (last 40 verdicts: δ ≈ 1.5× \\
the 2σ bar; the live −0.1 forfeit floor sits ≈ 2.4 sd below the mean valid \\
turn). \\
""")
    s = sub(s, """**Shadow first.** Since {WVK22_NOTICE} ~10:45 UTC every verdict carries \\
`verdict.shadow.sd_meter` (the new score computed next to the live one: per \\
side mean, margin, SE, z, bind fractions, would-crown, echo cost) — read \\
them to see where you stand before the flip.
""", """**Shadow read.** From {WVK22_NOTICE} 10:41 UTC to the flip every verdict carried \\
`verdict.shadow.sd_meter` (`role = "shadow"`); from the flip the same block \\
carries `role = "rule"` and is the deciding score.
""")
    BUILDER.write_text(s)
    print("patched", BUILDER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
