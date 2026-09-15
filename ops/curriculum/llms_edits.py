#!/usr/bin/env python
"""Idempotent patch of affine/scripts/build_llms_txt.py for the adaptive
curriculum (stages 1 + 2, 2026-09-15): TOC line, the "Adaptive curriculum"
section after "Public data", and the ops/curriculum sources under code/.
Re-running is a no-op. `--mode apply` swaps the mode line at stage 3.

    python ops/curriculum/llms_edits.py [--mode shadow|apply] && (cd affine && python scripts/build_llms_txt.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "affine" / "scripts" / "build_llms_txt.py"

TOC_ANCHOR = "- Public data — full field-level description of every published object\n"
TOC_LINE = ("- Adaptive curriculum (data event, no wvk; shadow since 2026-09-15) — how \\\n"
            "often each stratum of D is drawn: public ledger of the king's per-turn \\\n"
            "scores → published weight rule → shadow vector now, applied after the \\\n"
            "seven-item check\n")

SECTION_ANCHOR = "---\n\n## Source of truth (linked)\n"
SECTION_MARK = "## Adaptive curriculum"

SECTION = r'''---

## Adaptive curriculum (data event, no wvk — {CURRICULUM_MODE_LINE})

How often each stratum of D is drawn is now set by a published rule, not by \
hand. Nothing in the scoring rule changes; this is the sampling side of D.

- **Ledger.** Every scored duel adds one row per (side, turn) — turn score, \
forfeit, R and G legs, reference yield — built only from \
`evals/{challenge_id}.json.gz`, `data/history_full.jsonl.gz` and the corpus \
index each verdict stamps (`slice.manifest_sha256`). Rolled up per stratum, \
cell (group × source × harness × action kind × prefix-depth bin) and group \
with an exponential decay of half-life 60 verdicts. Files: \
`{DATA}/curriculum/ledger/<sha>.rows.parquet`, `<sha>.rollup.parquet`, \
`<sha>.json` (window, θ, knobs, input shas), pointer \
`{DATA}/curriculum/ledger/latest.json`. The sha is over the canonical sorted \
row stream, so it does not depend on the Parquet writer. Rebuild: \
`python ops/curriculum/ledger.py --history {BASE}/data/history_full.jsonl.gz \
--evals {BASE}/evals --check <sha>` must print `check OK`.
- **Rule v1.** `w_s = (M~_s + 0.02)^gamma · S~_s`, where `M~` is the sitting \
king's miss rate on stratum s (a forfeit, or a live turn score under θ = the \
bottom quartile of the king's live turn scores in the window) and `S~` the \
share of live turns (≥ 2 distinct teacher references — where the meter can \
see the thought), both shrunk stratum → cell → group → corpus with prior count \
n_0 = 8. Group share ∝ Σ w over the group's slice keys — the phase-9 buckets, \
the unit a duel draws one turn from; a bucket weighs the mean w of the base \
strata it merges (`groups.json` also shows the sum over base strata as \
`share_raw_base_strata`) — floored (coding + terminal ≥ 0.40, every group ≥ \
half its static `[mix]` share), capped at 0.60, moved at most 5 points per \
fold from the live slice share. Inside a group a stratum gets 1–3 sub-strata \
(`<stratum>#k`) by weight rank, so it is drawn 1–3 times per duel. gamma = 1. \
Knobs: `[curriculum]` in `rollouts/rollouts/sources.toml`; code \
`code/ops/curriculum/rule.py` (the math), `weights.py` (the job).
- **Published per fold** under `{DATA}/curriculum/<epoch>/` (and immutably \
under `{DATA}/curriculum/weights/<weights_sha256>/`): `rule.json` (version, \
mode, knobs, θ, window, input shas), `weights.parquet` (per stratum: n, M~, \
S~, w, m_shadow, m_applied), `groups.json` (raw → floored → clamped → applied \
shares with a reason code per group), `recurrence.json` (draws per stratum in \
the last 50 verdicts; projected expected draws per turn per duel), \
`deficit_by_source.json`, `counterfactual.json` (the last 20 verdicts \
re-weighted under the shadow vector), `criterion.json` (the seven-item \
apply check), `diff.md` (one page vs the previous fold). Pointer: \
`{DATA}/curriculum/latest.json`. The corpus manifest carries \
`curriculum.{rule_version, mode, ledger_sha256, weights_sha256, \
manifest_sha256}`; verdicts stamp `slice.curriculum_version` = \
`v<rule_version>@<weights_sha256[:12]>` and `slice.curriculum`.
- **Mode.** `shadow` = weights published, the static `[mix]` still decides the \
slice. `apply` = the fold uses `share_after_clamp` and `m_applied`. Any \
rebuild mismatch or guard trip falls back to the static mix; `off` is the \
kill switch. The move to `apply` needs the seven-item criterion \
(`criterion.json`: rebuild sha, ≥ 95 % turn join, counterfactual mean |z| \
within ±10 % with no sign change at |z| ≥ 2, shadow vector stable across two \
folds, recurrence within cap, floors and cap holding, a hand read of the top \
ten upweighted strata) on two consecutive shadow folds.
- **What does not change:** the scoring rule min(R, G), the `[duel]` knobs, \
`weight_version_key`, seeded slices (reveal block hash), fresh teacher \
references per duel, the teacher-probe admission gate. Recurrence is capped \
at 3 draws per stratum per duel and published; strata whose king score rises \
only on repeated draws are flagged and reset to 1 (stage 4).
- **Why.** The sitting king's failure states, labelled by the teacher's fresh \
references at duel time, are the curriculum. Training on the upweighted strata \
is the intended behaviour; memorising specific turns is not, and is what the \
recurrence cap and the fresh-vs-recurring check watch.

'''

MODE_LINES = {
    "shadow": "shadow since 2026-09-15: weights published, static mix still applied",
    "apply": "shadow since 2026-09-15, applied since {APPLY_DATE}",
}

SOURCES_ANCHOR = '    ("ops/corpus_build.py", '
SOURCES_LINES = (
    '    ("ops/curriculum/rule.py", "adaptive curriculum, rule v1: shrinkage, "\n'
    '     "weights, floors, clamp, multiplicity, recurrence projection"),\n'
    '    ("ops/curriculum/ledger.py", "adaptive curriculum ledger: king per-turn "\n'
    '     "rows from evals/ + history + the stamped corpus index; --check "\n'
    '     "rebuilds and compares the sha"),\n'
    '    ("ops/curriculum/weights.py", "adaptive curriculum job: rollup + live "\n'
    '     "index -> weights.parquet / groups.json / recurrence.json"),\n'
)


def patch(text: str, mode: str, apply_date: str) -> str:
    if TOC_LINE not in text:
        if TOC_ANCHOR not in text:
            raise SystemExit("TOC anchor not found")
        text = text.replace(TOC_ANCHOR, TOC_ANCHOR + TOC_LINE, 1)
    mode_line = MODE_LINES[mode].replace("{APPLY_DATE}", apply_date)
    section = SECTION.replace("{CURRICULUM_MODE_LINE}", mode_line)
    if SECTION_MARK in text:
        start = text.index("---\n\n" + SECTION_MARK)
        end = text.index(SECTION_ANCHOR, start)
        text = text[:start] + section + text[end:]
    else:
        if SECTION_ANCHOR not in text:
            raise SystemExit("section anchor not found")
        text = text.replace(SECTION_ANCHOR, section + SECTION_ANCHOR, 1)
    if '"ops/curriculum/rule.py"' not in text:
        i = text.index(SOURCES_ANCHOR)
        # insert after the corpus_build entry (ends at the next '),\n')
        j = text.index("),\n", i) + len("),\n")
        text = text[:j] + SOURCES_LINES + text[j:]
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="shadow", choices=sorted(MODE_LINES))
    ap.add_argument("--apply-date", default="2026-09-17")
    ap.add_argument("--check", action="store_true", help="exit 1 if the patch is not applied")
    args = ap.parse_args()
    text = TARGET.read_text(encoding="utf-8")
    new = patch(text, args.mode, args.apply_date)
    if args.check:
        sys.exit(0 if new == text else 1)
    if new != text:
        TARGET.write_text(new, encoding="utf-8")
        print(f"patched {TARGET}")
    else:
        print("already patched")


if __name__ == "__main__":
    main()
