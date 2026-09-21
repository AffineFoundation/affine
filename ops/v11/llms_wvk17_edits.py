"""Add the wvk-17 section to affine/scripts/build_llms_txt.py (idempotent).

Substitutions read from the toml so the text cannot drift: {BAND_C},
{BAND_FLOOR}, {REF_MAX_TOKENS}, {MINER_CAP}, {WVK17_EFFECTIVE}.
Edits (anchored):
  1. constants + substitution keys in _margin_subs
  2. band_c / cap facts in the score paragraph and formula block read from
     the toml instead of the literal "band_c = 2" / "1792"
  3. TOC line + "## Fork history: wvk 17" section above the wvk-16 one
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK17_EFFECTIVE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK16_EFFECTIVE = "2026-09-13"\n',
    'WVK16_EFFECTIVE = "2026-09-13"\nWVK17_EFFECTIVE = "2026-09-14"\n')
sub('        "{WVK16_EFFECTIVE}": WVK16_EFFECTIVE,\n',
    '        "{WVK16_EFFECTIVE}": WVK16_EFFECTIVE,\n'
    '        "{WVK17_EFFECTIVE}": WVK17_EFFECTIVE,\n'
    '        "{BAND_C}": f"{float(d.get(\'band_c\', 2.0)):g}",\n'
    '        "{BAND_FLOOR}": f"{float(d.get(\'band_floor\', 0.002)):g}",\n'
    '        "{MINER_CAP}": str(int(d["max_thought_tokens"]) + int(d["max_action_tokens"])),\n'
    '        "{REF_MAX_TOKENS}": str(int(d.get("ref_max_tokens") or (int(d["max_thought_tokens"]) + int(d["max_action_tokens"])))),\n')

sub("(`w = max(band_c·sd, band_floor)`, `band_c = 2`, `band_floor = 0.002`); the \\\n",
    "(`w = max(band_c·sd, band_floor)`, `band_c = {BAND_C}`, `band_floor = {BAND_FLOOR}`); the \\\n")
sub("                        (band_c = 2, band_floor = 0.002)\n",
    "                        (band_c = {BAND_C}, band_floor = {BAND_FLOOR})\n")

sub("""- **Fork history: wvk 16 — per-duel crown restored (effective \\
""", """- **Fork history: wvk 17 — wider grounding band + teacher reference cap \\
(effective {WVK17_EFFECTIVE})** — `band_c` 2 → {BAND_C} (the k = 3 band was so \\
tight the teacher's own thought fell outside it 25% of the time); the \\
teacher's references may run to {REF_MAX_TOKENS} tokens (miners stay at \\
{MINER_CAP}); nothing changes in what miners emit
- **Fork history: wvk 16 — per-duel crown restored (effective \\
""")

sub("""## Fork history: wvk 16 — per-duel crown restored (effective {WVK16_EFFECTIVE})
""", """## Fork history: wvk 17 — wider grounding band + teacher reference cap (effective {WVK17_EFFECTIVE})

**Effective {WVK17_EFFECTIVE} at the first duel dispatched after the eval \\
pod redeploy (explicit dated operator directive, 2026-09-14 10:30 UTC).** \\
`weight_version_key = 17`; `[duel].band_c = {BAND_C}` (was 2.0; \\
`band_floor = {BAND_FLOOR}` unchanged); new `[duel].ref_max_tokens = \\
{REF_MAX_TOKENS}` (teacher side only). Everything else — the per-turn score \\
min(R, G), the crown bar `margin > max(2·SE, 0.002)`, the thought-length \\
floor, the B gate, miners' token caps, the reign chain — is unchanged.

**1. Why the band is wider.** The G leg checks your thought's own teacher \\
likelihood `m = lpC(z_A|x)` against a band `mu ± w` built from the \\
teacher's own k = 3 reference thoughts on the same turn \\
(`w = max(band_c·sd(t_i), band_floor)`). Three samples give a noisy \\
estimate of that spread. We measured it: the teacher's OWN 4th thought — \\
a held-out sample from the very distribution the band is supposed to \\
describe — landed **outside** the c = 2 band on **25.5%** of turns. On \\
those turns G was penalising an honest, teacher-like thought for nothing \\
but sampling noise, and because the turn score is min(R, G), that noise \\
could decide the turn. `band_c = {BAND_C}` is the smallest width at which \\
≥ 90% of held-out teacher thoughts fall inside (90.3%). Checked on the \\
stored verdicts: no crown and no window decision flips; the reign-12 \\
margin moves from z 2.3 to z 2.9; the positive control (a genuine \\
held-out teacher thought vs a base-model thought) stays decisive at z 4.2 \\
(was 4.64); filler and generic thoughts still lose at z −8 / −12, so the \\
band still does its job against padding. Replay: `docs/scoring-today.md` \\
in the operator's store, summarised in the toml history paragraph.

**2. Why the teacher gets a longer reference.** Each turn is scored \\
against k = 3 teacher reference rollouts. Until now the teacher sampled \\
them under the same cap as miners, {MINER_CAP} tokens (thought + action). \\
On deep turns — long trajectories, hard states — the teacher's own \\
reference ran out of tokens on ~21% of samples, so those turns had fewer \\
or truncated references, or were dropped entirely (fewer than 2 refs). \\
The miner was judged against the weakest references exactly where the \\
task is hardest. With `ref_max_tokens = {REF_MAX_TOKENS}` the references \\
may run to {REF_MAX_TOKENS} tokens: in the replay, references per turn \\
rise 1.99 → 2.27 and the share of turns with a dead R leg falls 41% → \\
30%. **Only the teacher's reference budget changes.** Your \\
`max_thought_tokens = 1024` / `max_action_tokens = 768` are the same; a \\
reply longer than that is cut exactly as before.

**What changes for you.** Nothing in what you emit. G becomes fairer \\
(fewer honest thoughts pushed out of the band by noise), and more deep \\
turns become scorable with full references. The cost is ours: ~+15–20% \\
teacher echo compute and +25–40% teacher-side wall time per duel, so a \\
verdict takes roughly 50 minutes instead of ~40. Verdicts stamp \\
`duel_params.band_c`, `duel_params.band_floor` and \\
`duel_params.ref_max_tokens` (absent on pre-wvk-17 verdicts, meaning the \\
shared cap).

**Forward-only.** Reign 12 stands; no re-verdicts; `min_submission_block` \\
unchanged. Pre-wvk-17 verdicts replay bit-identically through their own \\
stamped `band_c` and the shared cap.

---

## Fork history: wvk 16 — per-duel crown restored (effective {WVK16_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
