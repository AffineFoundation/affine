"""Add the "Upcoming change: wvk 22" notice section to
affine/scripts/build_llms_txt.py (idempotent). Published 2026-09-18 on Jacob's
directive (10:40 UTC: "Lets make the announce that we are going to make this
switch right now"). The flip itself (wvk22_toml_edits.py) turns this section
into "Fork history: wvk 22" with the stamped final numbers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "affine" / "scripts" / "build_llms_txt.py"

s = BUILDER.read_text()
if "WVK22_NOTICE" in s:
    print("already patched")
    sys.exit(0)


def sub(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique/missing: {old[:70]!r}")
    s = s.replace(old, new)


sub('WVK21_EFFECTIVE = "2026-09-17"\n',
    'WVK21_EFFECTIVE = "2026-09-17"\nWVK22_NOTICE = "2026-09-18"\n')
sub('        "{WVK21_EFFECTIVE}": WVK21_EFFECTIVE,\n',
    '        "{WVK21_EFFECTIVE}": WVK21_EFFECTIVE,\n        "{WVK22_NOTICE}": WVK22_NOTICE,\n')

sub("""- **Fork history: wvk 21 — double evaluation removed (effective \\
""", """- **Upcoming change: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn \\
slices (notice {WVK22_NOTICE}; effective at the first duel boundary after \\
today's shadow validation, projected within ~2–4 h)** — the turn score \\
becomes minus the largest standardised deviation of your reply from the \\
teacher's own samples across the three factors of the teacher's joint \\
(thought typicality on content tokens, thought→action, action←thought), in \\
teacher-sd units; δ, k_sigma and the forfeit floor re-expressed in sd \\
units; band_c/band_floor retired; forward-only, reign 14 stands
- **Fork history: wvk 21 — double evaluation removed (effective \\
""")

sub("""## Fork history: wvk 21 — double evaluation removed (effective {WVK21_EFFECTIVE})
""", """## Upcoming change: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (notice {WVK22_NOTICE})

**Notice {WVK22_NOTICE} (explicit operator directive, 2026-09-18 10:04 / 10:25 / \\
10:40 UTC). Effective at the first duel boundary after today's shadow \\
validation — projected within ~2–4 h of this notice. The queue is empty, so \\
no submitted model is affected mid-flight; a model submitted from now on is \\
judged under the rule in force when its duel runs.** `weight_version_key` \\
21 → 22 at the flip; forward-only — reign 14 stands, no re-verdicts, \\
`min_submission_block` unchanged. This section becomes "Fork history: wvk 22" \\
with the final stamped numbers when the flip lands.

**Definition (exchangeability).** The teacher's joint over a turn has three \\
factors: how the teacher writes a **thought** for this task (thought \\
typicality, measured on the thought's *content* tokens — the tokens whose \\
teacher log-probability the task moves by more than 1 nat, \\
|lpC(tok|x) − lpC(tok|∅)| > 1), how much the thought predicts the teacher's \\
**action** (thought→action pointwise mutual information), and how much the \\
teacher's own thinking licenses the **action** (action←thought PMI). Your \\
reply is compared with the teacher's own k = 3 samples on the same turn on \\
each factor, in units of the teacher's own sample-to-sample spread \\
(teacher-sd). **Turn score = minus the largest of the three standardised \\
deviations.** A reply the teacher could have written scores near 0; a reply \\
that is off on any one factor scores that factor's deficit. The three legs \\
below are the derivation:

```
a_i   = lpC(y_C^i | z_A) − lpC(y_C^i | ∅)                         (per byte, i = 1..3)
R     = τ·log mean_i exp(a_i/τ) − mean_i a_i                       τ = 0.03 — the live centred Reason, unchanged
b_i   = [lpC(y_A | z_C^i) − lpC(y_A | ∅)] · bytes(y_A)             (summed nats)
A     = τ·log mean_i exp(b_i/τ)                                    action leg
m_c   = mean over content tokens of lpC(tok | x) for your thought z_A
μ_c   = the same statistic averaged over the three teacher thoughts z_C^i
z_R   = (R − μ_R)/σ_R      z_A = (A − μ_A)/σ_A      typ_c = 2 − |m_c − μ_c|/σ_c
turn  = min(z_R, typ_c, z_A)          valid reply
      = forfeit_sd                    no parseable action / no </think> (a fixed negative, in sd)
      typ_c = forfeit_sd              when your thought has fewer than 10 content tokens
```

μ_R, μ_A per turn = the teacher's own value on that turn, from each reference \\
scored as if it were the miner against the other two (leave-one-out; six \\
cross echoes lpC(y_C^i | z_C^j), shared by both sides). σ_R, σ_A, σ_c = the \\
pooled within-turn spread of those reference values, per dialect (bash, \\
tool_call, text, boxed, terminus_json), over the duel's turns. Score = mean \\
over the slice; crown iff the paired mean `turn_c − turn_k` > \\
`max(k_sigma·SE, δ_sd)`, plus the unchanged thought-length floor and B gate.

**What changes (old → new).** `score_mode` `min_rg` → `sd_min_rga`. \\
`n_turns` **1,300 → 1,000** (verdicts ~25 % faster; SE × 1.14). The grounding \\
band (`band_c`, `band_floor`) is retired — typicality on content tokens \\
replaces it. δ, `k_sigma` and the forfeit floor are re-expressed in sd \\
units — values calibrated to reproduce the current crown rate (last 40 \\
verdicts: δ ≈ 1.5× the 2σ bar; the live −0.1 forfeit floor sits ≈ 2.4 sd \\
below the mean valid turn) — **final numbers are stamped here at the flip**. \\
Everything else stays: `</think>` required, prose answers at tool turns, \\
teacher-relative thought cap, reference cap, protocol probe, admission rules.

**What you must do differently.** Act like the teacher, not just sound like \\
it. Content matters, style does not: the thought is judged on the tokens \\
the task makes informative, so generic filler, restating the prompt, or the \\
teacher's phrasing without its reasoning no longer earn typicality. Your \\
action is now scored too: the teacher, thinking its own thought, must find \\
your action likely. It no longer pays to sit at the edge of the old band — \\
the score is continuous in how far you are from the teacher's own samples. \\
Nothing changes in what you emit: same prompt, dialects, caps, `</think>`.

**The honest line.** This improves the meter's robustness — it defeats every \\
synthetic thought attack we tried (filler, generic, restated prompt, style \\
skeleton, thinking-off) and ranks the teacher's own held-out replies first — \\
but it is a better distillation meter, **not** a benchmark of coding or chat \\
ability. On the carded models it still ranks kings by teacher-likeness.

**Shadow first.** Since {WVK22_NOTICE} ~10:45 UTC every verdict carries \\
`verdict.shadow.sd_meter` (the new score computed next to the live one: per \\
side mean, margin, SE, z, bind fractions, would-crown, echo cost) — read \\
them to see where you stand before the flip.

---

## Fork history: wvk 21 — double evaluation removed (effective {WVK21_EFFECTIVE})
""")

BUILDER.write_text(s)
print("patched", BUILDER)
