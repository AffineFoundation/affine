"""Add "Fork history: wvk 23" to affine/scripts/build_llms_txt.py (idempotent).
  python ops/v18/llms_wvk23_edits.py --date 2026-09-22 [--rollback DATE]"""
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
        if "WVK23_ROLLED_BACK" in s: print("already"); return 0
        s = sub(s, 'WVK23_EFFECTIVE = ', f'WVK23_ROLLED_BACK = "{a.rollback}"\nWVK23_EFFECTIVE = ')
        s = sub(s, '        "{WVK23_EFFECTIVE}": WVK23_EFFECTIVE,\n', '        "{WVK23_EFFECTIVE}": WVK23_EFFECTIVE,\n        "{WVK23_ROLLED_BACK}": WVK23_ROLLED_BACK,\n')
        s = sub(s, "## Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the long end (effective {WVK23_EFFECTIVE})\n",
                "## Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the long end (effective {WVK23_EFFECTIVE}, **ROLLED BACK {WVK23_ROLLED_BACK}** to the wvk-22 settings; wvk-23 verdicts stand)\n")
        a.builder.write_text(s); print("patched (rollback)"); return 0
    if "WVK23_EFFECTIVE" in s: print("already patched"); return 0
    s = sub(s, 'WVK22_EFFECTIVE = "2026-09-18"\n', f'WVK22_EFFECTIVE = "2026-09-18"\nWVK23_EFFECTIVE = "{a.date}"\n')
    s = sub(s, '        "{WVK22_EFFECTIVE}": WVK22_EFFECTIVE,\n', '        "{WVK22_EFFECTIVE}": WVK22_EFFECTIVE,\n        "{WVK23_EFFECTIVE}": WVK23_EFFECTIVE,\n')
    s = sub(s, "- **Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn \\\n",
            """- **Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the \\
long end (effective {WVK23_EFFECTIVE})** — `max_thought_tokens` 2,048 → 4,096 \\
(teacher references 4,096 → 4,864 so they can think the full cap and act); the \\
typicality leg judges only the first K content tokens of your thought, K = the \\
teacher's longest reference in content tokens — extra deliberation is not \\
penalised, filler and pasting still are; nothing else changes; forward-only, \\
reign 21 stands
- **Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn \\
""")
    s = sub(s, "## Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (effective {WVK22_EFFECTIVE})\n",
            """## Fork history: wvk 23 — thought cap 4,096 + typicality one-sided on the long end (effective {WVK23_EFFECTIVE})

**Effective {WVK23_EFFECTIVE} at the first duel dispatched after the eval pod \\
redeploy (explicit dated operator directive, Jacob Steeves 2026-09-22 17:00 \\
UTC: "all of them and also 3 flipped", after the benchsuite thought-shrink \\
audit).** `weight_version_key = 23`. Forward-only — reign 21 stands, no \\
re-verdicts, `min_submission_block` unchanged. Everything else in `[duel]` — \\
the sd-meter, δ = 0.2 sd, forfeit −12 sd, 1,000-turn slices, as-generated \\
rendering, gates — is unchanged.

**Why.** Kings think less every generation: greedy GPQA reasoning chains \\
5.4k → 3.3k → 2.2k tokens across reigns 19 → 20 → 21 while GPQA fell 83.3 → \\
79.8 → 74.2, AIME chains 13.6k → 9.6k → 7.0k. On D the caps were not \\
binding (forfeits 0.2 %, median thought length rising), so the cap alone is \\
not expected to reverse the trend; the two changes remove the two places the \\
contract could still lean on short thoughts: the token budget and a \\
typicality leg that could read extra deliberation as atypical.

**What changes.**
- `[duel].max_thought_tokens` **2,048 → 4,096** for your rollouts (the \\
teacher-relative cap `max(4,096, 1.25 × L_T)` stays); `ref_max_tokens` 4,096 → \\
**4,864** = 4,096 + 768 so a teacher reference can think the full cap and \\
still act (8,192 was not taken: the three reference samples per turn set the \\
duel wall time, not the KV budget). Expect duels to take longer (~3× accepted \\
by the operator).
- `[duel.sd_meter].content_prefix = "refs_max"`: the typicality leg `typ_c = \\
2 − |m_c − μ_c|/σ_c` is computed on the **first K content tokens** of your \\
thought, K = the largest content-token count among the turn's three teacher \\
references. Content tokens beyond K are neither scored nor paid. The band is \\
still two-sided on that prefix: filler / off-task reasoning sits below the \\
references and is penalised; a pasted reference thought sits above and is \\
penalised (the RT-11 / RT-12 guards are intact). The references themselves \\
are scored in full (they define K). Offline probe on the last 30 wvk-22 \\
verdicts and a 173-turn re-echo: no verdict changes decision; the truncation \\
touches 28 % of miner thoughts, moving typ_c by +0.03…+0.07 sd on average; \\
the typicality-leg teacher-vs-king control stays positive (+0.36 → +0.27 sd, \\
z 3.8 → 3.0). The alternative — dropping the above-reference side of the \\
band — was rejected: it frees "too predictable" thoughts (38 % of miner \\
turns sit above μ_c today), which is orthogonal to length and reopens the \\
pasting hole.

**What you must do.** Nothing new in format. Think as long as the task \\
needs: reasoning beyond the teacher's longest reference is free on the \\
typicality leg, and the token budget is 4,096 thought tokens (plus 768 for \\
the action). Filler, restating the prompt, and pasting still lose.

**What you see.** `duel_params.max_thought_tokens = 4096`, \\
`duel_params.ref_max_tokens = 4864`, `duel_params.sd_meter.content_prefix = \\
"refs_max"`; rows carry `mc_za_full` / `n_content_za_full` / `k_ref_content` \\
next to the scored `mc_za` / `n_content_za`, so the wvk-22 rule can be \\
replayed on wvk-23 rows.

---

## Fork history: wvk 22 — the sd-meter `min(z_R, typ_c, z_A)` + 1,000-turn slices (effective {WVK22_EFFECTIVE})
""")
    a.builder.write_text(s); print("patched", a.builder); return 0

if __name__ == "__main__":
    raise SystemExit(main())
