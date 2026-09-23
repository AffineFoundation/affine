"""Extend the "Fork history: wvk 24" section with the 20:47 addendum (items 3 and 6). Idempotent."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--builder", type=Path, default=REPO / "affine" / "scripts" / "build_llms_txt.py"); a = ap.parse_args()
    s = a.builder.read_text()
    if "k-matched" in s: print("already"); return 0
    old = """**What you see.** `duel_params.sd_meter.forfeit_sd = -6`; verdict SE about 10 % \\
smaller for the same slice.
"""
    if s.count(old) != 1: raise SystemExit("anchor")
    new = """**Addendum, same fork (operator directive 2026-09-23 20:47 UTC), effective at the \\
eval pod redeploy that followed (the first wvk-24 duel, `chal-00678`, ran with the \\
floor only).**
- **Empty-thought reference rule** — `[duel.sd_meter].ref_min_content = 10`, \\
`typ_min_refs = 2`: a teacher reference whose thought has fewer than 10 content \\
tokens does not anchor the typicality leg (it is left out of μ_c and σ_c instead \\
of entering them as a mean over a handful of tokens), and a turn with fewer than \\
2 content-bearing references scores `min(z_R, z_A)` — the typicality leg is \\
dropped for that turn (counted in `n_leg_dropped.Gc`). Live data: 7.5 % of \\
references, ~4.8 % of turns. Counterfactual on the last 30 verdicts (with the \\
−6 floor): no decision changes.
- **Teacher-vs-king control, k-matched and floor-dropped** — published on every \\
verdict as `shadow.sd_meter.by_anchor.loo.control_kmatched` (`all`, `R`, `Gc`, \\
`A`: margin, SE, z, n), next to the legacy `teacher_vs_king`. Construction: for \\
each turn and each left-out reference j, the held-out reference and the king are \\
both scored against the mean of the other k−1 references (the legacy control \\
scored the king against all k, which handicapped the teacher by construction); \\
king turns that forfeited or sat on the content floor are excluded (the teacher \\
never forfeits, so the floor only ever entered one side). Pre-fork values on the \\
last 30 verdicts: overall −0.13 sd (z −2.5, all negative), R −0.19 (z −4.2, all \\
negative: kings' thoughts predict the teacher's actions better than the teacher's \\
own held-out thoughts do), typicality −0.03 (z −0.4, mixed), A +0.16 (z +4.3, all \\
positive). Telemetry only; it is the rollback signal for this fork (a sign flip \\
against those values).

**What you see.** `duel_params.sd_meter.forfeit_sd = -6`, `ref_min_content = 10`, \\
`typ_min_refs = 2`; verdict SE about 10 % smaller for the same slice; \\
`control_kmatched` on every verdict.
"""
    a.builder.write_text(s.replace(old, new)); print("patched", a.builder); return 0
if __name__ == "__main__":
    raise SystemExit(main())
