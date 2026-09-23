"""AGENTS.md snapshot for wvk 23 (run after the flip). Idempotent."""
from __future__ import annotations
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    for k in ("--flip-time", "--first-verdict", "--first-secs", "--first-forfeits", "--first-se", "--first-ctrl"):
        ap.add_argument(k, required=True)
    ap.add_argument("--agents", type=Path, default=REPO / "AGENTS.md")
    a = ap.parse_args(); s = a.agents.read_text()
    if "weight_version_key = 24" in s: print("already"); return 0
    def sub(old, new):
        nonlocal s
        if s.count(old) != 1: raise SystemExit(f"anchor: {old[:60]!r}")
        s = s.replace(old, new)
    sub("- `weight_version_key = 23` (", f"""- `weight_version_key = 24` ({a.flip_time}, explicit dated operator directive
  Jacob Steeves 2026-09-23 20:17 UTC "Lets do this", on the training-speed
  probe `internal/wvk23/training-speed-probe-2026-09-23.md`): `[duel.sd_meter].
  forfeit_sd` −12 → **−6** (also the typ_c floor for < 10 content tokens).
  Why: 2.0 % of side-turns at −12 carried 48 % of the per-turn score variance.
  −6 stays strictly below the genuine valid-turn p1 (−4.64; p0.5 −5.48); 0.32 %
  of valid turns score below it, oracle forfeit-seeking gain 0.007 sd/turn
  (3.5 % of δ). Counterfactual on the last 30 verdicts (`ops/v19/
  floor_counterfactual.md`): 0 flips, SE ×0.89 median (×0.78 best), z shifts
  within ±0.5 (one +0.99); a 2 % forfeit gap now costs ≈ 0.09 sd (half a δ).
  **Two-step flip at consecutive boundaries (same wvk):** floor live 20:45 UTC
  (`chal-00678` judged with the floor only); addendum (directive 20:47 "do
  it") from the next boundary, `chal-00679` on: (6) `ref_min_content = 10`,
  `typ_min_refs = 2` — a reference thought with < 10 content tokens does not
  anchor typicality, < 2 content-bearing refs → min(z_R, z_A) (7.5 % of refs,
  ~4.8 % of turns; combined counterfactual 0 flips / 30, `ops/v19/
  combined_counterfactual.md`); (3) `control_kmatched` on every verdict —
  teacher-vs-king with the king scored against the same k−1 refs as the
  held-out reference, king forfeits / content-floor turns dropped, overall +
  per leg; pre-fork: all −0.13 (z −2.5, 30/30 neg), R −0.19 (z −4.2, 30/30
  neg), Gc −0.03 (mixed), A +0.16 (z +4.3, 30/30 pos) — the rollback signal
  is a sign flip vs these. Nothing else changed; forward-only, reign 21
  stands. First full-bundle verdict
  `{a.first_verdict}`: {a.first_secs} s, forfeits {a.first_forfeits}, SE {a.first_se},
  control {a.first_ctrl}. Rollback = control z sign flip
  (`ops/v19/rollback_wvk24.sh`).
- `weight_version_key = 23` (""")
    sub("wvk 23 2026-09-22: thought cap 4096, typicality on the first K content tokens); min(R,G) v5 below is the wvk 10–21 rule\n",
        "wvk 23 2026-09-22: thought cap 4096, typicality on the first K content tokens; wvk 24 2026-09-23: forfeit floor −6 sd); min(R,G) v5 below is the wvk 10–21 rule\n")
    a.agents.write_text(s); print("patched", a.agents); return 0

if __name__ == "__main__":
    raise SystemExit(main())
