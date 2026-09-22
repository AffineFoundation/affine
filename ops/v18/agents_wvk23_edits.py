"""AGENTS.md snapshot for wvk 23 (run after the flip). Idempotent."""
from __future__ import annotations
import argparse
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    for k in ("--flip-time", "--first-verdict", "--first-secs", "--first-forfeits", "--first-lenz", "--first-ctrl"):
        ap.add_argument(k, required=True)
    ap.add_argument("--agents", type=Path, default=REPO / "AGENTS.md")
    a = ap.parse_args(); s = a.agents.read_text()
    if "weight_version_key = 23" in s: print("already"); return 0
    def sub(old, new):
        nonlocal s
        if s.count(old) != 1: raise SystemExit(f"anchor: {old[:60]!r}")
        s = s.replace(old, new)
    sub("- `weight_version_key = 22` (2026-09-18 20:41 UTC, explicit dated operator\n",
        f"""- `weight_version_key = 23` ({a.flip_time}, explicit dated operator directive
  Jacob Steeves 2026-09-22 17:00 UTC "all of them and also 3 flipped", after
  the benchsuite thought-shrink audit `internal/benchsuite/thought-shrink-
  mechanism-2026-09-22.md`): `[duel].max_thought_tokens` 2048 → **4096**,
  `ref_max_tokens` 4096 → **4864** (= 4096 + 768; the validator requires refs
  ≥ thought + action; 8192 not taken — the k = 3 reference samples per turn
  set the wall time, KV is not the constraint), and
  `[duel.sd_meter].content_prefix = "refs_max"`: typ_c is computed on the
  first K content tokens of the miner's thought, K = the teacher's longest
  reference in content tokens — extra deliberation unscored, the two-sided
  band on the scored prefix keeps RT-11/RT-12 (filler below, pasting above).
  Chosen over dropping the above side (38 % of miner turns sit above μ_c
  today — mode-hugging, not deliberation; would reopen the pasting hole).
  Probe (`ops/v18/probe_a.txt`, `probe_b.txt`; project store
  `internal/wvk23/g-one-sided-probe.md`): last 30 wvk-22 verdicts replayed
  under (i) one-sided — 0/30 decisions change; 173-turn re-echo under (ii)
  refs_max — truncation touches 28 % of miner thoughts by +0.03…+0.07 sd,
  typicality-leg control +0.36 → +0.27 sd (z 3.8 → 3.0). **Standing finding
  from the probe:** the overall teacher-vs-king control has been NEGATIVE on
  every one of the last 30 wvk-22 verdicts (−0.3…−0.7 sd, z −3.4…−8.6) — the
  kings beat the teacher's held-out replies on the R and A legs (typ leg
  still +0.36 for the teacher); the wvk-22 §3 trigger "control z ≤ −2" has
  therefore been true since ~09-20 and is read as the meter's asymptote, not
  a fault (Jacob informed 2026-09-22). Rollback for wvk 23 = a control SIGN
  FLIP relative to the pre-fork value on the first verdicts
  (`ops/v18/rollback_wvk23.sh`). Rows carry `mc_za_full` / `n_content_za_full`
  / `k_ref_content` so the wvk-22 rule replays on wvk-23 rows. Cost ~3×
  accepted. Forward-only, reign 21 stands. First wvk-23 verdict
  `{a.first_verdict}`: {a.first_secs} s, forfeits {a.first_forfeits}, median
  thought chars {a.first_lenz}, control {a.first_ctrl}.
  22 = 2026-09-18 20:41 UTC, explicit dated operator
""")
    sub("## 2. Frozen production scoring — sd-meter min(z_R, typ_c, z_A) since wvk 22 (2026-09-18); min(R,G) v5 below is the wvk 10–21 rule\n",
        "## 2. Frozen production scoring — sd-meter min(z_R, typ_c, z_A) since wvk 22 (2026-09-18; wvk 23 2026-09-22: thought cap 4096, typicality on the first K content tokens); min(R,G) v5 below is the wvk 10–21 rule\n")
    sub("`affine/evalsrv/sdmeter.py`). R below is still the R leg; G (the band) is\n",
        "`affine/evalsrv/sdmeter.py`; since wvk 23 typ_c reads only the first K\ncontent tokens of the miner's thought, K = the longest reference, and the\nthought cap is 4096 / refs 4864). R below is still the R leg; G (the band) is\n")
    a.agents.write_text(s); print("patched", a.agents); return 0

if __name__ == "__main__":
    raise SystemExit(main())
