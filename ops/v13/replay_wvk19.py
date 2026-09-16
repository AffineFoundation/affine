"""Evidence for the wvk-19 confirmation slice, from history.jsonl.

  * every crown since the wvk-10 reset (reign >= 1): all were first-slice
    passes -> all would have needed a confirmation slice;
  * where a second slice exists (near-miss pooled verdicts, wvk-15
    confirmation slices), apply the per-duel confirmation rule
    (slice-2 margin > 0 AND pooled margin > max(k_sigma·SE_pooled, δ)) and
    report who would have survived;
  * first-slice passes among ALL scored verdicts (challenger_wins or
    duel_rule_wins), i.e. how often the confirmation slice would run;
  * decision replay of the last 30 verdicts (delegates to ops/v10/replay_wvk16.py).

    python ops/v13/replay_wvk19.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "affine" / "state" / "history.jsonl"
K_SIGMA, DELTA = 2.0, 0.002


def per_duel_pass(slice2: dict, pooled: dict) -> bool | None:
    m2, pm, pse = slice2.get("margin"), pooled.get("margin"), pooled.get("se")
    if not all(isinstance(x, (int, float)) for x in (m2, pm, pse)):
        return None
    return m2 > 0 and pm > max(K_SIGMA * pse, DELTA)


def main() -> int:
    rows = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()]
    crowns = [r for r in rows if r.get("event") in ("crowned", "crown_revoked")
              and int(r.get("reign_number") or 0) >= 1]
    print(f"crowns since the wvk-10 reset: {len(crowns)} (reigns "
          f"{min(int(r['reign_number']) for r in crowns)}–{max(int(r['reign_number']) for r in crowns)})")
    print(f"{'reign':5s} {'challenge':11s} {'via':11s} {'margin':>8s} {'z':>6s}  second slice -> per-duel confirmation")
    with2 = 0
    survive = 0
    for r in crowns:
        v = r.get("verdict") or {}
        via = r.get("via") or v.get("via") or "duel"
        m, z = v.get("margin"), v.get("z")
        nm = v.get("near_miss") or {}
        conf = v.get("confirmation") or {}
        note = "no second slice (would have run one)"
        if len(nm.get("slices") or []) >= 2:
            s2 = nm["slices"][1]
            pooled = nm.get("pooled") or {"margin": m, "se": v.get("se")}
            ok = per_duel_pass(s2, pooled)
            with2 += 1
            survive += int(bool(ok))
            note = f"near-miss slice 2 margin {s2.get('margin'):+.5f}, pooled {pooled.get('margin'):+.5f} -> {'CONFIRMED' if ok else 'FAILED'}"
        elif conf.get("slice"):
            s2, pooled = conf["slice"], conf.get("pooled") or {}
            ok = per_duel_pass(s2, pooled)
            with2 += 1
            survive += int(bool(ok))
            note = f"wvk-15 confirmation slice margin {s2.get('margin'):+.5f}, pooled {pooled.get('margin'):+.5f} (se {pooled.get('se'):.5f}) -> {'CONFIRMED' if ok else 'FAILED'}"
        tag = " (revoked)" if r.get("event") == "crown_revoked" else ""
        print(f"{r.get('reign_number'):>5} {r.get('challenge_id'):11s} {via:11s} {m if m is None else f'{m:+.5f}':>8s} {z if z is None else f'{z:5.2f}':>6s}  {note}{tag}")
    print(f"\ncrowns with a stored second slice: {with2}; would have survived the per-duel confirmation: {survive}")

    scored = [r for r in rows if r.get("event") in ("verdict", "crowned", "crown_revoked")
              and isinstance((r.get("verdict") or {}).get("margin"), (int, float))]
    passes = [r for r in scored if (r["verdict"].get("challenger_wins") or r["verdict"].get("duel_rule_wins"))]
    print(f"scored verdicts: {len(scored)}; first-slice passes (duel rule): {len(passes)} "
          f"({100 * len(passes) / max(1, len(scored)):.1f}%) -> confirmation slices that would have run")
    se = sorted(r["verdict"]["se"] for r in scored[-50:] if isinstance(r["verdict"].get("se"), (int, float)))
    med_se = se[len(se) // 2] if se else float("nan")
    z_delta = DELTA / med_se
    # null challenger: P(first slice clears max(2SE, δ)) then P(slice2 > 0 and pooled > δ | slice1 ≈ δ)
    from statistics import NormalDist
    nd = NormalDist()
    p1 = 1 - nd.cdf(max(K_SIGMA, z_delta))
    # pooled over equal n: (m1+m2)/2 > δ with m1 ≈ δ -> m2 > δ -> P(z2 > z_delta); also m2 > 0 implied
    p2 = 1 - nd.cdf(z_delta)
    print(f"median SE (last 50): {med_se:.5f}; δ = {z_delta:.2f}·SE")
    print(f"null-challenger false crown: per-duel rule ≈ {100 * p1:.2f}% per attempt; with confirmation ≈ {100 * p1 * p2:.4f}% "
          f"(2σ-only framing: 2.3% -> ≈ {100 * 0.023 * (1 - nd.cdf(2.0)) * 2:.2f}%)")
    print("\n== decision replay (last 30) ==")
    out = subprocess.run([sys.executable, str(REPO / "ops/v10/replay_wvk16.py"), "--last", "30"],
                         capture_output=True, text=True)
    print("\n".join(out.stdout.splitlines()[-2:]))
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
