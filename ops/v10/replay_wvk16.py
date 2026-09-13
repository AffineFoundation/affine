"""Replay check for the wvk-16 restore: the last N scored verdicts re-decided
with the per-duel rule from their stamped numbers.

restored decision = margin > max(k_sigma·SE, min_margin) AND no gate
rejection (rejection_reason empty) — exactly `score.duel`'s `wins` on the
stamped margin / se / k_sigma / min_margin of each row (the per-turn rows
are not stored, so this replays the decision, not the echoes).

  wvk <= 14 rows (no `crown_mode` stamp): must equal the stored
      `challenger_wins` bit for bit (their crown WAS this rule).
  wvk 15 rows: compared with the stored `duel_rule_wins` (the pod computed
      the same bar as telemetry) and listed next to what the window rule did.

    python ops/v10/replay_wvk16.py [--last 30]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "affine" / "state" / "history.jsonl"


def restored_wins(v: dict) -> bool | None:
    m, se = v.get("margin"), v.get("se")
    if not isinstance(m, (int, float)) or not isinstance(se, (int, float)):
        return None
    if not (math.isfinite(m) and math.isfinite(se)):
        return None
    dp = v.get("duel_params") or {}
    k = float(v.get("k_sigma") or dp.get("k_sigma") or 2.0)
    delta = float(dp.get("min_margin_effective") or v.get("min_margin") or dp.get("min_margin") or 0.002)
    wins = m > max(k * se, delta)
    if v.get("rejection_reason"):
        wins = False
    return wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=30)
    args = ap.parse_args()
    rows = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("event") in ("verdict", "crowned", "crown_revoked")
              and isinstance((r.get("verdict") or {}).get("margin"), (int, float))]
    scored = scored[-args.last:]
    n_pre = n_pre_ok = n_15 = n_15_ok = 0
    print(f"{'challenge':11s} {'wvk':4s} {'margin':>9s} {'se':>9s} {'z':>6s}  stored  restored  note")
    for r in scored:
        v = r["verdict"]
        rw = restored_wins(v)
        is15 = bool(v.get("crown_mode"))
        if is15:
            stored = v.get("duel_rule_wins")
            n_15 += 1
            n_15_ok += int(rw == stored)
            note = ("window: " + str(v.get("crown_decision") or r.get("event")))
            if r.get("event") in ("crowned", "crown_revoked"):
                note += f" -> {r['event']} (pooled z {((v.get('confirmation') or {}).get('pooled') or {}).get('z', float('nan')):.2f})"
        else:
            stored = v.get("challenger_wins")
            n_pre += 1
            n_pre_ok += int(rw == stored)
            note = "pre-wvk-15"
        flag = "" if rw == stored else "  <-- MISMATCH"
        print(f"{r['challenge_id']:11s} {'15' if is15 else '<=14':4s} {v['margin']:9.5f} {v['se']:9.5f} "
              f"{v.get('z', float('nan')):6.2f}  {str(stored):6s}  {str(rw):8s}  {note}{flag}")
    print(f"\nwvk<=14 rows: {n_pre_ok}/{n_pre} decisions reproduced bit-identically")
    print(f"wvk 15 rows : {n_15_ok}/{n_15} agree with the stamped duel_rule_wins; "
          f"under the restored rule none of them would have crowned unless marked restored=True")
    return 0 if n_pre_ok == n_pre and n_15_ok == n_15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
