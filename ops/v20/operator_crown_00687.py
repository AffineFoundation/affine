"""Operator crown of chal-00687 (reign 22) from its stored wvk-24 verdict.

Explicit dated operator directive — Jacob Steeves, 2026-09-26 18:51 UTC: "Tell the
validator to crown the last miner model who scored the best against the king and
set weights to it immediately while we consider the cut over."

Pick (every challenger verdict since reign 21's crown chal-00662, 2026-09-22
13:03): chal-00687, uid 62, hotkey 5EzaX8pVDyqCFE3Gcw8r2SAb8SKy4ZZydLQeTFDjbFBzPyqT,
digest 7f066f2c5f95…, paired margin +0.0728 sd, SE 0.0267, z +2.73 (cleared the
2·SE bar 0.053, did not clear δ = 0.20 sd under wvk 24), forfeits 0.3 % / king
0.2 %, protocol probe 0.90 pass, hygiene + arch pin passed at dispatch, not a
byte/ε-copy of reigns 14–21 (0/18 shard hashes shared; tensor sample vs reign 21:
median 30 % of elements changed, ‖Δ‖/‖king‖ ≈ 1e-3), different coldkey from the
grpo lineage. Runner-ups chal-00677 (+0.054, z 1.22), chal-00682 (+0.047, z 1.32).

Same path as ops/v16/retro_crown_00556.py (reign 14, wvk 21). Run with the
validator STOPPED at a duel boundary and the validator env sourced (R2 keys):
  * AccessController.promote → public copy models/sha256/<digest>/ (falls back to
    the private ref; the validator re-promotes on its sweep);
  * State.record_verdict with the stored verdict flipped to challenger_wins,
    via = "operator_crown", operator_crown = {directive, date, note} → one `crowned`
    row, king = reign 22, crowned_at = now (payout window starts now);
  * bench card enqueued (label reign-22); kingctl follows state.json.
The original `verdict` row stays untouched. No scoring change, no wvk bump.

    python ops/v20/operator_crown_00687.py --check
    python ops/v20/operator_crown_00687.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.state import QueueEntry, State  # noqa: E402

CID = "chal-00687"
UID = 62
HOTKEY = "5EzaX8pVDyqCFE3Gcw8r2SAb8SKy4ZZydLQeTFDjbFBzPyqT"
DIGEST = "7f066f2c5f95b34105c23faa313e25908c8610fbb65091e797e8b9391e926dfc"
EXPECT_KING_CID = "chal-00662"
DIRECTIVE_DATE = "2026-09-26"
DIRECTIVE = ("explicit dated operator directive — Jacob Steeves, 2026-09-26 18:51 UTC: "
             "\"Tell the validator to crown the last miner model who scored the best against "
             "the king and set weights to it immediately while we consider the cut over.\"")
NOTE = "operator crown 2026-09-26; did not clear δ under wvk 24"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    state = State(cfg.state_dir)
    state.load()
    if state.in_flight is not None:
        raise SystemExit(f"in_flight is {state.in_flight.challenge_id}; stop the validator at a boundary first")
    if state.king is None or state.king.challenge_id != EXPECT_KING_CID:
        raise SystemExit(f"king is not {EXPECT_KING_CID} (reign 21): {state.king and state.king.challenge_id}")
    rows = [json.loads(l) for l in (cfg.state_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    vrows = [r for r in rows if r.get("challenge_id") == CID and r.get("event") == "verdict"]
    if len(vrows) != 1:
        raise SystemExit(f"expected one verdict row for {CID}, found {len(vrows)}")
    if any(r.get("event") == "crowned" and r.get("challenge_id") == CID for r in rows):
        raise SystemExit(f"{CID} already has a crowned row")
    row = vrows[0]
    if row.get("hotkey") != HOTKEY or row.get("revision") != DIGEST:
        raise SystemExit("verdict row hotkey/revision differ from the pick")
    v = dict(row["verdict"])
    if v.get("challenger_wins"):
        raise SystemExit("verdict already has challenger_wins = true")
    k, d = float(v["k_sigma"]), float(v["min_margin"])
    margin, se = float(v["margin"]), float(v["se"])
    if not (margin > k * se):
        raise SystemExit(f"margin {margin} does not clear k*SE {k * se}")
    if not (margin < d):
        raise SystemExit(f"margin {margin} clears δ {d} — this would have been a rule crown; check the row")
    if not (v.get("protocol_probe") or {}).get("passed"):
        raise SystemExit("protocol probe did not pass")
    regs = json.loads((cfg.state_dir / "registrations.json").read_text())["records"]
    rec = next(r for r in regs.values() if r.get("challenge_id") == CID)
    if rec.get("model_digest") != DIGEST:
        raise SystemExit("registration digest differs from the pick")
    entry = QueueEntry(challenge_id=CID, hotkey=HOTKEY, repo=row["repo"], revision=row["revision"],
                       block=int(rec.get("ready_block") or 0), queued_at=rec.get("created_at") or "")
    king = state.king
    print("plan:")
    print(f"  king now : {king.challenge_id} reign {king.reign_number} {king.revision[:12]} {king.hotkey[:12]}")
    print(f"  crown    : {CID} uid {UID} {entry.revision[:12]} {HOTKEY[:12]} block {entry.block} -> reign {king.reign_number + 1}")
    print(f"  verdict  : margin {margin:+.4f} sd, se {se:.4f}, z {float(v['z']):+.2f}; k*SE {k * se:.4f} cleared, δ {d} not cleared")
    print(f"  probe    : pass_rate {v['protocol_probe'].get('pass_rate')}; forfeits {v.get('n_forfeit_turns')} / {v.get('n_paired_turns')} paired")
    print(f"  repo     : {entry.repo} -> public copy models/sha256/{entry.revision}/")
    print(f"  stamp    : via = operator_crown; note = {NOTE!r}")
    if a.check:
        return 0

    from affine.registrations import AccessController  # noqa: E402  (needs the validator env)
    ac = AccessController.build_if_configured(cfg, state, lambda info: None)
    public_ref = None
    if ac is not None:
        try:
            public_ref = ac.promote(entry)
            print("promoted ->", public_ref)
        except Exception as e:  # noqa: BLE001 — same fallback as the validator
            print("PROMOTION FAILED, crowning the private ref (validator re-promotes on sweep):", e)
    crowned_entry = replace(entry, repo=public_ref) if public_ref else entry
    v["challenger_wins"] = True
    v["rejection_reason"] = None
    v["via"] = "operator_crown"
    v["operator_crown"] = {"directive": DIRECTIVE, "directive_date": DIRECTIVE_DATE, "note": NOTE,
                           "original_verdict_at": row["at"],
                           "original_outcome": "margin > 2*SE but below min_margin_sd 0.2 (wvk 24); not crowned",
                           "rule_applied": "operator crown: best paired margin vs reign 21 since chal-00662, "
                                           "probe + hygiene passed, not a byte/epsilon-copy",
                           "runner_ups": ["chal-00677 +0.054 z 1.22", "chal-00682 +0.047 z 1.32"],
                           "scoring_change": False, "weight_version_key": int(cfg.weight_version_key)}
    if public_ref:
        v["private_repo"] = entry.repo
    try:
        import bittensor as bt  # noqa: E402
        crown_block = int(bt.Subtensor("finney").block)
    except Exception as e:  # noqa: BLE001
        print("chain block unavailable:", e)
        crown_block = None
    new_king = state.record_verdict(crowned_entry, v, uid=UID, crown_block=crown_block,
                                    min_margin_peak=float(cfg.duel.min_margin))
    state.enqueue_bench(crowned_entry.repo, entry.revision, HOTKEY, list(cfg.bench.suites),
                        f"reign-{new_king.reign_number}")
    state.flush()
    if ac is not None:
        ac.flush()
    print(f"crowned: {new_king.challenge_id} reign {new_king.reign_number} {new_king.revision[:12]} "
          f"crowned_at {new_king.crowned_at} crown_block {crown_block} repo {new_king.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
