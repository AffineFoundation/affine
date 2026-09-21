"""Retroactive crown of chal-00556 from its stored slice-1 verdict (wvk 21).

Explicit dated operator directive 2026-09-17 10:07 UTC: "Feel free to crown
the last model which passed but failed the crown." chal-00556 (uid 175,
hotkey 5CX396QeJuLZ3WKYfueQRRXsxRyvdwhPP3Q4d73brxk2bYCq, revision
0f4029fd…) cleared the per-duel bar on slice 1 (margin +0.0022, z 3.13) and
was rejected only by the wvk-19 confirmation slice (pooled +0.0014 < δ).

Run with the validator STOPPED and the validator env sourced (R2 / Cloudflare
keys for the public copy). Uses the normal crown path — no re-duel:
  * promotes the private prefix to the public bucket (AccessController.promote,
    registration record -> crowned / public_ref) — falls back to the private
    ref if the copy fails (the validator re-promotes on its sweep);
  * State.record_verdict with the stored verdict flipped to challenger_wins
    (+ via / retroactive stamps, the confirmation block kept for audit)
    -> one `crowned` row, king = reign 14, crowned_at = now (payout window
    starts now), king.previous += reign 13;
  * bench card enqueued as for any crown (label reign-14); kingctl follows
    state.json on its own (king seat).
The original `verdict` row (confirmation_failed) stays in history untouched.

    python ops/v16/retro_crown_00556.py --check
    python ops/v16/retro_crown_00556.py --apply
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

CID = "chal-00556"
UID = 175
HOTKEY = "5CX396QeJuLZ3WKYfueQRRXsxRyvdwhPP3Q4d73brxk2bYCq"
DIRECTIVE = ("operator directive 2026-09-17 10:07 UTC (Jacob Steeves): remove the double eval "
             "on kings; crown the last model which passed but failed the crown")


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
    rows = [json.loads(l) for l in (cfg.state_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    vrows = [r for r in rows if r.get("challenge_id") == CID and r.get("event") == "verdict"]
    if len(vrows) != 1:
        raise SystemExit(f"expected one verdict row for {CID}, found {len(vrows)}")
    if any(r.get("event") == "crowned" and r.get("challenge_id") == CID for r in rows):
        raise SystemExit(f"{CID} already has a crowned row")
    row = vrows[0]
    v = dict(row["verdict"])
    if v.get("rejection_reason") != "confirmation_failed" or row.get("hotkey") != HOTKEY:
        raise SystemExit("verdict row is not the confirmation-only rejection expected")
    k, d = float(v["k_sigma"]), float(v["duel_params"]["min_margin"])
    bar = max(k * float(v["se"]), d)
    if not float(v["margin"]) > bar:
        raise SystemExit(f"slice-1 margin {v['margin']} does not clear the bar {bar}")
    regs = json.loads((cfg.state_dir / "registrations.json").read_text())["records"]
    rec = next(r for r in regs.values() if r.get("challenge_id") == CID)
    entry = QueueEntry(challenge_id=CID, hotkey=HOTKEY, repo=row["repo"], revision=row["revision"],
                       block=int(rec.get("ready_block") or 0), queued_at=rec.get("created_at") or "")
    king = state.king
    print("plan:")
    print(f"  king now : {king.challenge_id} reign {king.reign_number} {king.revision[:12]}")
    print(f"  crown    : {CID} uid {UID} {entry.revision[:12]} block {entry.block} -> reign {king.reign_number + 1}")
    print(f"  slice 1  : margin {float(v['margin']):+.5f} se {float(v['se']):.5f} z {float(v['z']):.2f} > bar {bar:.5f}; "
          f"confirmation was pooled {float(v['confirmation']['pooled_margin']):+.5f} (kept for audit)")
    print(f"  repo     : {entry.repo} -> public copy models/sha256/{entry.revision}/")
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
    v["via"] = "retroactive_wvk21"
    v["retroactive"] = {"directive": DIRECTIVE, "original_verdict_at": row["at"],
                        "original_rejection": "confirmation_failed",
                        "rule_applied": "margin > max(k_sigma*SE, min_margin) on slice 1 (wvk 21)"}
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
