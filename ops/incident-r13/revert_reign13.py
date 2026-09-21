"""Uncrown reign 13 (chal-00461, a re-upload of reign 12's weights) and put
reign 12 (king-d76150805915, hotkey 5DJ91T4Z…, uid 156) back on the throne.

Operator directive 2026-09-13 12:10 UTC ("uncrown the copied model").
Run ONLY while `affine-validator` is stopped (State.load re-reads these
files on start). Backs everything up first. Forward-only: no other verdict
is touched; the copier's hotkey stays burned; the reign-13 public copy is
left in the bucket (identical bytes to reign 12's public copy).

Touches:
  affine/state/state.json
    king                      <- king.previous[0] (reign 12) + challenge_id
                                 chal-00454 (from its crowned row)
    king.previous             <- king.previous[1:]  (reign 13 not in lineage)
    crown_window              <- null (window 2516 is not closed under the
                                 window rule; recorded as a window_close row
                                 with outcome king_stays_fork_wvk16)
    stats.accepted            <- -1
    (seen_hotkeys, completed_revisions, in_flight, queue: unchanged)
  affine/state/history.jsonl
    crowned row of chal-00461 <- event "crown_revoked" (+ revoked_at,
                                 revoked_reason, revoked_by); every other
                                 field kept. Needed: State.load re-crowns the
                                 LAST `crowned` row when its reign_number is
                                 above the king's.
    + failed row              chal-00461, error_code rejected_model_copy
    + window_close row        window 2516, outcome king_stays_fork_wvk16
  affine/state/registrations.json
    records[381ce843…].state  <- "rejected"; detail <- model_copy note
  ops/king-datagen: nothing edited. kingctl reads state.json every 60 s:
    the reign-13 box (never published) is removed on its next tick; the
    reign-12 box is published and serving and stays; if it were gone,
    kingctl re-rents ("no box for this king").

    python ops/incident-r13/revert_reign13.py --check     # dry run, prints the plan
    python ops/incident-r13/revert_reign13.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "affine" / "state"
STATE = STATE_DIR / "state.json"
HISTORY = STATE_DIR / "history.jsonl"
REGS = STATE_DIR / "registrations.json"
BACKUP_ROOT = REPO / "ops" / "incident-r13" / "backups"

COPY_CID = "chal-00461"
COPY_REV = "ed3111d3b0b95c01c9646b7e4ba8fe3473f8d742662015923959fc2bc18ae837"
COPY_HOTKEY = "5EkegWphCBzBt8paLwKqp3PcEZggNZwHMeTsqoUKmSt9V7UP"
COPY_REG = "381ce84341497daa1d13131ab0a8b77a99448e6f9c3df0f5500819e2644c7ef8"
KING_CID = "chal-00454"
KING_REV = "d76150805915a988723d16aed4c45dc8e61bedeb08fa8599289285792f3f7795"
KING_HOTKEY = "5DJ91T4ZUQk2BCgcRXmapG14pwxeGUKnGocWVBZ7FaxsgGCn"
DIRECTIVE = "operator directive 2026-09-13 12:10 UTC (uncrown the copied model)"
REASON = ("model_copy: reign 13 was reign 12's weights re-uploaded — 1,026/1,026 "
          "tensors byte-identical (70,214,363,872 bytes), re-sharded 16→2 files; "
          "crowned under the wvk-15 window rule on z = 0.93 / pooled +0.000045")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rows() -> list[str]:
    return HISTORY.read_text().splitlines(keepends=True)


def plan(state: dict, rows: list[str], regs: dict) -> dict:
    king = state.get("king") or {}
    if king.get("challenge_id") != COPY_CID or king.get("revision") != COPY_REV:
        raise SystemExit(f"state.json king is {king.get('challenge_id')} @ "
                         f"{str(king.get('revision'))[:12]} — not reign 13; nothing to do")
    prev = list(king.get("previous") or [])
    if not prev or prev[0].get("revision") != KING_REV or prev[0].get("hotkey") != KING_HOTKEY:
        raise SystemExit("king.previous[0] is not reign 12; refusing")
    crowned_idx = [i for i, l in enumerate(rows)
                   if l.strip() and json.loads(l).get("event") == "crowned"
                   and json.loads(l).get("challenge_id") == COPY_CID]
    if len(crowned_idx) != 1:
        raise SystemExit(f"expected exactly one crowned row for {COPY_CID}, found {len(crowned_idx)}")
    last_crowned = max(i for i, l in enumerate(rows)
                       if l.strip() and json.loads(l).get("event") == "crowned")
    if last_crowned != crowned_idx[0]:
        raise SystemExit("a later crowned row exists; refusing (reign moved on)")
    rec = regs["records"].get(COPY_REG)
    if rec is None or rec.get("hotkey") != COPY_HOTKEY:
        raise SystemExit("registration record for the copy not found")
    r12 = prev[0]
    new_king = {
        "hotkey": r12["hotkey"], "repo": r12["repo"], "revision": r12["revision"],
        "block": int(r12.get("block") or 0), "challenge_id": KING_CID,
        "reign_number": int(r12["reign_number"]), "crowned_at": r12["crowned_at"],
        "score": r12.get("score"), "previous": prev[1:],
        "crown_block": r12.get("crown_block"), "min_margin_peak": r12.get("min_margin_peak"),
    }
    return {"crowned_idx": crowned_idx[0], "new_king": new_king, "reg": rec}


def window_close_row(state: dict, new_king: dict) -> dict | None:
    cw = state.get("crown_window")
    if not cw:
        return None
    verdicts = list(cw.get("verdicts") or [])
    cands = sorted((v for v in verdicts if isinstance(v.get("margin"), (int, float))
                    and v["margin"] > 0 and not v.get("rejection_reason")),
                   key=lambda v: -float(v["margin"]))
    W = int(cw.get("window_blocks") or 3600)
    wid = int(cw["window_id"])
    return {
        "event": "window_close", "at": now_iso(), "crown_mode": "window_best",
        "window_id": wid, "window_blocks": W, "window_blocks_range": [wid * W, (wid + 1) * W - 1],
        "decision_block": None,
        "king": {"challenge_id": new_king["challenge_id"], "reign_number": new_king["reign_number"],
                 "hotkey": new_king["hotkey"], "revision": new_king["revision"],
                 "note": "window king was reign 13 = the same weights; uncrowned by " + DIRECTIVE},
        "verdicts_considered": [{k: v.get(k) for k in ("challenge_id", "hotkey", "margin", "se", "z",
                                                        "rejection_reason", "duel_rule_wins",
                                                        "decision_block", "at")} for v in verdicts],
        "candidates": [{k: v.get(k) for k in ("challenge_id", "hotkey", "margin", "se", "z",
                                               "n_paired_turns", "n_slices")} for v in cands],
        "dropped": [], "close_attempts": 0, "confirmations": [], "winner": None,
        "outcome": "king_stays_fork_wvk16", "crown_block": None,
        "note": "window rule retired (weight_version_key 15→16); no candidate was confirmed or crowned",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    state = json.loads(STATE.read_text())
    rows = load_rows()
    regs = json.loads(REGS.read_text())
    p = plan(state, rows, regs)
    wc = window_close_row(state, p["new_king"])
    print("plan:")
    print("  king      :", state["king"]["challenge_id"], state["king"]["revision"][:12],
          "reign", state["king"]["reign_number"], "->", KING_CID, KING_REV[:12], "reign",
          p["new_king"]["reign_number"], "hotkey", KING_HOTKEY[:12])
    print("  previous  :", len(state["king"]["previous"]), "->", len(p["new_king"]["previous"]))
    print("  crowned row index", p["crowned_idx"], "-> event crown_revoked")
    print("  + failed row rejected_model_copy for", COPY_CID)
    print("  crown_window:", (state.get("crown_window") or {}).get("window_id"), "-> null",
          f"(+ window_close row, {len(wc['verdicts_considered'])} verdicts)" if wc else "(none open)")
    print("  registration", COPY_REG[:12], p["reg"]["state"], "-> rejected")
    print("  stats.accepted", state["stats"].get("accepted"), "->", state["stats"].get("accepted", 1) - 1)
    if args.check:
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bdir = BACKUP_ROOT / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    for f in (STATE, HISTORY, REGS):
        shutil.copy2(f, bdir / f.name)
    print("backups ->", bdir)

    # history: rewrite the crowned row in place, append failed + window_close
    row = json.loads(rows[p["crowned_idx"]])
    row["event"] = "crown_revoked"
    row["revoked_at"] = now_iso()
    row["revoked_reason"] = REASON
    row["revoked_by"] = DIRECTIVE
    row["restored_king"] = {"challenge_id": KING_CID, "revision": KING_REV, "hotkey": KING_HOTKEY,
                            "reign_number": p["new_king"]["reign_number"]}
    rows[p["crowned_idx"]] = json.dumps(row) + "\n"
    failed = {
        "event": "failed", "at": now_iso(), "challenge_id": COPY_CID, "hotkey": COPY_HOTKEY,
        "repo": row.get("repo"), "revision": COPY_REV,
        "error_code": "rejected_model_copy", "error_detail": REASON + "; " + DIRECTIVE,
        "uid": row.get("uid"),
    }
    rows.append(json.dumps(failed) + "\n")
    if wc:
        rows.append(json.dumps(wc) + "\n")
    tmp = HISTORY.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(rows))
    tmp.replace(HISTORY)

    # state.json
    state["king"] = p["new_king"]
    state["crown_window"] = None
    state.pop("crown_window", None)
    state["stats"]["accepted"] = max(0, int(state["stats"].get("accepted", 1)) - 1)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)

    # registrations.json
    rec = p["reg"]
    rec["state"] = "rejected"
    rec["detail"] = ("rejected_model_copy: crown of chal-00461 revoked — " + REASON)[:500]
    rec["updated_at"] = now_iso()
    tmp = REGS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(regs, indent=1))
    tmp.replace(REGS)

    # read-back
    s2 = json.loads(STATE.read_text())
    k = s2["king"]
    assert k["challenge_id"] == KING_CID and k["revision"] == KING_REV and k["reign_number"] == 12
    assert "crown_window" not in s2
    crowned = [json.loads(l) for l in HISTORY.read_text().splitlines()
               if l.strip() and json.loads(l).get("event") == "crowned"]
    assert crowned[-1]["challenge_id"] == KING_CID, crowned[-1]["challenge_id"]
    print("applied: king =", k["challenge_id"], k["revision"][:12], "reign", k["reign_number"],
          "| last crowned row =", crowned[-1]["challenge_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
