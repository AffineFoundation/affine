"""Uncrown the sitting king and restore the previous reign as the standing king.

Generalised from ops/incident-r13/revert_reign13.py (reign 13 → 12,
2026-09-13). First use: reign 12 → 11, operator directive 2026-09-14 12:46
UTC ("Remove the last king also, it was a spam model").

Run ONLY while `affine-validator` is stopped (State.load re-reads these
files on start). Backs everything up first. Forward-only: no verdict is
re-decided; the removed king's hotkey stays burned; its public copy stays in
the bucket.

Touches:
  affine/state/state.json
    king              <- king.previous[0] (the restored reign) with its
                         ORIGINAL crowned_at / block / crown_block (the payout
                         window is clocked from crowned_at — a restore must not
                         mint a fresh window) + challenge_id from its crowned row
    king.previous     <- king.previous[1:]  (the removed reign leaves the lineage,
                         so payout.lineage_rows never sees it)
    crown_window      <- removed if present (window rule retired at wvk 16)
    stats.accepted    <- -1
  affine/state/history.jsonl
    crowned row of <cid>  <- event "crown_revoked" (+ revoked_at, revoked_code,
                             revoked_reason, revoked_by, restored_king); every
                             other field kept. Needed: State.load re-crowns the
                             LAST `crowned` row when its reign_number is above
                             the king's.
    + failed row           <cid>, error_code <code>
  affine/state/registrations.json
    records[<reg of cid>].state <- "rejected"; detail <- code + reason
  kingctl: nothing edited; it reads state.json every 60 s and rents /
    publishes the restored king's box, removing the removed king's box once
    the new one serves (the removed box keeps serving datagen until then).

    python ops/uncrown/uncrown_king.py --cid chal-00454 --restore-cid chal-00409 \
        --code revoked_operator_spam_model \
        --reason "operator decision: spam model" \
        --directive "operator directive 2026-09-14 12:46 UTC" --check
    ... --apply
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
BACKUP_ROOT = REPO / "ops" / "uncrown" / "backups"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows_of(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def plan(state: dict, rows: list[str], regs: dict, cid: str, restore_cid: str) -> dict:
    king = state.get("king") or {}
    if king.get("challenge_id") != cid:
        raise SystemExit(f"state.json king is {king.get('challenge_id')} — not {cid}; nothing to do")
    prev = list(king.get("previous") or [])
    if not prev:
        raise SystemExit("king.previous is empty; refusing")
    parsed = [(i, json.loads(l)) for i, l in enumerate(rows) if l.strip()]
    crowned = [(i, r) for i, r in parsed if r.get("event") == "crowned"]
    mine = [i for i, r in crowned if r.get("challenge_id") == cid]
    if len(mine) != 1:
        raise SystemExit(f"expected exactly one crowned row for {cid}, found {len(mine)}")
    if crowned[-1][0] != mine[0]:
        raise SystemExit("a later crowned row exists; refusing (reign moved on)")
    restore_rows = [r for _, r in crowned if r.get("challenge_id") == restore_cid]
    if len(restore_rows) != 1:
        raise SystemExit(f"expected exactly one crowned row for {restore_cid}, found {len(restore_rows)}")
    rr = restore_rows[0]
    r_prev = prev[0]
    if r_prev.get("revision") != rr.get("revision") or r_prev.get("hotkey") != rr.get("hotkey"):
        raise SystemExit(f"king.previous[0] ({str(r_prev.get('revision'))[:12]}) is not {restore_cid}'s "
                         f"model ({str(rr.get('revision'))[:12]}); refusing")
    reg = next((x for x in regs["records"].values()
                if x.get("hotkey") == king.get("hotkey") and x.get("model_digest") == king.get("revision")), None)
    if reg is None:
        raise SystemExit("registration record for the removed king not found")
    new_king = {
        "hotkey": r_prev["hotkey"], "repo": r_prev["repo"], "revision": r_prev["revision"],
        "block": int(r_prev.get("block") or rr.get("block") or 0), "challenge_id": restore_cid,
        "reign_number": int(r_prev["reign_number"]), "crowned_at": r_prev["crowned_at"],
        "score": r_prev.get("score"), "previous": prev[1:],
        "crown_block": r_prev.get("crown_block"), "min_margin_peak": r_prev.get("min_margin_peak"),
    }
    return {"crowned_idx": mine[0], "new_king": new_king, "reg": reg, "dead": king}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cid", required=True, help="challenge id of the sitting king to remove")
    ap.add_argument("--restore-cid", required=True, help="challenge id of the reign to restore (king.previous[0])")
    ap.add_argument("--code", required=True, help="error_code / revoked_code, e.g. revoked_operator_spam_model")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--directive", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    state = json.loads(STATE.read_text())
    rows = rows_of(HISTORY)
    regs = json.loads(REGS.read_text())
    p = plan(state, rows, regs, a.cid, a.restore_cid)
    nk, dead = p["new_king"], p["dead"]
    print("plan:")
    print(f"  king      : {dead['challenge_id']} {dead['revision'][:12]} reign {dead['reign_number']} "
          f"-> {nk['challenge_id']} {nk['revision'][:12]} reign {nk['reign_number']} hotkey {nk['hotkey'][:12]} "
          f"crowned_at {nk['crowned_at']} (original)")
    print(f"  previous  : {len(dead.get('previous') or [])} -> {len(nk['previous'])}")
    print(f"  crowned row index {p['crowned_idx']} -> event crown_revoked ({a.code})")
    print(f"  + failed row {a.code} for {a.cid}")
    print(f"  crown_window: {(state.get('crown_window') or {}).get('window_id')} -> removed")
    print(f"  registration {p['reg']['registration_id'][:12]} {p['reg']['state']} -> rejected")
    print(f"  stats.accepted {state['stats'].get('accepted')} -> {max(0, int(state['stats'].get('accepted', 1)) - 1)}")
    if a.check:
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bdir = BACKUP_ROOT / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    for f in (STATE, HISTORY, REGS):
        shutil.copy2(f, bdir / f.name)
    print("backups ->", bdir)

    row = json.loads(rows[p["crowned_idx"]])
    row["event"] = "crown_revoked"
    row["revoked_at"] = now_iso()
    row["revoked_code"] = a.code
    row["revoked_reason"] = a.reason
    row["revoked_by"] = a.directive
    row["restored_king"] = {"challenge_id": nk["challenge_id"], "revision": nk["revision"],
                            "hotkey": nk["hotkey"], "reign_number": nk["reign_number"]}
    rows[p["crowned_idx"]] = json.dumps(row) + "\n"
    rows.append(json.dumps({
        "event": "failed", "at": now_iso(), "challenge_id": a.cid, "hotkey": dead["hotkey"],
        "repo": row.get("repo"), "revision": dead["revision"],
        "error_code": a.code, "error_detail": (a.reason + "; " + a.directive)[:2000],
        "uid": row.get("uid"),
    }) + "\n")
    tmp = HISTORY.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(rows))
    tmp.replace(HISTORY)

    state["king"] = nk
    state.pop("crown_window", None)
    state["stats"]["accepted"] = max(0, int(state["stats"].get("accepted", 1)) - 1)
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)

    reg = p["reg"]
    reg["state"] = "rejected"
    reg["detail"] = f"{a.code}: crown of {a.cid} revoked — {a.reason}; {a.directive}"[:500]
    reg["updated_at"] = now_iso()
    tmp = REGS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(regs, indent=1))
    tmp.replace(REGS)

    s2 = json.loads(STATE.read_text())
    k = s2["king"]
    assert k["challenge_id"] == a.restore_cid and k["revision"] == nk["revision"]
    assert k["crowned_at"] == nk["crowned_at"] and "crown_window" not in s2
    crowned = [json.loads(l) for l in HISTORY.read_text().splitlines()
               if l.strip() and json.loads(l).get("event") == "crowned"]
    assert crowned[-1]["challenge_id"] == a.restore_cid, crowned[-1]["challenge_id"]
    print(f"applied: king = {k['challenge_id']} {k['revision'][:12]} reign {k['reign_number']} "
          f"crowned_at {k['crowned_at']} | last crowned row = {crowned[-1]['challenge_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
