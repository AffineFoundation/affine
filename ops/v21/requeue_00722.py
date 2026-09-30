"""Re-admit chal-00722 (uid 3) after the protocol-probe gap of 2026-09-28.

Operator directive (Jacob Steeves, 2026-09-28 15:23 UTC): "requeue chal-00722 (uid 3) via the
requeue path; its rejection was our rule's gap, not the miner's fault, so it should be judged.
Stamp the history row with the reason ('probe gap 09-28, re-admitted')."

What happened: the first verdict under the enforced code-fence probe rule (12:25 UTC) rejected
chal-00722 at pass_rate 0.875 on five `no_think_close` replies to the code prompts — zero fence
faults — i.e. on think budget, which the rule was never meant to judge. The neutral rule
(15:18 UTC) fixes that going forward; under it the stored probe scores 35/35.

Mechanics = ops/requeue_infra_failed.py: direct queue inject under the ORIGINAL challenge id
(queue order is canonical by id), State.enqueue() is NOT used (hotkey already in seen_hotkeys,
revision already in completed_revisions). Run with the validator STOPPED at a duel boundary
(the wrapper does that):

    python ops/v21/requeue_00722.py --check
    python ops/v21/requeue_00722.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))
from affine.state import QueueEntry, State  # noqa: E402

STATE_DIR = REPO / "affine" / "state"
CID = "chal-00722"
UID = 3
REASON = "probe gap 09-28, re-admitted"
DETAIL = ("rejected 2026-09-28 13:49 UTC by the code-fence probe at pass_rate 0.875 on 5 no_think_close "
          "replies to the code prompts with zero fence faults (think budget, not the fence habit the rule "
          "targets); neutral rule live 15:18 UTC scores the same probe 35/35. Re-admitted under the original "
          "challenge id on explicit operator directive (Jacob Steeves 2026-09-28 15:23 UTC). Slot not burned twice.")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true"); g.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    state = State(STATE_DIR); state.load()
    if state.in_flight is not None:
        raise SystemExit(f"in_flight is {state.in_flight.challenge_id}; stop the validator at a boundary first")
    rows = [json.loads(l) for l in (STATE_DIR / "history.jsonl").read_text().splitlines() if l.strip()]
    mine = [r for r in rows if r.get("challenge_id") == CID]
    verdicts = [r for r in mine if r.get("event") == "verdict"]
    if len(verdicts) != 1:
        raise SystemExit(f"expected one verdict row for {CID}, found {len(verdicts)}")
    if any(r.get("event") == "requeued" for r in mine):
        raise SystemExit(f"{CID} already has a requeued row")
    row = verdicts[0]; v = row["verdict"]
    rej = str(v.get("rejection_reason") or "")
    if not rej.startswith("protocol:") or "no_think_close" not in rej or row.get("uid") != UID:
        raise SystemExit(f"verdict row is not the probe rejection expected: {rej!r} uid {row.get('uid')}")
    if any(e.challenge_id == CID for e in state.queue):
        raise SystemExit(f"{CID} already on the queue")
    regs = json.loads((STATE_DIR / "registrations.json").read_text())["records"]
    rec = next(r for r in regs.values() if r.get("challenge_id") == CID)
    block = int(rec.get("ready_block") or 0)
    entry = QueueEntry(challenge_id=CID, hotkey=row["hotkey"], repo=row["repo"], revision=row["revision"],
                       block=block, queued_at=datetime.now(timezone.utc).isoformat(), retry_count=0)
    print("plan:")
    print(f"  inject   : {CID} uid {UID} {entry.hotkey[:12]} {entry.revision[:12]} block {block}")
    print(f"  repo     : {entry.repo}")
    print(f"  original : {row['at']} {rej}")
    print(f"  queue now: {[e.challenge_id for e in state.queue]} -> {CID} sorts to the front (canonical id order)")
    print(f"  history  : event=requeued reason={REASON!r}")
    if a.check:
        return 0
    bak = STATE_DIR / f"state.json.bak-requeue-00722-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    shutil.copy2(STATE_DIR / "state.json", bak); print("backed up", bak)
    state.queue.append(entry)
    state.stats["queued"] = int(state.stats.get("queued", 0)) + 1
    state.queue.sort(key=lambda e: e.order_key)
    state._append_history({
        "event": "requeued", "at": datetime.now(timezone.utc).isoformat(), "challenge_id": CID,
        "hotkey": entry.hotkey, "repo": entry.repo, "revision": entry.revision, "uid": UID,
        "reason": REASON, "error_code": "probe_gap_readmitted", "error_detail": DETAIL,
        "original_verdict_at": row["at"], "original_rejection": rej,
        "directive": "Jacob Steeves 2026-09-28 15:23 UTC",
    })
    state.flush()
    print("queue order now:", [e.challenge_id for e in state.queue])
    print(f"requeued {CID}; history row written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
