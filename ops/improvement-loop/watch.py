#!/usr/bin/env python
"""Improvement-loop watcher (pm2 `affine-improvement-loop`): after every
benchmark-suite publish of a NEW king (a scorecard under
affine/state/benchsuite/ whose king digest has not been attributed yet),
write the per-reign attribution report and post its one-line axis readout
to the private Arbos Discord channel.

Loop every --interval seconds:
  1. list scorecards (publish.py output); skip cards without rows, cards
     already attributed, and cards of a king that has no earlier reign to
     compare against.
  2. run attribution.attribute(run_id) -> affine/state/improvement_loop/
     <run_id>.md + .json, latest.json, axis.jsonl; Discord one-liner.
  3. a card still marked partial (status != complete) is attributed once as
     "partial" and again when it completes (the report is rewritten, the
     Discord line is posted again with "(final)").

State: ops/improvement-loop/state/watch.json. `--once` does one tick;
`--dry-run` logs the Discord line instead of posting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from attribution import OUT_DIR, attribute, load_cards, log, pick_previous

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "state"
WATCH_JSON = STATE_DIR / "watch.json"


def load_watch() -> dict:
    if WATCH_JSON.exists():
        try:
            return json.loads(WATCH_JSON.read_text())
        except ValueError:
            pass
    return {"done": {}}


def save_watch(w: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WATCH_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(w, indent=1))
    tmp.replace(WATCH_JSON)


def tick(w: dict, out_dir: Path, discord: bool, dry_run: bool) -> None:
    cards = load_cards()
    for card in cards:
        run_id = card["run_id"]
        status = card.get("status") or "complete"
        prior = w["done"].get(run_id)
        if prior and (prior.get("status") == "complete" or prior.get("status") == status):
            continue
        if card.get("mode") == "challenger" or not (card.get("king") or {}).get("reign"):
            w["done"][run_id] = {"status": status, "skipped": "not a crowned king (challenger / comparables card)"}
            continue
        if not pick_previous(cards, card):
            w["done"][run_id] = {"status": status, "skipped": "no previous reign"}
            continue
        try:
            res = attribute(run_id, None, out_dir, discord, dry_run)
        except SystemExit as e:
            log(f"{run_id}: {e}")
            w["done"][run_id] = {"status": status, "error": str(e)}
            save_watch(w)
            continue
        w["done"][run_id] = {"status": status, "answer": res["answer"], "against": res["against"],
                             "at": res["generated_at"]}
        log(f"{run_id}: attributed ({status}) -> {res['answer']}")
    save_watch(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    w = load_watch()
    while True:
        try:
            tick(w, Path(a.out), not a.no_discord, a.dry_run)
        except Exception as e:  # noqa: BLE001  (a watcher must not die on one bad card)
            log(f"tick failed: {e!r}")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
