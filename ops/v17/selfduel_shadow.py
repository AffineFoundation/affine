#!/usr/bin/env python3
"""Null self-duel on the idle eval pod to read the sd-meter shadow while the
queue is empty (2026-09-18): challenger = the standing king. Talks to the pod
directly (127.0.0.1:9000 via the validator tunnel) — the validator, its
state and history are untouched; a real dispatch supersedes this job.
Expected: live and shadow margin ≈ 0, would_crown false; the shadow's σ per
dialect, bind fractions, teacher-vs-king control, live-floor calibration and
echo cost are the numbers the wvk-22 gate needs.

  python ops/v17/selfduel_shadow.py            # start + follow, writes ops/v17/selfduel_<ts>.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:9000"


def env(name: str) -> str:
    v = os.environ.get(name)
    if v:
        return v
    for line in (Path.home() / ".affine-validator.env").read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing")


def main() -> int:
    tok = env("AFFINE_EVAL_TOKEN")
    state = json.load(open(REPO / "affine" / "state" / "state.json"))
    king = state["king"]
    hk = king.get("hotkey") or "self-duel"
    payload = {
        "king_repo": king["repo"], "king_revision": king["revision"],
        "challenger_repo": king["repo"], "challenger_revision": king["revision"],
        "challenger_hotkey": hk,
        "block_hash": "0x" + "5d" * 32,     # slice seed only
        "challenger_weight_bytes": 0,
    }
    h = {"X-Affine-Token": tok}
    with httpx.Client(timeout=60) as c:
        r = c.post(f"{BASE}/duel", json=payload, headers=h)
        r.raise_for_status()
        job = r.json()["job_id"]
        print(time.strftime("%FT%TZ", time.gmtime()), "job", job, "king", king["repo"][-40:], flush=True)
        last = None
        while True:
            time.sleep(30)
            try:
                s = c.get(f"{BASE}/duel/{job}", headers=h).json()
            except Exception as e:  # tunnel hiccup
                print("poll error", e, flush=True)
                continue
            st = s.get("state")
            ph = s.get("phase")
            prog = s.get("progress")
            key = (st, ph, json.dumps(prog, sort_keys=True) if prog else None)
            if key != last:
                print(time.strftime("%FT%TZ", time.gmtime()), st, ph, prog, flush=True)
                last = key
            if st in ("completed", "failed", "superseded", "error"):
                break
        out = HERE / f"selfduel_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        out.write_text(json.dumps(s, indent=1))
        print("wrote", out, flush=True)
        v = s.get("verdict") or {}
        sd = (v.get("shadow") or {}).get("sd_meter")
        print("live margin", v.get("margin"), "se", v.get("se"), "z", v.get("z"),
              "n", v.get("n_paired_turns"), "duel_seconds", v.get("duel_seconds"))
        if sd:
            print(json.dumps({k: sd.get(k) for k in ("anchor", "margin", "se", "z", "sd_diff", "would_crown",
                                                       "sigma_by_dialect", "n_loo_turns_by_dialect")}, indent=1))
            print("cost", json.dumps(sd.get("cost"), indent=1))
            for mode, b in sd["by_anchor"].items():
                if b.get("available"):
                    print(mode, "binds chal", b["challenger"]["bind_frac"], "teacher_vs_king", b["teacher_vs_king"])
        else:
            print("NO shadow block:", list(v.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
