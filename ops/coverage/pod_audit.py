"""Which bench-king-* pods belong to whom — and release the orphans.

    python pod_audit.py            # list: ledger / pass / ORPHAN
    python pod_audit.py --release  # release every ORPHAN (a pod no running pass
                                   # and no ledger row claims), verify in the Lium listing

An orphan is what a killed driver leaves behind when its kingpod rent finished
after the kill (2026-09-17 00:37 UTC: two duplicate K13 launches).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
import lium_api  # noqa: E402

BENCH = REPO / "ops" / "benchsuite"
PY = str(REPO / ".venv" / "bin" / "python")
GRACE_S = 15 * 60   # a pod younger than this may belong to a pass that has not logged its name yet


def owners() -> tuple[set[str], set[str]]:
    ledger = {r["pod"] for r in json.loads((HERE / "state" / "backfill_pods.json").read_text())}
    procs = subprocess.run(["pgrep", "-af", "pass.sh|kingpod.py|fast_pass"], capture_output=True, text=True).stdout
    owned = set(re.findall(r"kingpod.py (?:wait|rent) (\S+)", procs))
    # any pass (mine, the watcher's, the benchsuite worker's by-hand / fast passes) that is still
    # writing its log owns every pod named in it; a log silent for > 3 h with no .exit is dead
    for log_path in (BENCH / "state").glob("pass-*.log"):
        if log_path.with_suffix(".exit").exists() or time.time() - log_path.stat().st_mtime > 3 * 3600:
            continue
        text = log_path.read_text(errors="replace")
        owned |= set(re.findall(r"(bench-king-[0-9a-f\-]+)", text))
    return ledger, owned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true")
    args = ap.parse_args()
    pods = {k: v for k, v in json.loads((BENCH / "state" / "pods.json").read_text()).items()
            if isinstance(v, dict) and v.get("state") != "released"}
    ledger, owned = owners()
    sess = lium_api.session()
    listed = {lium_api.pod_name(p) for p in (lium_api.pods(sess) or [])}
    orphans = []
    for name, m in sorted(pods.items()):
        age = time.time() - m["rented_at"]
        who = "ledger" if name in ledger else "pass" if name in owned else ("young" if age < GRACE_S else "ORPHAN")
        print(f"{name:40} {m['state']:8} {m['plan']['name']:11} ${m['price']:<5} {age / 3600:5.1f}h listed={name in listed} {who}")
        if who == "ORPHAN" and name in listed:
            orphans.append(name)
    if args.release:
        for name in orphans:
            r = subprocess.run([PY, str(BENCH / "kingpod.py"), "release", name], capture_output=True, text=True, cwd=str(BENCH))
            print("release", name, "->", (r.stdout or r.stderr).strip()[-120:])
        if orphans:
            time.sleep(20)
            listed = {lium_api.pod_name(p) for p in (lium_api.pods(sess) or [])}
            print("still listed:", [n for n in orphans if n in listed] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
