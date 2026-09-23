"""Merge the per-pod driver output pulled by `run_pods.sh collect` into one
continuations.jsonl (one row per continuation, all arms, `pod` + `arm`
stamped) and a flat traces/ dir (<rollout>_<turn>_<arm>.json) for analyze.py.

    python collect_results.py --src /tmp/fa_outcome/collect --out results/frontier_arbiter/outcome
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DROP = ("stderr",)   # the 3.6k-char stderr tails stay in the per-pod json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="dir with <pod>/out/<tag>*/results/*.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="continuation jsonl rows collected earlier from a pod that is now unreachable")
    args = ap.parse_args()
    rows: dict[str, dict] = {}
    (args.out / "traces").mkdir(parents=True, exist_ok=True)
    n_tr = 0
    for p in sorted(args.src.glob("*/out/*/results/*.json")):
        pod = p.parts[len(args.src.parts)]
        r = json.loads(p.read_text())
        r = {k: v for k, v in r.items() if k not in DROP}
        rep = r.get("report") or {}
        rep.pop("stderr_tail", None)
        r["pod"] = pod
        r["arm"] = r["state_id"].rsplit(":", 1)[1]
        r["kept_state_id"] = r["state_id"].rsplit(":", 1)[0] + ":frontier"
        key = f"{r['state_id']}#{r.get('continuation', 0)}"
        # a later pass (retry) overrides an earlier errored row
        if key not in rows or rows[key].get("status") != "ok":
            rows[key] = r
    for x in args.extra:
        for line in open(x, encoding="utf-8"):
            r = json.loads(line)
            key = f"{r['state_id']}#{r.get('continuation', 0)}"
            if key not in rows or rows[key].get("status") != "ok":
                rows[key] = r
    for p in sorted(args.src.glob("*/out/*/traces/*.json")):
        shutil.copyfile(p, args.out / "traces" / p.name)
        n_tr += 1
    with open(args.out / "continuations.jsonl", "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} continuation rows, {n_tr} traces -> {args.out}")


if __name__ == "__main__":
    main()
