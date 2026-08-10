"""Per-provider yield/cost report from state.jsonl.

  python -m datagen.stats [--state /root/datagen/state.jsonl] \
      [--prices '{"openrouter/z-ai/glm-5.2": [0.76, 2.42]}'] [--since HOURS]

Prices are $/M tokens [input, output] keyed by the turn-record model label;
lanes without a price report tokens only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_PRICES = {
    "openrouter/z-ai/glm-5.2": (0.76, 2.42),
    "openrouter/z-ai/glm-4.5-air": (0.13, 0.85),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", type=Path,
                    default=Path("/root/datagen/state.jsonl"))
    ap.add_argument("--prices", type=json.loads, default=None)
    ap.add_argument("--since", type=float, default=0.0,
                    help="only count records from the last N hours")
    args = ap.parse_args()
    prices = dict(DEFAULT_PRICES)
    if args.prices:
        prices.update({k: tuple(v) for k, v in args.prices.items()})

    cutoff = time.time() - args.since * 3600 if args.since else 0.0
    lanes: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    t_min = t_max = None
    for line in open(args.state, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        ts = float(rec.get("ts") or 0)
        if ts < cutoff:
            continue
        t_min = ts if t_min is None else min(t_min, ts)
        t_max = ts if t_max is None else max(t_max, ts)
        lane = lanes[rec.get("provider") or "?"]
        lane[rec.get("outcome", "?")] += 1
        lane["n_turns"] += rec.get("n_turns", 0)
        lane["prompt_tokens"] += rec.get("prompt_tokens", 0)
        lane["completion_tokens"] += rec.get("completion_tokens", 0)

    span_h = ((t_max - t_min) / 3600) if (t_min and t_max and t_max > t_min) else None
    report = {"span_hours": round(span_h, 2) if span_h else None, "lanes": {}}
    for name, lane in sorted(lanes.items()):
        n = sum(lane[k] for k in ("resolved", "unresolved", "error"))
        row = {
            "instances": int(n),
            "resolved": int(lane["resolved"]),
            "unresolved": int(lane["unresolved"]),
            "error": int(lane["error"]),
            "turns": int(lane["n_turns"]),
            "prompt_Mtok": round(lane["prompt_tokens"] / 1e6, 2),
            "completion_Mtok": round(lane["completion_tokens"] / 1e6, 2),
        }
        if name in prices:
            pin, pout = prices[name]
            cost = (lane["prompt_tokens"] * pin
                    + lane["completion_tokens"] * pout) / 1e6
            row["est_cost_usd"] = round(cost, 2)
            if lane["resolved"]:
                row["usd_per_resolved"] = round(cost / lane["resolved"], 3)
        if n:
            row["resolve_rate"] = round(lane["resolved"] / n, 3)
        report["lanes"][name] = row
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
