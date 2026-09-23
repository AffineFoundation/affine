#!/usr/bin/env python
"""E4 — build continuation states for the usefulness gate.

For every probe turn on a resumable harness (mini-swe textbased, verifiers
bash, Terminus 2) and every requested hint, write TWO state files in the
ops/recoverable format (PR #13, `states.py::build_state`):

  <rollout>:<turn>:nohint    the prefix as recorded
  <rollout>:<turn>:<hint>    the same prefix with the hint appended to the
                             last user / tool message (what the teacher sees
                             when it samples a hinted reference)

and list each state `--reps` times in states.jsonl (distinct state_ids,
same file) so run_states.py samples that many continuations. The
usefulness gate then reads: keep the hint iff solved(hinted) > solved(nohint).

  python e4_states.py --turns turns.jsonl --hints RUN/hints.jsonl \
      --recoverable-src /tmp/hints-data/recoverable_src/ops/recoverable \
      --out RUN/e4 --hint-conds deepseek:fact --reps 2 --limit 60
"""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(HERE))

from clients import HINT_HEADER  # noqa: E402

TRACES = Path("/tmp/hints-data/box/king_review/traces")


def load_trace_index() -> dict[str, dict]:
    idx = {}
    with open(TRACES / "rollout_index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            idx[r["rollout_id"]] = r
    return idx


def load_envelope(ri: dict) -> dict:
    with gzip.open(TRACES / "chunks" / ri["chunk"], "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ri["line"]:
                return json.loads(line)
    raise KeyError(ri["rollout_id"])


def hinted_messages(messages: list[dict], hint: str) -> list[dict]:
    msgs = copy.deepcopy(messages)
    for m in reversed(msgs):
        if m["role"] in ("user", "tool"):
            m["content"] = (m.get("content") or "") + HINT_HEADER + hint
            break
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--hints", required=True)
    ap.add_argument("--recoverable-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hint-conds", default="deepseek:fact")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--groups", default="king_loop_onset,king_pivot")
    ap.add_argument("--grounded-only", action="store_true")
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    sys.path.insert(0, args.recoverable_src)
    import states as RS  # noqa: E402  (PR #13 ops/recoverable/states.py)

    out = Path(args.out)
    (out / "states").mkdir(parents=True, exist_ok=True)
    hints = {}
    for line in open(args.hints):
        r = json.loads(line)
        if r.get("ok") and r.get("text"):
            hints[(r["turn_id"], f"{r['generator']}:{r['level']}")] = r
    conds = args.hint_conds.split(",")
    turns = [json.loads(l) for l in open(args.turns)]
    turns = [t for t in turns if t["group"] in args.groups.split(",")
             and t["harness"] in RS.RESUMABLE and t["harness"] != "null"]
    rng = random.Random(args.seed)
    rng.shuffle(turns)
    # stratify by harness: round-robin pick until the limit
    by_h = collections.defaultdict(list)
    for t in turns:
        by_h[t["harness"]].append(t)
    picked = []
    while len(picked) < args.limit and any(by_h.values()):
        for h in sorted(by_h):
            if by_h[h] and len(picked) < args.limit:
                picked.append(by_h[h].pop())
    tidx = load_trace_index()
    rows = []
    counts = collections.Counter()
    for t in picked:
        ri = tidx.get(t["rollout_id"])
        if ri is None:
            counts["no_trace"] += 1
            continue
        env = load_envelope(ri)
        label = {"node_id": t.get("node_id"), "group": t["group"], "turn_id": t["turn_id"]}
        base = RS.build_state(env, int(t["turn_idx"]), "probe", label)
        if base is None:
            counts["bad_turn"] += 1
            continue
        arms = [("nohint", None)]
        for c in conds:
            h = hints.get((t["turn_id"], c))
            if h is None or (args.grounded_only and not h["grounding"]["grounded"]):
                counts[f"no_hint:{c}"] += 1
                continue
            arms.append((c.replace(":", "-"), h))
        if len(arms) < 2:
            continue
        for arm, h in arms:
            st = dict(base)
            st["arm"] = arm
            st["hint"] = h["text"] if h else None
            st["hint_id"] = h.get("hint_id") if h else None
            st["hint_grounded"] = (h.get("grounding") or {}).get("grounded") if h else None
            st["probe_turn_id"] = t["turn_id"]
            st["probe_group"] = t["group"]
            if h:
                st["messages"] = hinted_messages(base["messages"], h["text"])
            sid_base = f"{t['rollout_id']}:{t['turn_idx']}:{arm}"
            path = out / "states" / (sid_base.replace(":", "_") + ".json")
            st["state_id"] = sid_base
            path.write_text(json.dumps(st, ensure_ascii=False))
            for rep in range(args.reps):
                meta = {k: v for k, v in st.items()
                        if k not in ("messages", "king_reply", "task_system_prompt", "task_prompt", "label")}
                meta["state_id"] = f"{sid_base}:r{rep}"
                meta["rep"] = rep
                meta["path"] = str(path)
                rows.append(meta)
                counts[f"state:{arm}:{t['harness']}"] += 1
    with open(out / "states.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}", file=sys.stderr)
    print(f"{len(rows)} continuation jobs over {len(picked)} turns -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
