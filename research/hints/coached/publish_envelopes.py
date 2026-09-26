#!/usr/bin/env python
"""Wrap the SOLVED coached continuations as trace envelopes (the datagen
schema the fold reads: rollout_id / source / env_id / task / policy / trace)
with one extra top-level block, `privileged`, that the fold must never put in
a prefix:

  privileged.origin   the king state the continuation started from
                      (king_digest, king_rollout_id, king_turn_idx, state_id,
                      state_kind, resume_kind, plain / coached solve counts,
                      hint_decisive)
  privileged.hints    one entry per teacher step of the continuation:
                      cont_turn, node_idx / node_id of the sampled reply in
                      `trace.nodes`, the note the teacher read (or null when
                      the step ran unhinted, with the reason), the coach's
                      fact / plan / action levels, the gate results
  privileged.coach    coach model, prompt version, inject rule

The trace itself is hint-free by construction (the harness never saw the
note; the proxy appended it downstream of the interception server), so the
D view materializes the same prefixes a miner would see. Policy id is
`coached_<harness>` (model qwen3.8-27b), so the fold's king-group routing
(`policy.id.startswith("king_")`) does not pick these up by accident; a
`king_coached` group would read `privileged.origin` instead.

  python publish_envelopes.py --states RUN/states/states.jsonl \
      --run-out RUN/out --agg RUN/agg --out RUN/envelopes --run-id coached-20260913a
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import uuid
from pathlib import Path

import common as C

HINT_LEVELS_MODEL = "deepseek/deepseek-v4-pro-0813"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--run-out", required=True, type=Path)
    ap.add_argument("--agg", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--all-solved", action="store_true",
                    help="every solved coached continuation (default: only states flagged "
                         "hint_decisive, i.e. the plain teacher solved none of its tries)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    states = {}
    for line in args.states.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            s = json.loads(line)
            states[s.get("proxy_key") or s["state_id"]] = s
    rows = [json.loads(l) for l in (args.agg / "side_table.jsonl").read_text().split("\n") if l.strip()]
    n = 0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with gzip.open(args.out / f"{args.run_id}-coached-envelopes.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            if not r["coached_solved_any"] or not (args.all_solved or r["hint_decisive"]):
                continue
            s = states[r["unit"]]
            full = json.loads(Path(s["path"]).read_text(encoding="utf-8")) if Path(s["path"]).is_file() else None
            if full is None:
                alt = args.states.parent / "states" / Path(s["path"]).name
                full = json.loads(alt.read_text(encoding="utf-8"))
            for trace_id, h in r["hints"].items():
                k = h["continuation"]
                tp = args.run_out / "traces" / f"{C.unit_stem(r['unit'])}.coached.c{k}.json"
                if not tp.is_file():
                    continue
                trace = json.loads(tp.read_text(encoding="utf-8"))
                env = {
                    "schema": 3,
                    "rollout_id": uuid.uuid5(uuid.NAMESPACE_URL, f"{args.run_id}/{r['unit']}/coached/c{k}").hex,
                    "source": s["source"],
                    "env_id": s["env_id"],
                    "task": full["task"],
                    "policy": {"id": f"coached_{s['harness']}", "model": "engy/qwen3.8-27b",
                               "harness": s["harness"], "endpoint": "coach-proxy",
                               "action_kind": s.get("action_kind") or "bash"},
                    "stored_at": stamp,
                    "trace": trace,
                    "privileged": {
                        "origin": {"king_digest": s.get("king_digest"), "king_rollout_id": s["rollout_id"],
                                   "king_turn_idx": s["turn_idx"], "state_id": s["state_id"],
                                   "state_kind": s["state_kind"], "resume_kind": s["resume_kind"],
                                   "turn_id": s.get("turn_id"), "node_id": s.get("node_id"),
                                   "king_policy_id": s.get("policy_id"),
                                   "coached_n": r["coached_n"], "coached_n_solved": r["coached_n_solved"],
                                   "plain_n": r["plain_n"], "plain_n_solved": r["plain_n_solved"],
                                   "plain_source": r["plain_source"],
                                   "hint_decisive": r["hint_decisive"],
                                   "hint_decisive_strict": r["hint_decisive_strict"],
                                   "run_id": args.run_id, "continuation": k},
                        "hints": h["steps"],
                        "coach": {"model": HINT_LEVELS_MODEL, "prompt_version": C.PROMPT_VERSION,
                                  "inject": C.INJECT_DEFAULT, "hint_header": C.HINT_HEADER.strip()},
                    },
                }
                f.write(json.dumps(env, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} coached envelopes -> {args.out}")


if __name__ == "__main__":
    main()
