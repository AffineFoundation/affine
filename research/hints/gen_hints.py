#!/usr/bin/env python
"""Generate and gate hints for every turn of the turn set.

Writes <run_dir>/hints.jsonl, one row per (turn, generator, level):
  turn_id, generator (deepseek | self | pivot), level (fact | plan | action),
  text, grounding {entities, missing, grounded}, leak {leaks_future, overlap},
  n_sentences, raw model text, usage / cost, latency, model, prompt sha.
Resumable: (turn_id, generator) pairs already present are skipped.

  python gen_hints.py --turns /tmp/hints-data/turns.jsonl --run-dir RUN \
      --generators deepseek,self,pivot --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hints as H  # noqa: E402

_lock = threading.Lock()


def prefix_text(turn: dict) -> str:
    return "\n".join(m["content"] for m in turn["prefix"])


def future_actions(turn: dict) -> list[str]:
    acts = (turn.get("hindsight") or {}).get("actions") or []
    t = int(turn["turn_idx"])
    fut = [a for a in acts[t:] if a]
    if turn.get("reference_action"):
        fut.append(turn["reference_action"])
    return fut


def gate(turn: dict, text: str) -> dict:
    return {"grounding": H.grounding_check(text, prefix_text(turn)),
            "leak": H.leak_check(text, future_actions(turn), prefix_text(turn)),
            "n_sentences": H.sentence_count(text), "n_chars": len(text)}


def rows_for(turn: dict, generator: str, parsed: dict | None, raw: dict, prompt_sha: str) -> list[dict]:
    out = []
    levels = parsed or {}
    for level in ("fact", "plan", "action"):
        text = levels.get(level)
        row = {"turn_id": turn["turn_id"], "group": turn["group"],
               "action_kind": turn["action_kind"], "generator": generator,
               "level": level, "text": text, "ok": bool(text),
               "model": raw.get("model"), "usage": raw.get("usage"),
               "cost_usd": raw.get("cost_usd"), "ms": raw.get("ms"),
               "finish": raw.get("finish"), "error": raw.get("error"),
               "prompt_sha": prompt_sha, "prompt_version": H.PROMPT_VERSION}
        if text:
            row.update(gate(turn, text))
            row["hint_id"] = H.hint_id(turn["turn_id"], generator, level, text)
        out.append(row)
    return out


def gen_one(turn: dict, generator: str, cfg: dict) -> tuple[list[dict], dict]:
    hs = turn.get("hindsight") or {}
    transcript = H.transcript_with_boundary(hs.get("transcript", ""), int(turn["turn_idx"]))
    outcome = f"{turn.get('outcome')} (stop={hs.get('stop_condition')})"
    messages = H.build_messages(transcript, int(turn["turn_idx"]), outcome)
    prompt_sha = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:16]
    if generator == "deepseek":
        raw = H.openrouter_hints(messages, cfg["openrouter_key"])
    elif generator == "self":
        raw = H.vllm_hints(messages, cfg["self_base_url"], cfg["self_key"], cfg["self_model"])
    else:
        raise ValueError(generator)
    parsed = H.parse_hints(raw.get("text", ""))
    rows = rows_for(turn, generator, parsed, raw, prompt_sha)
    rec = {"turn_id": turn["turn_id"], "generator": generator, "prompt_sha": prompt_sha,
           "messages": messages, "raw": raw}
    return rows, rec


def pivot_rows(turn: dict) -> list[dict]:
    piv = turn.get("pivot") or {}
    out = []
    for level, key in (("action", "should_have"), ("plan", "rationale")):
        text = piv.get(key)
        if not text:
            continue
        text = " ".join(str(text).split())
        row = {"turn_id": turn["turn_id"], "group": turn["group"],
               "action_kind": turn["action_kind"], "generator": "pivot",
               "level": level, "text": text, "ok": True, "model": "deepseek/deepseek-v4-pro-0813",
               "prompt_version": "kr-v2", "hint_id": H.hint_id(turn["turn_id"], "pivot", level, text)}
        row.update(gate(turn, text))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--generators", default="deepseek,self,pivot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-base-url")
    ap.add_argument("--self-key-env", default="SELF_KEY")
    ap.add_argument("--pods-state", default=os.environ.get("HINTS_PODS_STATE", "/tmp/hints-secrets/pods.json"),
                    help="take the self-hint endpoint (base_url + bearer) from this pods.json")
    ap.add_argument("--pod-index", type=int, default=0)
    ap.add_argument("--self-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--max-usd", type=float, default=60.0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "hints.jsonl"
    raw_path = run_dir / "hint_calls.jsonl"
    turns = [json.loads(l) for l in open(args.turns)]
    if args.limit:
        turns = turns[: args.limit]
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["turn_id"], r["generator"]))
    gens = args.generators.split(",")
    cfg = {"openrouter_key": os.environ.get("OPENROUTER_API_KEY", ""),
           "self_base_url": args.self_base_url, "self_key": os.environ.get(args.self_key_env, ""),
           "self_model": args.self_model}
    if "self" in gens and not cfg["self_base_url"] and Path(args.pods_state).exists():
        pods = [m for m in json.loads(Path(args.pods_state).read_text())["pods"].values()
                if m.get("ready_at")]
        mem = pods[args.pod_index % len(pods)]
        cfg["self_base_url"], cfg["self_key"] = mem["base_url"], mem["key"]
    spent = 0.0
    jobs = []
    for t in turns:
        for g in gens:
            if (t["turn_id"], g) in done:
                continue
            jobs.append((t, g))
    print(f"{len(jobs)} (turn, generator) jobs; {len(done)} already done", file=sys.stderr)
    t0 = time.time()
    n = 0
    with open(out_path, "a") as out, open(raw_path, "a") as raw_out, \
            cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for t, g in jobs:
            if g == "pivot":
                rows = pivot_rows(t)
                for r in rows:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                if not rows:
                    out.write(json.dumps({"turn_id": t["turn_id"], "generator": "pivot",
                                          "level": None, "text": None, "ok": False}) + "\n")
                continue
            futs.append(ex.submit(gen_one, t, g, cfg))
        for fut in cf.as_completed(futs):
            try:
                rows, rec = fut.result()
            except Exception as e:  # noqa: BLE001 — one bad call must not stop the pass
                print(f"job failed: {e!r}", file=sys.stderr)
                continue
            with _lock:
                for r in rows:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    spent += float(r.get("cost_usd") or 0) / 3
                raw_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush(); raw_out.flush()
                n += 1
                if n % 20 == 0:
                    print(f"{n}/{len(futs)} done, ${spent:.2f}, {time.time() - t0:.0f}s",
                          file=sys.stderr)
                if spent > args.max_usd:
                    print(f"STOP: spend ${spent:.2f} > cap ${args.max_usd}", file=sys.stderr)
                    for f in futs:
                        f.cancel()
                    break
    print(f"done: {n} calls, ≈${spent:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
