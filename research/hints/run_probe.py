#!/usr/bin/env python
"""E1 / E2 / E3 measurement pass: hinted vs unhinted teacher references.

For every turn of the turn set and every CONDITION (a hint text, or none):
  * sample k = 3 teacher references (z_C^i, y_i) under x + h at the duel's
    temperature / token cap (H0 = no hint samples 4: 3 refs + 1 held-out);
  * echo, under PLAIN x (G stays unhinted by design):
      lp_own    = lpC(y_i | x, z_C^i)      lp_empty = lpC(y_i | x, ∅)
      lp_thought = lpC(z_C^i | x)          (band echo t_i)
  * for every MINER thought z_A (teacher held-out, king live sample, the
    recorded reply, stored duel king / challenger where they exist):
      m = lpC(z_A | x), B echoes lpC(y_A|x,z_A), lpC(y_A|x,∅) once per miner,
      and lpC(y_i | x, z_A) for every ref of every condition (the a_i numerator).
Everything (raw completion text, split, every echo, seeds, timings) is
appended to <run_dir>/results.jsonl, one row per turn. Resumable: turns
already in results.jsonl are skipped. E2 (ref mixing) and every metric are
computed offline from these rows by analyze.py — no extra GPU work.

  python run_probe.py --turns turns.jsonl --hints RUN/hints.jsonl --run-dir RUN \
      --conditions H0,ds_fact,ds_plan,ds_action,self_fact,pivot_action \
      --turn-conc 6 --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(HERE))

from affine import dialects  # noqa: E402
from evalsrv.chat import split_rollout, think_closed  # noqa: E402

from clients import Box, load_boxes, load_king, with_hint  # noqa: E402

TEMPERATURE = 0.8            # [duel].temperature
MAX_TOKENS = 1024 + 768      # max_thought_tokens + max_action_tokens
K_REFS = 3                   # [duel].n_teacher_samples
EMPTY = ""

# condition name -> (generator, level); H0 is the unhinted baseline.
CONDITIONS = {
    "H0": None,
    "ds_fact": ("deepseek", "fact"),
    "ds_plan": ("deepseek", "plan"),
    "ds_action": ("deepseek", "action"),
    "self_fact": ("self", "fact"),
    "self_plan": ("self", "plan"),
    "self_action": ("self", "action"),
    "pivot_action": ("pivot", "action"),
    "pivot_plan": ("pivot", "plan"),
}


def log(msg: str) -> None:
    print(f"[probe] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_hints(path: Path) -> dict[tuple[str, str, str], dict]:
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        r = json.loads(line)
        if r.get("ok") and r.get("text"):
            out[(r["turn_id"], r["generator"], r["level"])] = r
    return out


def recorded_miner(turn: dict) -> tuple[str, str] | None:
    """The reply actually recorded at this turn (king's own for the king
    groups, the teacher's own for completion), split like a rollout."""
    thought = turn.get("reference_thought") or ""
    visible = turn.get("reference_turn") or ""
    text = thought + "\n</think>\n" + visible
    z, y = split_rollout(text, turn["action_kind"], require_think_close=True)
    if not y:
        return None
    return z, y


def stored_miners(turn: dict) -> dict[str, tuple[str, str]]:
    out = {}
    for rec in turn.get("stored") or []:
        if rec.get("king") and rec.get("chal"):
            out["stored_king"] = (rec["king"]["z"], rec["king"]["y"])
            out["stored_chal"] = (rec["chal"]["z"], rec["chal"]["y"])
            out["_stored_cid"] = rec["cid"]
            break
    return out


async def sample_side(box: Box, prefix: list[dict], n: int, *, action_kind: str,
                      require_think_close: bool) -> list[dict]:
    texts = await asyncio.gather(*[box.sample_raw(prefix, TEMPERATURE, MAX_TOKENS)
                                   for _ in range(n)])
    out = []
    for raw in texts:
        z, y = split_rollout(raw, action_kind, require_think_close=require_think_close)
        out.append({"raw": raw, "z": z, "y": y, "valid": bool(y),
                    "think_closed": think_closed(raw),
                    "n_chars": len(raw)})
    return out


async def score_refs(box: Box, prefix: list[dict], refs: list[dict]) -> None:
    valid = [r for r in refs if r["valid"]]
    if not valid:
        return
    res = await asyncio.gather(
        *[box.score_action(prefix, r["z"], r["y"]) for r in valid],
        *[box.score_action(prefix, EMPTY, r["y"]) for r in valid],
        *[box.score_thought(prefix, r["z"]) for r in valid])
    k = len(valid)
    for i, r in enumerate(valid):
        r["lp_own"] = res[i]["lp_per_byte"]
        r["lp_empty"] = res[k + i]["lp_per_byte"]
        r["lp_thought"] = res[2 * k + i]["lp_per_byte"]
        r["echo_tokens"] = {"own": res[i]["n_tokens"], "thought": res[2 * k + i]["n_tokens"]}


async def score_miner(box: Box, prefix: list[dict], miner: dict,
                      conditions: dict[str, dict]) -> None:
    z, y = miner["z"], miner["y"]
    tasks = [box.score_thought(prefix, z),
             box.score_action(prefix, z, y),
             box.score_action(prefix, EMPTY, y)]
    keys = []
    for cname, cond in conditions.items():
        for i, r in enumerate(cond["refs"]):
            if r["valid"]:
                tasks.append(box.score_action(prefix, z, r["y"]))
                keys.append((cname, i))
    res = await asyncio.gather(*tasks)
    miner["m"] = res[0]["lp_per_byte"]
    miner["lpC_ya_za"] = res[1]["lp_per_byte"]
    miner["lpC_ya_e"] = res[2]["lp_per_byte"]
    a: dict[str, list] = {c: [None] * len(conditions[c]["refs"]) for c in conditions}
    for (cname, i), r in zip(keys, res[3:]):
        a[cname][i] = r["lp_per_byte"]
    miner["lpC_yc_za"] = a


async def run_turn(turn: dict, box: Box, king: Box | None, hints, cond_names: list[str],
                   seed: int) -> dict:
    t0 = time.time()
    prefix = turn["prefix"]
    kind = turn["action_kind"]
    row = {"turn_id": turn["turn_id"], "group": turn["group"], "action_kind": kind,
           "harness": turn["harness"], "source": turn["source"],
           "n_prefix_chars": turn["n_prefix_chars"], "turn_idx": turn["turn_idx"],
           "box": box.name, "seed": seed, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "conditions": {}, "miners": {}, "errors": []}
    # 1. teacher samples per condition (H0 samples one extra held-out).
    cond_specs = []
    for cname in cond_names:
        if cname == "H0":
            cond_specs.append((cname, None))
            continue
        gen, level = CONDITIONS[cname]
        h = hints.get((turn["turn_id"], gen, level))
        if h is None:
            continue
        cond_specs.append((cname, h))
    samples = await asyncio.gather(*[
        sample_side(box, with_hint(prefix, h["text"] if h else None),
                    K_REFS + (1 if cname == "H0" else 0), action_kind=kind,
                    require_think_close=False)
        for cname, h in cond_specs])
    heldout = None
    for (cname, h), sm in zip(cond_specs, samples):
        if cname == "H0":
            # The 4th unhinted sample is the teacher-as-miner control; it
            # must satisfy the miner rule (closed </think>).
            extra = sm[K_REFS]
            z, y = split_rollout(extra["raw"], kind, require_think_close=True)
            heldout = {"source": "teacher_heldout", "raw": extra["raw"], "z": z, "y": y,
                       "valid": bool(y), "think_closed": extra["think_closed"]}
            sm = sm[:K_REFS]
        row["conditions"][cname] = {
            "hint_id": h.get("hint_id") if h else None,
            "hint": h.get("text") if h else None,
            "generator": h.get("generator") if h else None,
            "level": h.get("level") if h else None,
            "grounded": (h.get("grounding") or {}).get("grounded") if h else None,
            "leaks_future": (h.get("leak") or {}).get("leaks_future") if h else None,
            "refs": sm, "n_valid": sum(1 for r in sm if r["valid"]),
        }
    # 2. ref echoes under plain x.
    await asyncio.gather(*[score_refs(box, prefix, c["refs"]) for c in row["conditions"].values()])
    # 3. miners.
    miners: dict[str, dict] = {}
    if heldout:
        miners["teacher_heldout"] = heldout
    rec = recorded_miner(turn)
    if rec:
        miners["recorded"] = {"source": turn.get("policy_id"), "z": rec[0], "y": rec[1], "valid": True}
    st = stored_miners(turn)
    for name in ("stored_king", "stored_chal"):
        if name in st:
            miners[name] = {"source": st.get("_stored_cid"), "z": st[name][0], "y": st[name][1], "valid": True}
    if king is not None:
        try:
            ks = (await sample_side(king, prefix, 1, action_kind=kind, require_think_close=True))[0]
            ks["source"] = "king_live"
            miners["king_live"] = ks
        except Exception as e:  # noqa: BLE001 — the king box is best-effort, low rate
            row["errors"].append(f"king_live: {type(e).__name__}: {str(e)[:200]}")
    await asyncio.gather(*[score_miner(box, prefix, m, row["conditions"])
                           for m in miners.values() if m.get("valid")])
    row["miners"] = miners
    row["seconds"] = round(time.time() - t0, 1)
    row["n_calls_box"] = box.n_calls
    return row


async def worker(box: Box, queue: asyncio.Queue, out, hints, cond_names, king, seed, lock, stats):
    while True:
        turn = await queue.get()
        if turn is None:
            queue.task_done()
            return
        try:
            row = await run_turn(turn, box, king, hints, cond_names, seed)
        except Exception as e:  # noqa: BLE001 — record, continue with the next turn
            row = {"turn_id": turn["turn_id"], "group": turn["group"], "box": box.name,
                   "failed": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc()[-1500:]}
            log(f"{box.name} {turn['turn_id'][:40]} FAILED {row['failed']}")
        async with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            stats["done"] += 1
            if "seconds" in row:
                stats["secs"] += row["seconds"]
                log(f"{box.name} {stats['done']}/{stats['total']} {turn['turn_id'][:40]} "
                    f"{row['seconds']}s prefix={turn['n_prefix_chars']} "
                    f"conds={len(row['conditions'])} miners={len(row['miners'])}")
        queue.task_done()


async def main_async(args) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "results.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            if not r.get("failed") or args.retry_failed is False:
                done.add(r["turn_id"])
    turns = [json.loads(l) for l in open(args.turns)]
    if args.groups:
        turns = [t for t in turns if t["group"] in args.groups.split(",")]
    rng = random.Random(args.seed)
    rng.shuffle(turns)
    turns = [t for t in turns if t["turn_id"] not in done]
    if args.limit:
        turns = turns[: args.limit]
    hints = load_hints(Path(args.hints)) if args.hints else {}
    cond_names = args.conditions.split(",")
    boxes = load_boxes(Path(args.pods_state), concurrency=args.box_conc)
    king = load_king(Path(args.king_tok_dir)) if args.king else None
    log(f"{len(turns)} turns to do ({len(done)} done), boxes={[b.name for b in boxes]}, "
        f"king={'yes' if king else 'no'}, conditions={cond_names}, hints={len(hints)}")
    queue: asyncio.Queue = asyncio.Queue()
    for t in turns:
        queue.put_nowait(t)
    n_workers = len(boxes) * args.turn_conc
    for _ in range(n_workers):
        queue.put_nowait(None)
    lock = asyncio.Lock()
    stats = {"done": 0, "total": len(turns), "secs": 0.0}
    with open(out_path, "a") as out:
        tasks = [asyncio.create_task(worker(b, queue, out, hints, cond_names, king, args.seed, lock, stats))
                 for b in boxes for _ in range(args.turn_conc)]
        await asyncio.gather(*tasks)
    log(f"finished: {stats['done']} turns, {stats['secs']:.0f} turn-seconds")
    for b in boxes:
        await b.aclose()
    if king:
        await king.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--hints")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--conditions", default="H0,ds_fact,ds_plan,ds_action,self_fact,pivot_action")
    ap.add_argument("--groups", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--turn-conc", type=int, default=6, help="turns in flight per box")
    ap.add_argument("--box-conc", type=int, default=64, help="max concurrent requests per box")
    ap.add_argument("--king", action="store_true", help="also sample the live king (KING_* env)")
    ap.add_argument("--king-tok-dir", default=os.environ.get("HINTS_KING_TOK", "/tmp/hints-data/king_tok"))
    ap.add_argument("--pods-state", default=os.environ.get("HINTS_PODS_STATE", "/tmp/hints-secrets/pods.json"))
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
