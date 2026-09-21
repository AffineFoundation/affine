"""Split-states probe (2026-09-21) — stage 2 (Y2): at states the teacher never
solved (s = 0/N from the arm-T run; plus yesterday's contested states whose
single T continuation failed) take the frontier proposals already sampled in
samples.jsonl (glm-5.3 greedy = arm F1, glm-5.3 @T0.8 = arm F1b), sample one
more glm-5.3 @T0.8 proposal (arm F1c) when fewer than two distinct parsed
proposals exist, judge the unjudged ones (glm-5.3-flash, sample.judge_pair),
and write the F1 arm files for the pod driver (forced_reply = the proposal,
then the teacher finishes).

    python select_f1.py --selected .../split_states/selected.jsonl --results .../split_states/continuations.jsonl \
        --samples .../outcome/samples.jsonl --kept .../outcome/kept.jsonl --prev .../outcome/continuations.jsonl \
        --out /tmp/fa_split/arms_split --proposals .../split_states/proposals.jsonl --max-states 16
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Engy, jaccard, norm_action, read_jsonl, write_jsonl  # noqa: E402
from sample import FRONTIER, SAMPLE_MAX_TOKENS, action_of, judge_pair  # noqa: E402
from select_states import META_DROP, forced_reply_of  # noqa: E402
from split_yield import first_action  # noqa: E402

ARM_OF = {"greedy": "F1", "t08": "F1b", "extra": "F1c"}


def usable(p: dict | None) -> bool:
    return bool(p and p.get("y") and p.get("harness_valid", True) and not p.get("finish_reply"))


async def build(args: argparse.Namespace) -> None:
    selected = {m["state_id"].replace(":T", ":frontier"): m for m in read_jsonl(args.selected)}
    samples = {r["state_id"]: r for r in read_jsonl(args.samples)}
    kept = {r["state_id"]: r for r in read_jsonl(args.kept)}
    prev = read_jsonl(args.prev)
    results = read_jsonl(args.results) if args.results.exists() else []
    by_state: dict[str, list[dict]] = collections.defaultdict(list)
    for r in results:
        sid, arm = r["state_id"].rsplit(":", 1)
        if arm == "T":
            by_state[sid + ":frontier"].append(r)
    # ceiling states from today's run: every continuation finished (or >= 3 OK) and none solved
    ceilings = []
    for fid, m in selected.items():
        runs = by_state[fid]
        ok = [r for r in runs if r.get("status") == "ok"]
        if (len(runs) >= args.n_cap or len(ok) >= 3) and ok and not any(r.get("outcome") == "solved" for r in ok):
            ceilings.append((fid, f"0/{len(ok)}", [first_action(r, m["action_kind"]) for r in ok]))
    prev_T = {r["kept_state_id"]: r for r in prev if r.get("arm") == "T"}
    yesterday = []
    for fid, k in kept.items():
        pr = prev_T.get(fid)
        if fid in selected or k.get("kept_class") != "contested" or not pr:
            continue
        if pr.get("status") == "ok" and pr.get("outcome") != "solved":
            yesterday.append((fid, "0/1", [first_action(pr, k["action_kind"])]))
    prev_F1 = {r["kept_state_id"]: r for r in prev if r.get("arm") == "F1" and r.get("status") == "ok"}
    # yesterday's states whose greedy F1 already ran come first (one arm banked)
    yesterday.sort(key=lambda t: t[0] not in prev_F1)
    print(f"ceiling states today: {len(ceilings)}; yesterday's contested T-failed (not selected): {len(yesterday)}")
    targets = (ceilings + yesterday)[: args.max_states]
    engy = Engy(concurrency=8, timeout=1200)
    arm_dir = args.out / "F1"
    for d in ("states", "seed_results", "seed_traces"):
        (arm_dir / d).mkdir(parents=True, exist_ok=True)
    metas, prop_rows = [], []
    # a second wave keeps the first wave's proposals (and arm files) as they are
    existing = {p["state_id"]: p for p in read_jsonl(args.proposals)} if args.proposals.exists() else {}
    for fid, t_desc, t_firsts in targets:
        row = samples.get(fid)
        if not row:
            continue
        if fid in existing:
            prop_rows.append(existing[fid])
            for q in existing[fid]["proposals"]:
                sid = fid.replace(":frontier", f":{q['arm']}")
                path = arm_dir / "states" / (sid.replace(":", "_") + ".json")
                meta = {k: v for k, v in json.loads(path.read_text()).items() if k not in META_DROP}
                meta["path"] = str(path)
                meta["teacher_desc"] = t_desc
                metas.append(meta)
            continue
        st = json.loads(Path(row["path"]).read_text())
        harness, kind = st["harness"], st["action_kind"]
        teacher_actions = [p["y"] for p in row.get("teacher") or [] if p and p.get("y")] + [a for a in t_firsts if a]
        props: list[dict] = []
        cost = 0.0

        def add(p: dict, src: str) -> bool:
            if not usable(p):
                return False
            if any(norm_action(p["y"], kind) == norm_action(q["y"], kind) for q in props):
                return False
            props.append(dict(p, source=src))
            return True
        add(row.get("frontier_greedy"), "greedy")
        add(row.get("frontier_t08"), "t08")
        tries = 0
        while len(props) < 2 and tries < args.extra_tries and not args.no_sample:
            tries += 1
            extra = {"tools": st["tools"]} if harness == "bash" and st.get("tools") else {}
            try:
                rep = await engy.chat(FRONTIER, st["messages"], temperature=0.8, max_tokens=SAMPLE_MAX_TOKENS, **extra)
            except Exception as e:  # noqa: BLE001
                print("  extra sample failed:", repr(e)[:200])
                continue
            cost += float(rep.get("cost_usd") or 0)
            add(action_of(rep, harness, kind), "extra")
        for p in props:
            if p["source"] == "greedy" and "same_decision" in (row.get("judge") or {}):
                p["judge"] = row["judge"]
                continue
            if args.no_judge:
                continue
            closest = max(teacher_actions, key=lambda t: jaccard(p["y"], t)) if teacher_actions else ""
            j = await judge_pair(engy, st, {"frontier_greedy": p, "closest_teacher_action": closest})
            cost += float(j.pop("_cost", 0) or 0)
            p["judge"] = j
            p["closest_teacher_action"] = closest
        out_props = []
        for p in props:
            arm = ARM_OF[p["source"]]
            sid = fid.replace(":frontier", f":{arm}")
            s = dict(st)
            s["state_id"] = sid
            s["arm"] = arm
            s["probe"] = "split_states"
            s["forced_reply"] = forced_reply_of({"frontier_greedy": p, "harness": harness})
            s["forced_action"] = p["y"]
            path = arm_dir / "states" / (sid.replace(":", "_") + ".json")
            path.write_text(json.dumps(s, ensure_ascii=False))
            meta = {k: v for k, v in s.items() if k not in META_DROP}
            meta["path"] = str(path)
            meta["teacher_desc"] = t_desc
            metas.append(meta)
            if arm == "F1" and fid in prev_F1:
                (arm_dir / "seed_results" / (sid.replace(":", "_") + ".json")).write_text(json.dumps(prev_F1[fid]))
                for tp in args.traces.glob(f"*/out/*/traces/{sid.replace(':', '_')}.json"):
                    shutil.copyfile(tp, arm_dir / "seed_traces" / tp.name)
                    break
            out_props.append({"arm": arm, "source": p["source"], "y": p["y"], "judge": p.get("judge"),
                              "closest_teacher_action": p.get("closest_teacher_action", row.get("closest_teacher_action")),
                              "seeded": arm == "F1" and fid in prev_F1})
        prop_rows.append({"state_id": fid, "harness": harness, "source": row["source"], "orig_outcome": row["orig_outcome"],
                          "action_kind": kind, "depth": row["depth"], "teacher_desc": t_desc,
                          "teacher_first_actions": teacher_actions, "proposals": out_props, "cost_usd": cost})
        print(f"  {fid[:12]} {harness:18s} T {t_desc}  proposals {[(q['arm'], q['source'], (q.get('judge') or {}).get('relation'), q['seeded']) for q in out_props]}  ${cost:.3f}")
    write_jsonl(arm_dir / "states.jsonl", metas)
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.proposals, prop_rows)
    n_seed = len(list((arm_dir / "seed_results").glob("*.json")))
    print(f"{len(prop_rows)} states, {len(metas)} F1 arm files ({n_seed} seeded from yesterday → "
          f"{len(metas) - n_seed} new continuations), sampling+judge ${engy.cost_usd:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--prev", type=Path, required=True)
    ap.add_argument("--traces", type=Path, default=Path("/tmp/fa_outcome/collect"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_split/arms_split"))
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--max-states", type=int, default=16)
    ap.add_argument("--n-cap", type=int, default=4)
    ap.add_argument("--extra-tries", type=int, default=3)
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    asyncio.run(build(ap.parse_args()))


if __name__ == "__main__":
    main()
