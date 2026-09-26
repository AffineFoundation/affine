"""Split-states probe (2026-09-21) — stage 1: pick the states that get N
independent teacher continuations (arm T, --continuations N) and write the
arm files for the pod driver, seeded with yesterday's arm-T results so those
count as one of the N.

    python select_split.py --states .../outcome/states.jsonl --kept .../outcome/kept.jsonl \
        --prev .../outcome/continuations.jsonl --traces /tmp/fa_outcome/collect \
        --out /tmp/fa_split/arms --n-failed 24 --n-solved 12

Selection: per harness (mini_swe_textbased / bash / terminus_2) n_failed/3
states whose ORIGINAL teacher trajectory failed and n_solved/3 whose original
solved; depth 3-15 (every candidate is); `terminal_lego` excluded (its env
errored on dg5 yesterday); states whose arm T errored yesterday excluded.
Inside a cell the order of preference is (0) kept states with an OK arm-T
result yesterday (image pulled, one continuation already banked), (1) other
kept states, (2) the remaining candidates — round-robin over sources inside
a tier, seeded shuffle. Failed-origin states come first in states.jsonl so
their s/N is known earliest (Y2 needs the 0/N ones).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl, write_jsonl  # noqa: E402

META_DROP = ("messages", "teacher_reply", "king_reply", "task_system_prompt", "task_prompt",
             "tools", "forced_reply")
HARNESSES = ("mini_swe_textbased", "bash", "terminus_2")
EXCLUDE_SOURCES = {"terminal_lego"}


def stem(state_id: str) -> str:
    return state_id.replace(":", "_")


def pick(cands: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Tier by tier, round-robin over sources."""
    out: list[dict] = []
    for tier in (0, 1, 2):
        by_src = collections.defaultdict(list)
        for c in cands:
            if c["tier"] == tier:
                by_src[c["source"]].append(c)
        for v in by_src.values():
            rng.shuffle(v)
        srcs = sorted(by_src)
        rng.shuffle(srcs)
        while len(out) < n and any(by_src.values()):
            for s in srcs:
                if len(out) >= n:
                    break
                if by_src[s]:
                    out.append(by_src[s].pop())
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--prev", type=Path, required=True, help="yesterday's continuations.jsonl")
    ap.add_argument("--traces", type=Path, default=Path("/tmp/fa_outcome/collect"),
                    help="dir with <pod>/out/<tag>/traces/<stem>.json from yesterday's collect")
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_split/arms"))
    ap.add_argument("--n-failed", type=int, default=24)
    ap.add_argument("--n-solved", type=int, default=12)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cands = [s for s in read_jsonl(args.states) if s.get("state_kind") == "frontier"]
    kept = {r["state_id"] for r in read_jsonl(args.kept)}
    prev_T: dict[str, dict] = {}
    for r in read_jsonl(args.prev):
        if r.get("arm") == "T" and r.get("continuation", 0) == 0:
            prev_T[r["kept_state_id"]] = r
    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for s in cands:
        if s["source"] in EXCLUDE_SOURCES or not 3 <= s["depth"] <= 15:
            continue
        pr = prev_T.get(s["state_id"])
        if pr and pr.get("status") != "ok":
            continue
        s["tier"] = 0 if pr else 1 if s["state_id"] in kept else 2
        s["prev_T"] = None if not pr else pr.get("outcome")
        cells[(s["harness"], s["orig_outcome"])].append(s)
    print("pool per cell:", {f"{h}/{o}": collections.Counter(c["tier"] for c in v)
                             for (h, o), v in sorted(cells.items())})
    selected: list[dict] = []
    for oc, n in (("failed", args.n_failed), ("solved", args.n_solved)):
        per = [n // len(HARNESSES)] * len(HARNESSES)
        for i in range(n - sum(per)):
            per[i] += 1
        picked = {h: pick(cells[(h, oc)], k, rng) for h, k in zip(HARNESSES, per)}
        # interleave harnesses so the driver's parallel slots mix container types
        for i in range(max(per)):
            for h in HARNESSES:
                if i < len(picked[h]):
                    selected.append(picked[h][i])
    arm = args.out / "T"
    for d in ("states", "seed_results", "seed_traces"):
        (arm / d).mkdir(parents=True, exist_ok=True)
    metas = []
    n_seed = n_tr = 0
    for s in selected:
        st = json.loads(Path(s["path"]).read_text())
        sid = st["state_id"].replace(":frontier", ":T")
        st["state_id"] = sid
        st["arm"] = "T"
        st["probe"] = "split_states"
        p = arm / "states" / (stem(sid) + ".json")
        p.write_text(json.dumps(st, ensure_ascii=False))
        meta = {k: v for k, v in st.items() if k not in META_DROP}
        meta["path"] = str(p)
        meta["tier"] = s["tier"]
        meta["prev_T"] = s["prev_T"]
        metas.append(meta)
        pr = prev_T.get(s["state_id"])
        if pr:
            (arm / "seed_results" / (stem(sid) + ".json")).write_text(json.dumps(pr))
            n_seed += 1
            for tp in args.traces.glob(f"*/out/*/traces/{stem(sid)}.json"):
                shutil.copyfile(tp, arm / "seed_traces" / tp.name)
                n_tr += 1
                break
    write_jsonl(arm / "states.jsonl", metas)
    write_jsonl(args.out / "selected.jsonl", metas)
    c = collections.Counter((m["harness"], m["orig_outcome"], m["tier"]) for m in metas)
    for k, v in sorted(c.items()):
        print(f"{v:3d} {k}")
    print(f"{len(metas)} states, {n_seed} seeded with yesterday's arm-T result ({n_tr} traces), "
          f"prev_T outcomes: {collections.Counter(m['prev_T'] for m in metas)}")
    print("sources:", collections.Counter(m["source"] for m in metas))


if __name__ == "__main__":
    main()
