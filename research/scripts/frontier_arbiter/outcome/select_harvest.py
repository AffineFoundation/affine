"""Harvest N = 8 (2026-09-22) — stage 1: candidate DECISION STATES for the
verified-action table, sharded over the reachable datagen pods.

    python select_harvest.py --out /tmp/fa_harvest/arms --n 300 \
        --pods dg5:8,dg6:6,dg2:4,bf4:4

Candidates: FAILED-origin teacher trajectories (env grader primary reward
< 1) on a resumable harness; depth >= 10 for mini_swe_textbased / bash
(the split-rich band of the 2026-09-21 probe: 60 % of failed-origin states
at depth 10-15 split, 14 % at 6-9), depth >= 3 for terminus_2 (supply is
~25 rollouts); depth <= 15; one state per task; `terminal_lego` excluded
(its env errored on dg5 2026-09-20); tasks whose image failed to build /
pull in the 2026-09-20/21 runs excluded; inside a (harness, source) cell,
tasks whose image ran an OK continuation before come first. The
failed-origin states of the 2026-09-21 split probe (N = 4 teacher
continuations already banked) are seeded as Phase-A-complete units
(`seed_results`), whatever their depth.

Sharding: `--pods name:workers,...` — a state goes to the pod whose bucket
blake2b(state id) % sum(workers) falls in (work proportional to the pod's
container budget). Writes <out>/<pod>/T/{states.jsonl,states/,seed_results/}
and <out>/selected.jsonl (every state, `pod` stamped).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from build_states import ChunkCache, build_state  # noqa: E402
from common import read_jsonl, write_jsonl  # noqa: E402
from select_states import META_DROP  # noqa: E402

CODING = {"swesmith", "multiswe", "scaleswe", "swerebench_v2", "swelego"}
EXCLUDE_SOURCES = {"terminal_lego"}
MIN_DEPTH = {"mini_swe_textbased": 10, "bash": 10, "terminus_2": 3}
MAX_DEPTH = 15
RESULTS_DIR = HERE.parents[2] / "results" / "frontier_arbiter"
IMAGE_ERROR_MARKS = ("docker_build_failed", "pull access denied", "manifest unknown", "not found: manifest",
                     "no such image", "image not found")


def stem(state_id: str) -> str:
    return state_id.replace(":", "_")


def parse_pods(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        name, _, w = part.partition(":")
        out.append((name.strip(), int(w or 1)))
    return out


def pod_of(base_id: str, pods: list[tuple[str, int]]) -> str:
    """Weighted bucket of blake2b(base state id); the same function maps a
    state's later arms (F1 / TX / Z) to the pod that holds its T results."""
    total = sum(w for _, w in pods)
    h = hashlib.blake2b(base_id.encode("utf-8"), digest_size=8).digest()
    b = int.from_bytes(h, "big") % total
    for name, w in pods:
        if b < w:
            return name
        b -= w
    return pods[-1][0]


def image_history() -> tuple[set[str], set[str]]:
    """(images with an OK continuation, images whose every run errored on an
    image/build problem) from the 2026-09-20/21 runs."""
    states = {}
    for f in ("outcome/states.jsonl", "split_states/selected.jsonl"):
        for r in read_jsonl(RESULTS_DIR / f):
            states[r["state_id"].rsplit(":", 1)[0]] = r
    ok, bad = set(), set()
    for f in ("outcome/continuations.jsonl", "split_states/continuations.jsonl"):
        for r in read_jsonl(RESULTS_DIR / f):
            s = states.get(r["state_id"].rsplit(":", 1)[0])
            img = ((s or {}).get("task") or {}).get("image") or ((s or {}).get("runtime") or {}).get("image")
            if not img:
                continue
            if r.get("status") == "ok":
                ok.add(img)
            elif any(m in str(r.get("error") or "").lower() for m in IMAGE_ERROR_MARKS):
                bad.add(img)
    return ok, bad - ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", type=Path, default=Path("/tmp/fa_outcome/teacher_index.jsonl"))
    ap.add_argument("--chunks", type=Path, default=Path("/tmp/fa_outcome/chunks"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/fa_outcome/traces_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_harvest/arms"))
    ap.add_argument("--n", type=int, default=300, help="new states (seeded ones come on top)")
    ap.add_argument("--pods", default="dg5:8,dg6:6,dg2:4,bf4:4")
    ap.add_argument("--since", default="2026-09-05", help="chunk created_at floor (images pullable)")
    ap.add_argument("--multiswe-since", default="2026-08-25")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--split-selected", type=Path, default=RESULTS_DIR / "split_states" / "selected.jsonl")
    ap.add_argument("--split-results", type=Path, default=Path("/tmp/fa_split/collect/dg5/out/split/results"))
    ap.add_argument("--no-seed", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pods = parse_pods(args.pods)
    created = {ch["key"].split("/")[-1]: ch["created_at"] for ch in json.load(open(args.manifest))["chunks"]}
    ok_images, bad_images = image_history()
    print(f"image history: {len(ok_images)} seen working, {len(bad_images)} failed")
    # yesterday's failed-origin split-probe states are seeded, their tasks are not re-drawn
    seeded: list[dict] = []
    seeded_tasks: set[str] = set()
    if not args.no_seed:
        for m in read_jsonl(args.split_selected):
            if m.get("orig_outcome") != "failed":
                continue
            base = m["state_id"].rsplit(":", 1)[0]
            res = sorted(args.split_results.glob(stem(base) + "_T*.json"))
            if not res:
                continue
            seeded.append({"meta": m, "results": res})
            seeded_tasks.add(m["task"]["sid"])
    pool: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in (json.loads(l) for l in open(args.index)):
        if r["outcome"] != "failed" or r["source"] in EXCLUDE_SOURCES:
            continue
        floor = args.multiswe_since if r["source"] == "multiswe" else args.since
        if created.get(r["chunk"], "") < floor:
            continue
        if r["n_replies"] < MIN_DEPTH[r["harness"]] + 2 or r["sid"] in seeded_tasks:
            continue
        if r.get("image") in bad_images:
            continue
        pool[(r["harness"], r["source"])].append(r)
    for v in pool.values():
        rng.shuffle(v)
        # images seen working first (pop() takes from the end)
        v.sort(key=lambda r: r.get("image") in ok_images)
    print("pool:", {f"{h}/{s}": len(v) for (h, s), v in sorted(pool.items())})
    # quotas: terminus takes everything it has; the rest is split evenly over textbased / bash
    n_term = sum(len(v) for (h, _), v in pool.items() if h == "terminus_2")
    want = {"terminus_2": min(n_term, args.n // 6), "mini_swe_textbased": 0, "bash": 0}
    rest = args.n - want["terminus_2"]
    want["mini_swe_textbased"] = rest // 2
    want["bash"] = rest - rest // 2
    cache = ChunkCache(args.chunks)
    seen_tasks: set[str] = set(seeded_tasks)
    picked: dict[str, list[dict]] = {h: [] for h in want}
    counts: collections.Counter = collections.Counter()
    for h, n in want.items():
        srcs = sorted(s for (hh, s) in pool if hh == h)
        while len(picked[h]) < n and any(pool[(h, s)] for s in srcs):
            for s in srcs:
                if len(picked[h]) >= n or not pool[(h, s)]:
                    continue
                r = pool[(h, s)].pop()
                if r["sid"] in seen_tasks:
                    counts["dup_task"] += 1
                    continue
                env = cache.load(r["chunk"], r["line"])
                hi = min(MAX_DEPTH, r["n_replies"] - 2)
                turn_idx = rng.randint(MIN_DEPTH[h], hi)
                st = build_state(env, turn_idx)
                if st is None:
                    counts[f"unbuildable:{h}"] += 1
                    continue
                st["orig_outcome"] = r["outcome"]
                st["orig_rewards"] = r["rewards"]
                st["image_seen_ok"] = r.get("image") in ok_images
                seen_tasks.add(r["sid"])
                picked[h].append(st)
                counts[f"state:{h}/{s}"] += 1
    # interleave harnesses so each pod's parallel slots mix container types
    order: list[dict] = []
    for i in range(max(len(v) for v in picked.values())):
        for h in ("mini_swe_textbased", "bash", "terminus_2"):
            if i < len(picked[h]):
                order.append(picked[h][i])
    metas: list[dict] = []
    per_pod: collections.Counter = collections.Counter()
    for name, _ in pods:
        for d in ("states", "seed_results"):
            (args.out / name / "T" / d).mkdir(parents=True, exist_ok=True)
    for st in order:
        base = st["state_id"].rsplit(":", 1)[0]
        pod = pod_of(base, pods)
        sid = base + ":T"
        st["state_id"] = sid
        st["arm"] = "T"
        st["probe"] = "harvest_n8"
        p = args.out / pod / "T" / "states" / (stem(sid) + ".json")
        p.write_text(json.dumps(st, ensure_ascii=False))
        meta = {k: v for k, v in st.items() if k not in META_DROP}
        meta.update(path=str(p), pod=pod, seeded=False)
        metas.append(meta)
        per_pod[pod] += 1
    n_seed_res = 0
    for s in seeded:
        m = dict(s["meta"])
        base = m["state_id"].rsplit(":", 1)[0]
        pod = pod_of(base, pods)
        src = Path(m["path"])
        if not src.is_file():
            alt = Path("/tmp/fa_split/arms_split/T/states") / src.name
            src = alt if alt.is_file() else src
        st = json.loads(src.read_text())
        st["probe"] = "harvest_n8"
        st["seeded_from"] = "split_states"
        p = args.out / pod / "T" / "states" / (stem(base + ":T") + ".json")
        p.write_text(json.dumps(st, ensure_ascii=False))
        for rp in s["results"]:
            shutil.copyfile(rp, args.out / pod / "T" / "seed_results" / rp.name)
            n_seed_res += 1
        m.update(path=str(p), pod=pod, seeded=True, probe="harvest_n8")
        m.pop("tier", None)
        metas.append(m)
        per_pod[pod] += 1
    for name, _ in pods:
        write_jsonl(args.out / name / "T" / "states.jsonl", [m for m in metas if m["pod"] == name])
    write_jsonl(args.out / "selected.jsonl", metas)
    for k, v in sorted(counts.items()):
        print(f"{v:5d} {k}")
    print(f"{len(order)} new states + {len(seeded)} seeded ({n_seed_res} seed results); per pod {dict(per_pod)}")
    print("depth:", collections.Counter(m["depth"] for m in metas))
    print("image seen ok:", collections.Counter(m.get("image_seen_ok") for m in metas))
    print("harness x source:", collections.Counter((m["harness"], m["source"]) for m in metas))


if __name__ == "__main__":
    main()
