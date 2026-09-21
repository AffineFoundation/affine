"""Stage 0 — pull the public trace chunks for the resumable coding/terminal
sources and index every TEACHER rollout in them (frontier-arbiter outcome
probe, 2026-09-20).

    python fetch_traces.py --since 2026-08-25 --out /tmp/fa_outcome

Writes <out>/chunks/*.jsonl.gz (verbatim copies of data.affine.io) and
<out>/teacher_index.jsonl: one row per teacher rollout on a resumable harness
(mini_swe_textbased / bash / terminus_2) with source, outcome, n_replies,
image, stop condition — the pool build_states.py samples from.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import REPO  # noqa: E402

LIVE_TREE = Path(os.environ.get("AFFINE_LIVE_TREE", "/tmp/box/affine"))
sys.path.insert(0, str(HERE))
from traceutil import rollout_outcome  # noqa: E402

DATA_BASE = "https://data.affine.io/"
SOURCES = {"swesmith", "multiswe", "scaleswe", "swerebench_v2", "swelego",
           "terminal_lego", "terminal_bench_2"}
CODING = {"swesmith", "multiswe", "scaleswe", "swerebench_v2", "swelego"}
RESUMABLE = {"mini_swe_textbased": "textbased", "bash": "bash", "terminus_2": "terminus"}
UA = {"User-Agent": "curl/8.5"}   # data.affine.io 403s python-urllib


def fetch(cli: httpx.Client, key: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    for attempt in range(4):
        try:
            r = cli.get(DATA_BASE + key)
            r.raise_for_status()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dst)
            return dst
        except httpx.HTTPError as e:  # noqa: PERF203
            if attempt == 3:
                raise
            print(f"retry {key}: {e}", file=sys.stderr)
    return dst


def n_sampled_replies(trace: dict) -> int:
    return sum(1 for nd in trace.get("nodes") or []
               if nd.get("sampled") and (nd.get("message") or {}).get("role") == "assistant")


def index_chunk(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f.read().split("\n")):
            if not line.strip():
                continue
            e = json.loads(line)
            pol = e.get("policy") or {}
            harness = pol.get("harness") or ""
            if not str(pol.get("id", "")).startswith("teacher_") or harness not in RESUMABLE:
                continue
            if e.get("source") not in SOURCES:
                continue
            tr = e["trace"]
            runtime = (tr.get("agent") or {}).get("runtime") or {}
            rows.append({
                "chunk": path.name, "line": i, "rollout_id": e["rollout_id"],
                "source": e["source"], "policy_id": pol["id"], "harness": harness,
                "resume_kind": RESUMABLE[harness], "model": pol.get("model"),
                "outcome": rollout_outcome(tr), "stop": tr.get("stop_condition"),
                "n_replies": n_sampled_replies(tr), "image": runtime.get("image"),
                "sid": (e.get("task") or {}).get("sid"),
                "rewards": {k: (v or {}).get("score") for k, v in (tr.get("rewards") or {}).items()},
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-08-25")
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_outcome"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    (args.out / "chunks").mkdir(parents=True, exist_ok=True)
    cli = httpx.Client(headers=UA, timeout=180)
    man_path = args.out / "traces_manifest.json"
    if not man_path.exists():
        man_path.write_bytes(cli.get(DATA_BASE + "traces/manifest.json").content)
    man = json.load(open(man_path))
    chunks = [ch for ch in man["chunks"]
              if set(ch["sources"]) & SOURCES and ch["created_at"] >= args.since]
    print(f"{len(chunks)} chunks, {sum(c['bytes'] for c in chunks) / 1e9:.2f} GB")
    paths: list[Path] = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, cli, ch["key"], args.out / "chunks" / ch["key"].split("/")[-1]): ch
                for ch in chunks}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            paths.append(fut.result())
            if n % 100 == 0:
                print(f"  fetched {n}/{len(chunks)}")
    rows: list[dict] = []
    with cf.ProcessPoolExecutor(max(1, os.cpu_count() or 2)) as ex:
        for part in ex.map(index_chunk, sorted(paths)):
            rows.extend(part)
    with open(args.out / "teacher_index.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    import collections
    c = collections.Counter((r["source"], r["harness"], r["outcome"]) for r in rows)
    for k, v in sorted(c.items()):
        print(f"{v:6d} {k}")
    print(f"{len(rows)} teacher rollouts indexed -> {args.out / 'teacher_index.jsonl'}")


if __name__ == "__main__":
    main()
