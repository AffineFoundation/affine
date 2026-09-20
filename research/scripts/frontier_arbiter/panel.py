"""Frontier-arbiter panel probe (2026-09-20): does "closeness to the frontier
where the frontier disagrees with the teacher" track held-out BENCHMARK scores
across the modern king panel (kings 11-20 + genesis + teacher)?

Stages (all resumable; everything under research/results/frontier_arbiter/panel/):

  select   pick ~150 turns per stored duel record (30 per dialect where
           available) from each king's crowning duel (king = challenger) and one
           duel where it sat as king; materialize prefixes from the pinned
           public corpus; keep both sides' actions + the k=3 teacher refs.
           -> turns.jsonl.gz, meta.json
  sample   frontier samples per turn: glm-5.3 x2 @T=0.8, deepseek-v4.1-flash x1
           @T=0.8 (max_tokens 2048+768); genesis (qwen3.6-35b-a3b) x1 on a
           300-turn subsample (30 per king, from the crowning records).
           -> frontier/<model>.jsonl (one line per (turn_id, sample_idx))
  judge    semantic judge (glm-5.3-flash, T=0) on a 200-turn subsample: is the
           frontier action the same decision as each teacher ref?
           -> judge.jsonl
  analyze  disagreement tables, per-model stats, Spearman vs benchmark axes,
           within-crown paired reads, surface-vs-semantic calibration.
           -> report.txt, report.json

  cd /workspace && source .venv/bin/activate
  python research/scripts/frontier_arbiter/panel.py select
  python research/scripts/frontier_arbiter/panel.py sample [--budget 80] [--concurrency 24]
  python research/scripts/frontier_arbiter/panel.py judge
  python research/scripts/frontier_arbiter/panel.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.REPO / "research" / "results" / "frontier_arbiter" / "panel"
MATRIX_URL = "https://affine.io/api/v1/matrix"

# reign -> (digest12, crowning duel [king appears as challenger], king-side duel)
KINGS: dict[int, tuple[str, str, str]] = {
    11: ("0ce59769300c", "chal-00409", "chal-00410"),
    12: ("6c877fd2242d", "chal-00502", "chal-00503"),
    13: ("6d0ee567e33e", "chal-00517", "chal-00518"),
    14: ("0f4029fd59ed", "chal-00556", "chal-00568"),
    15: ("5b1679a0db66", "chal-00581", "chal-00582"),
    16: ("6ea86b6e1868", "chal-00588", "chal-00590"),
    17: ("e416aab8599d", "chal-00598", "chal-00599"),
    18: ("73dd5bbcf1f7", "chal-00602", "chal-00603"),
    19: ("f82deda2ffbd", "chal-00606", "chal-00607"),
    20: ("50e3c081ab1c", "chal-00614", "chal-00615"),
}
PREV_KING = {11: "93b1f299dfdb"}          # reign 10 (hidden on the matrix; no bench card)
GENESIS_DIGEST = "995ad96eacd9"
TEACHER_KEY = "teacher"
GENESIS_KEY = "genesis"

DIALECTS = ("bash", "tool_call", "text", "boxed", "terminus_json")
PER_DIALECT = 30
PER_RECORD = 150
MAX_PREFIX_CHARS = 160_000
GENESIS_PER_KING = 30
JUDGE_TURNS = 200

FRONTIER = "glm-5.3"
FRONTIER_N = 2
CROSS = "deepseek-v4.1-flash"
CROSS_N = 1
JUDGE_MODEL = "glm-5.3-flash"
SAMPLE_TEMP = 0.8

GROUP_OF_SOURCE = {
    **{s: "coding" for s in ("multiswe", "r2e_gym", "scaleswe", "swelego", "swerebench_v2", "swesmith")},
    **{s: "terminal" for s in ("affine_tmax", "terminal_bench_2", "terminal_lego")},
    **{s: "math" for s in ("affine_i3math", "affine_math")},
    **{s: "tool_use" for s in ("affine_agent", "affine_notool", "affine_tau2", "affine_tau2_synth",
                                "affine_when2call", "affine_wiki")},
    **{s: "nl2repo" for s in ("affine_nl2lib", "nl2repobench")},
}
AGENTIC_GROUPS = ("coding", "terminal")
DEEP_MIN_ASSISTANT = 8

BENCH_AGENTIC = ("swebench-verified", "terminal-bench-2", "tau2-airline", "tau2-retail", "tau2-telecom",
                 "bfcl-v3", "when2call")
BENCH_CHAT = ("mmlu-pro", "gpqa-diamond", "math500", "aime25", "ifeval", "ifbench", "humaneval",
              "livecodebench")
BUDGET_VARIANTS = ("@16k", "@8k", "@4h250")


def group_of(source: str | None) -> str:
    return GROUP_OF_SOURCE.get(source or "", "general")


def turns_path() -> Path:
    """turns.jsonl while `select` is appending; turns.jsonl.gz once it finished
    (145 MB of prefixes plain — kept gzipped so the results dir stays well under
    the GitHub 100 MB cap)."""
    plain = OUT / "turns.jsonl"
    return plain if plain.exists() else plain.with_suffix(".jsonl.gz")


def load_turns() -> list[dict]:
    return C.read_jsonl(turns_path())


# ------------------------------------------------------------------ select
def _side_action(row: dict | None) -> dict | None:
    if not C.not_forfeit(row):
        return None
    p = row["pairs"][0]
    return {"z": p.get("z_a") or "", "y": p.get("y_a") or ""}


def _pick_stratified(cands: dict[str, list[str]], rng: random.Random, per_dialect: int,
                     total: int) -> list[str]:
    """`per_dialect` per dialect where available; shortfalls refilled from the
    dialects that have spare turns, proportionally to their spare pool."""
    picked: list[str] = []
    spare: dict[str, list[str]] = {}
    for k, ids in cands.items():
        ids = list(ids)
        rng.shuffle(ids)
        picked += ids[:per_dialect]
        spare[k] = ids[per_dialect:]
    need = total - len(picked)
    while need > 0 and any(spare.values()):
        ks = [k for k, v in spare.items() if v]
        w = [len(spare[k]) for k in ks]
        k = rng.choices(ks, weights=w)[0]
        picked.append(spare[k].pop())
        need -= 1
    return picked


def cmd_select(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "turns.jsonl"
    if not dst.exists() and dst.with_suffix(".jsonl.gz").exists():
        with gzip.open(dst.with_suffix(".jsonl.gz"), "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
    done_recs = {r["record"] for r in C.read_jsonl(dst)}
    meta_p = OUT / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"records": {}}
    rng = random.Random(args.seed)
    jobs: list[tuple[str, int, str]] = []
    for reign, (digest, crown_rec, king_rec) in KINGS.items():
        jobs.append((crown_rec, reign, "crowning"))
        jobs.append((king_rec, reign, "as_king"))
    for rec, reign, role in jobs:
        if rec in done_recs and not args.force:
            print(f"{rec}: already selected")
            continue
        d = C.load_verdict(rec)
        req, v = d["request"], d["verdict"]
        refs = d["teacher_refs"]
        k_by = {r["turn_id"]: r for r in d["king_rows"]}
        c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
        sl = v["slice"]
        corpus = C.corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
        rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
        cands: dict[str, list[str]] = collections.defaultdict(list)
        n_seen = n_long = 0
        for t in d["turn_ids"]:
            if t not in rows or len(refs.get(t) or []) < 3:
                continue
            if not (C.not_forfeit(k_by.get(t)) and C.not_forfeit(c_by.get(t))):
                continue
            if not all(r.get("y") for r in refs[t]):
                continue
            n_seen += 1
            if rows[t]["n_prefix_chars"] > MAX_PREFIX_CHARS:
                n_long += 1
                continue
            cands[rows[t].get("action_kind", "bash")].append(t)
        picked = _pick_stratified(cands, rng, PER_DIALECT, PER_RECORD)
        mats = corpus.materialize_turns([rows[t] for t in picked])
        king_digest = req["king_revision"][:12]
        chal_digest = req["challenger_revision"][:12]
        recs = []
        for t, m in zip(picked, mats):
            prefix = m["prefix"]
            recs.append({
                "record": rec, "reign": reign, "role": role, "turn_id": t,
                "kind": m.get("action_kind", "bash"), "source": m.get("source"),
                "group": group_of(m.get("source")), "phase": m.get("phase"),
                "depth": sum(1 for x in prefix if x["role"] == "assistant"),
                "n_prefix_chars": rows[t]["n_prefix_chars"],
                "prefix": prefix,
                "refs": [{"z": r.get("z") or "", "y": r["y"]} for r in refs[t]],
                "sides": {"king": {"model": king_digest, **_side_action(k_by[t])},
                          "challenger": {"model": chal_digest, **_side_action(c_by[t])}},
            })
        for r in recs:
            C.append_jsonl(dst, r)
        meta["records"][rec] = {
            "reign": reign, "role": role, "king": king_digest, "challenger": chal_digest,
            "n_turns": len(recs), "n_candidates": n_seen, "n_dropped_long": n_long,
            "by_kind": dict(collections.Counter(r["kind"] for r in recs)),
            "manifest_sha256": sl["manifest_sha256"], "margin": v.get("margin"), "z": v.get("z"),
            "score_mode": v["duel_params"].get("score_mode"), "challenger_wins": v.get("challenger_wins"),
        }
        meta_p.write_text(json.dumps(meta, indent=1))
        print(f"{rec} reign {reign} {role}: {len(recs)} turns {meta['records'][rec]['by_kind']} "
              f"(cands {n_seen}, dropped long {n_long}) king={king_digest} chal={chal_digest}")
    # genesis subsample: 30 turns per king from its crowning record
    turns = C.read_jsonl(dst)
    grng = random.Random(args.seed + 1)
    gen: list[str] = []
    for reign, (_, crown_rec, _) in KINGS.items():
        pool = [r["turn_id"] for r in turns if r["record"] == crown_rec]
        grng.shuffle(pool)
        gen += pool[:GENESIS_PER_KING]
    meta["genesis_turns"] = gen
    jrng = random.Random(args.seed + 2)
    pool = [r["turn_id"] for r in turns]
    jrng.shuffle(pool)
    meta["judge_turns"] = pool[:JUDGE_TURNS]
    meta_p.write_text(json.dumps(meta, indent=1))
    with open(dst, "rb") as fi, gzip.open(dst.with_suffix(".jsonl.gz"), "wb", compresslevel=6) as fo:
        fo.write(fi.read())
    dst.unlink()
    chars = sum(r["n_prefix_chars"] for r in turns)
    print(f"total {len(turns)} turns, {chars / 1e6:.1f}M prefix chars (~{chars / 4e6:.1f}M tokens per call); "
          f"genesis subsample {len(gen)}, judge subsample {len(meta['judge_turns'])}")
    return 0


# ------------------------------------------------------------------ sample
def frontier_path(model: str) -> Path:
    return OUT / "frontier" / f"{model}.jsonl"


def load_samples(model: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in C.read_jsonl(frontier_path(model)):
        if not r.get("error"):
            out[r["turn_id"]].append(r)
    return out


async def cmd_sample(args) -> int:
    turns = {r["turn_id"]: r for r in load_turns()}
    meta = json.loads((OUT / "meta.json").read_text())
    if not turns:
        raise SystemExit("run select first")
    plan: list[tuple[str, int, list[str]]] = [
        (FRONTIER, FRONTIER_N, list(turns)),
        (CROSS, CROSS_N, list(turns)),
        (C.GENESIS_ENGY, 1, meta.get("genesis_turns", [])),
    ]
    if args.models:
        plan = [p for p in plan if p[0] in args.models]
    engy = C.Engy(concurrency=args.concurrency)
    spent_before = sum(json.loads((OUT / "frontier" / "cost.json").read_text()).values()) \
        if (OUT / "frontier" / "cost.json").exists() else 0.0
    lock = asyncio.Lock()
    t0 = time.time()
    stop = False

    async def one(model: str, t: dict, idx: int, dst: Path, prog: dict) -> None:
        nonlocal stop
        if stop:
            return
        try:
            reply = await engy.chat(model, t["prefix"], SAMPLE_TEMP, C.DUEL_MAX_TOKENS)
            ro = C.reply_to_rollout(reply, t["kind"])
            rec = {"turn_id": t["turn_id"], "idx": idx, "model": model, "kind": t["kind"], **ro,
                   "usage": reply.get("usage"), "error": None}
        except Exception as e:  # noqa: BLE001 — one bad turn must not kill the run
            rec = {"turn_id": t["turn_id"], "idx": idx, "model": model, "kind": t["kind"],
                   "error": repr(e)[:300]}
        async with lock:
            C.append_jsonl(dst, rec)
            prog["n"] += 1
            total = spent_before + engy.cost_usd
            if prog["n"] % 25 == 0 or prog["n"] == prog["todo"]:
                print(f"  {model}: {prog['n']}/{prog['todo']} ({time.time() - t0:.0f}s) "
                      f"run ${engy.cost_usd:.2f} total ${total:.2f}", flush=True)
            if total > args.budget:
                if not stop:
                    print(f"BUDGET CAP ${args.budget} reached (total ${total:.2f}); stopping", flush=True)
                stop = True

    for model, n, tids in plan:
        dst = frontier_path(model)
        have = collections.Counter(r["turn_id"] for r in C.read_jsonl(dst) if not r.get("error"))
        todo = [(turns[t], i) for t in tids for i in range(n) if i >= have.get(t, 0)]
        rng = random.Random(0)
        rng.shuffle(todo)
        todo.sort(key=lambda x: x[1])      # first sample of every turn before any second sample
        if args.limit:
            todo = todo[: args.limit]
        prog = {"n": 0, "todo": len(todo)}
        print(f"{model}: {len(todo)} calls to make ({sum(have.values())} done)", flush=True)
        await asyncio.gather(*[one(model, t, i, dst, prog) for t, i in todo])
        if stop:
            break
    cost_p = OUT / "frontier" / "cost.json"
    prev = json.loads(cost_p.read_text()) if cost_p.exists() else {}
    for m, u in engy.usage.items():
        prev[m] = prev.get(m, 0.0) + u["cost_usd"]
    cost_p.write_text(json.dumps(prev, indent=1))
    usage_p = OUT / "frontier" / "usage.jsonl"
    C.append_jsonl(usage_p, {"at": time.time(), "usage": engy.usage})
    print(f"done in {time.time() - t0:.0f}s; this run ${engy.cost_usd:.2f}; cumulative ${sum(prev.values()):.2f}; "
          f"usage={json.dumps(engy.usage)}")
    return 0


# ------------------------------------------------------------------ judge
JUDGE_SYS = ("You compare two candidate next actions taken by an AI agent at the same point of a task. "
             "Decide whether they are THE SAME DECISION: an agent taking either one would be pursuing the "
             "same next step — the same operation on the same target, or the same final answer / conclusion. "
             "Cosmetic differences (flags, quoting, wording, formatting, ordering, extra harmless output, "
             "slightly different scope of the same inspection) do NOT matter. Different targets, different "
             "operations, or different final answers ARE different decisions.\n"
             "Examples: `ls -la /app` vs `ls /app` -> SAME. `cat src/a.py` vs `cat src/b.py` -> DIFFERENT. "
             "`pytest tests/test_x.py -q` vs `python -m pytest tests/test_x.py` -> SAME. "
             "Editing file F to fix bug X vs running the tests -> DIFFERENT. Two prose replies that reach the "
             "same conclusion -> SAME; different conclusions -> DIFFERENT. \\boxed{42} vs \\boxed{41} -> DIFFERENT.\n"
             "Think briefly, then finish with exactly one word on its own line: SAME or DIFFERENT.")
JUDGE_MAX_TOKENS = 1500


def parse_verdict(reply: dict) -> bool | None:
    for text in (reply.get("content") or "", reply.get("reasoning") or ""):
        hits = re.findall(r"\b(SAME|DIFFERENT)\b", text.upper())
        if hits:
            return hits[-1] == "SAME"
    return None


def _ctx(t: dict) -> str:
    last_user = next((m["content"] for m in reversed(t["prefix"]) if m["role"] == "user"), "")
    return last_user[-1500:]


async def cmd_judge(args) -> int:
    turns = {r["turn_id"]: r for r in load_turns()}
    meta = json.loads((OUT / "meta.json").read_text())
    fr = load_samples(FRONTIER)
    dst = OUT / "judge.jsonl"
    done = {(r["turn_id"], r["f_idx"], r["ref_idx"]) for r in C.read_jsonl(dst) if not r.get("error")}
    engy = C.Engy(concurrency=args.concurrency)
    lock = asyncio.Lock()
    jobs = []
    for tid in meta["judge_turns"]:
        t = turns[tid]
        fs = [s for s in fr.get(tid, []) if s["parsed"]]
        if not fs:
            continue
        f = fs[0]
        for j, ref in enumerate(t["refs"]):
            if (tid, f["idx"], j) not in done:
                jobs.append((t, f, j, ref))
    print(f"judge: {len(jobs)} pairs to judge ({len(done)} done)")

    async def one(t, f, j, ref):
        msgs = [{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content":
                 f"Dialect: {t['kind']}\n\nMost recent observation (truncated):\n{_ctx(t)}\n\n"
                 f"ACTION A:\n{f['y'][:3000]}\n\nACTION B:\n{ref['y'][:3000]}\n\nSAME or DIFFERENT?"}]
        try:
            r = await engy.chat(JUDGE_MODEL, msgs, 0.0, JUDGE_MAX_TOKENS)
            same = parse_verdict(r)
            rec = {"turn_id": t["turn_id"], "f_idx": f["idx"], "ref_idx": j, "same": same,
                   "raw": (r["content"] or "")[-60:], "finish": r["finish"],
                   "error": None if same is not None else "no verdict"}
        except Exception as e:  # noqa: BLE001
            rec = {"turn_id": t["turn_id"], "f_idx": f["idx"], "ref_idx": j, "error": repr(e)[:200]}
        async with lock:
            C.append_jsonl(dst, rec)

    await asyncio.gather(*[one(*j) for j in jobs])
    cost_p = OUT / "frontier" / "cost.json"
    prev = json.loads(cost_p.read_text()) if cost_p.exists() else {}
    prev["judge:" + JUDGE_MODEL] = prev.get("judge:" + JUDGE_MODEL, 0.0) + engy.cost_usd
    cost_p.write_text(json.dumps(prev, indent=1))
    print(f"judge done: ${engy.cost_usd:.3f}; usage={json.dumps(engy.usage)}")
    return 0


# ------------------------------------------------------------------ stats helpers
def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def permutation_p(a: list[float], b: list[float], observed: float, trials: int = 20000) -> float:
    rng = random.Random(0)
    sh = list(b)
    hits = 0
    for _ in range(trials):
        rng.shuffle(sh)
        if abs(spearman(a, sh)) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (trials + 1)


def boot_spearman_ci(a: list[float], b: list[float], n_boot: int = 2000) -> tuple[float, float]:
    rng = random.Random(1)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        x = [a[i] for i in idx]
        y = [b[i] for i in idx]
        r = spearman(x, y)
        if not math.isnan(r):
            vals.append(r)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def mean_se(v: list[float]) -> tuple[float, float]:
    if not v:
        return float("nan"), float("nan")
    if len(v) < 2:
        return v[0], float("nan")
    return st.mean(v), st.stdev(v) / math.sqrt(len(v))


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# ------------------------------------------------------------------ benchmark cards
def bench_axes() -> tuple[dict[str, dict[str, float]], dict]:
    """key (digest12 | 'teacher' | 'genesis') -> {total, agentic, chat, cells}."""
    cache = OUT / "matrix.json"
    if cache.exists():
        m = json.loads(cache.read_text())
    else:
        m = httpx.get(MATRIX_URL, timeout=60).json()
        cache.write_text(json.dumps(m))
    bench_keys = sorted({c["key"][6:] for c in m["columns"] if c["kind"] == "bench"
                         and not any(c["key"].endswith(v) for v in BUDGET_VARIANTS)})
    out: dict[str, dict] = {}
    for row in m["rows"]:
        key = {"teacher": TEACHER_KEY, "genesis": GENESIS_KEY}.get(row["kind"], row.get("digest12"))
        if row["kind"] == "reference":
            key = "ref:" + row["label"]
        cells = row.get("cells") or {}
        vals: dict[str, float] = {}
        for b in bench_keys:
            c = cells.get("bench:" + b)
            if not (isinstance(c, dict) and c.get("score") is not None):
                for v in BUDGET_VARIANTS:
                    c = cells.get(f"bench:{b}{v}")
                    if isinstance(c, dict) and c.get("score") is not None:
                        break
            if isinstance(c, dict) and c.get("score") is not None:
                vals[b] = float(c["score"])
        out[key] = {"label": row["label"], "cells": vals}
    return out, {"bench_keys": bench_keys, "generated_at": m.get("generated_at")}


AXES = ("total", "agentic", "agentic_no_tau2", "chat")


def axis_means(cards: dict[str, dict], keys: list[str]) -> dict[str, dict]:
    """Per-row axis scores. Cards have different cell sets (king17 has 10 cells,
    king20 lacks swebench/terminal-bench), so a raw mean over available cells
    is not comparable across rows. Axis value = mean over the row's available
    axis cells of (score − panel mean of that cell) ("centered"); the raw mean
    over available cells is kept as `<axis>_raw`, and `<axis>_n` = cells used.
    Cells constant across the panel (gaia2-ambiguity = 0 everywhere) are
    dropped from `total`."""
    all_names = sorted({n for k in keys for n in cards[k]["cells"]})
    cell_mean: dict[str, float] = {}
    for n in all_names:
        xs = [cards[k]["cells"][n] for k in keys if n in cards[k]["cells"]]
        if len(xs) >= 3 and max(xs) - min(xs) > 1e-9:
            cell_mean[n] = st.mean(xs)
    usable = [n for n in all_names if n in cell_mean]
    axes = {"total": usable,
            "agentic": [n for n in BENCH_AGENTIC if n in usable],
            "agentic_no_tau2": [n for n in BENCH_AGENTIC if n in usable and not n.startswith("tau2")],
            "chat": [n for n in BENCH_CHAT if n in usable]}
    res: dict[str, dict] = {}
    for k in keys:
        row: dict = {}
        for ax, names in axes.items():
            xs = [(n, cards[k]["cells"][n]) for n in names if n in cards[k]["cells"]]
            row[ax] = st.mean(v - cell_mean[n] for n, v in xs) if xs else None
            row[ax + "_raw"] = st.mean(v for _, v in xs) if xs else None
            row[ax + "_n"] = len(xs)
        res[k] = row
    res["_axes"] = axes
    res["_cell_mean"] = cell_mean
    return res


# ------------------------------------------------------------------ analyze
def turn_metrics(t: dict, fr: list[dict], cr: list[dict], judge: dict | None) -> dict:
    """Per-turn frontier/teacher comparisons. F = parsed glm-5.3 samples,
    D = parsed deepseek sample(s)."""
    kind = t["kind"]
    refs_y = [r["y"] for r in t["refs"]]
    yF = [s["y"] for s in fr if s["parsed"]]
    yD = [s["y"] for s in cr if s["parsed"]]
    m: dict = {"turn_id": t["turn_id"], "kind": kind, "group": t["group"], "depth": t["depth"],
               "reign": t["reign"], "record": t["record"], "role": t["role"],
               "nF": len(yF), "nD": len(yD)}
    if not yF:
        return m
    # F vs teacher refs (per F sample, then averaged)
    ex = [C.exact(y, refs_y, kind) for y in yF]
    j = [C.agree(y, refs_y) for y in yF]
    m["F_exact_any"] = st.mean(ex)                 # frac of F samples exactly matching >=1 ref
    m["F_jac"] = st.mean(j)                        # mean over F samples of best Jaccard vs refs
    m["F_j30"] = st.mean(x >= 0.3 for x in j)
    m["F_j50"] = st.mean(x >= 0.5 for x in j)
    m["contested_exact"] = not any(ex)             # every F sample disagrees with all 3 refs
    m["contested_j30"] = all(x < 0.3 for x in j)
    m["contested_j50"] = all(x < 0.5 for x in j)
    m["unanimous_agree"] = all(ex)                 # every F sample matches >=1 ref exactly
    # frontier self / cross agreement
    if len(yF) >= 2:
        m["F_self_exact"] = float(C.exact(yF[0], yF[1:], kind))
        m["F_self_jac"] = C.jaccard(yF[0], yF[1])
    if yD:
        m["FD_exact"] = st.mean(C.exact(y, yD, kind) for y in yF)
        m["FD_jac"] = st.mean(C.agree(y, yD) for y in yF)
        m["D_exact_any"] = st.mean(C.exact(y, refs_y, kind) for y in yD)
        m["D_jac"] = st.mean(C.agree(y, refs_y) for y in yD)
    # teacher self-agreement (ref vs the other refs)
    m["T_self_exact"] = st.mean(C.exact(r, refs_y[:i] + refs_y[i + 1:], kind) for i, r in enumerate(refs_y))
    m["T_self_jac"] = st.mean(C.agree(r, refs_y[:i] + refs_y[i + 1:]) for i, r in enumerate(refs_y))
    # refs' closeness to F (baseline for excess_F and for the proposed rule)
    refs_aF = [C.agree(r, yF) for r in refs_y]
    m["refs_aF"] = st.mean(refs_aF)
    m["refs_aF_max"] = max(refs_aF)
    if yD:
        m["refs_aD"] = st.mean(C.agree(r, yD) for r in refs_y)
    if judge is not None:
        m["judge_same_any"] = judge.get("same_any")
    # per-side stats
    sides = {}
    for side, s in t["sides"].items():
        y = s.get("y") or ""
        sides[s["model"]] = _model_turn_stats(y, refs_y, yF, yD, kind, m, side)
    # teacher LOO row: ref i as "the model", other two as the teacher
    loo = []
    for i, r in enumerate(refs_y):
        others = refs_y[:i] + refs_y[i + 1:]
        loo.append(_model_turn_stats(r, others, yF, yD, kind, m, "teacher", refs_aF_base=st.mean(
            C.agree(o, yF) for o in others)))
    sides[TEACHER_KEY] = {k: st.mean(x[k] for x in loo) if isinstance(loo[0][k], (int, float, bool)) else loo[0][k]
                          for k in loo[0]}
    m["sides"] = sides
    return m


def _model_turn_stats(y: str, refs_y: list[str], yF: list[str], yD: list[str], kind: str, m: dict,
                      side: str, refs_aF_base: float | None = None) -> dict:
    base = m["refs_aF"] if refs_aF_base is None else refs_aF_base
    aT = C.agree(y, refs_y)
    aF = C.agree(y, yF)
    d = {"side": side, "agree_T": aT, "agree_F": aF, "excess_F": aF - base,
         "excess_T": aT - m["T_self_jac"],
         "sides_with_F": float(aF - aT > 0.2), "gap_F": aF - base,
         "beats_T_toward_F": float(aF > base), "exact_T": float(C.exact(y, refs_y, kind)),
         "exact_F": float(C.exact(y, yF, kind))}
    if yD:
        aD = C.agree(y, yD)
        d["agree_D"] = aD
        d["excess_D"] = aD - m["refs_aD"]
    return d


def _fmt(x, w=8, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (w - 1) + "-"
    return f"{x:{w}.{p}f}"


def cmd_analyze(args) -> int:
    turns = load_turns()
    meta = json.loads((OUT / "meta.json").read_text())
    fr = load_samples(FRONTIER)
    cr = load_samples(CROSS)
    gen = load_samples(C.GENESIS_ENGY)
    judge_rows = [r for r in C.read_jsonl(OUT / "judge.jsonl") if not r.get("error")]
    judge: dict[str, dict] = collections.defaultdict(lambda: {"same": {}})
    for r in judge_rows:
        judge[r["turn_id"]]["same"][r["ref_idx"]] = r["same"]
    for tid, j in judge.items():
        j["same_any"] = any(j["same"].values()) if len(j["same"]) == 3 else None
    cost = json.loads((OUT / "frontier" / "cost.json").read_text()) if (OUT / "frontier" / "cost.json").exists() else {}

    # genesis: add as a side on its subsample turns
    turns_by_id = {t["turn_id"]: t for t in turns}
    for tid, ss in gen.items():
        s = next((x for x in ss if x["parsed"]), None)
        if tid in turns_by_id and s:
            turns_by_id[tid]["sides"]["genesis"] = {"model": GENESIS_KEY, "z": s["z"], "y": s["y"]}

    mets = [turn_metrics(t, fr.get(t["turn_id"], []), cr.get(t["turn_id"], []), judge.get(t["turn_id"]))
            for t in turns]
    mets = [m for m in mets if "sides" in m]
    L: list[str] = []
    P = L.append
    report: dict = {"n_turns_selected": len(turns), "n_turns_with_frontier": len(mets), "cost_usd": cost,
                    "cost_total_usd": sum(cost.values())}
    P("Frontier-arbiter panel probe — kings 11–20 + genesis + teacher (2026-09-20)")
    P(f"turns selected {len(turns)} from {len(meta['records'])} stored duels; with >=1 parsed {FRONTIER} sample: {len(mets)}")
    P(f"frontier F = {FRONTIER} x{FRONTIER_N} @T={SAMPLE_TEMP}; cross D = {CROSS} x{CROSS_N}; genesis = {C.GENESIS_ENGY} x1 "
      f"on {len(gen)} turns; judge = {JUDGE_MODEL} T=0 on {len({r['turn_id'] for r in judge_rows})} turns")
    P(f"Engy spend: {json.dumps({k: round(v, 2) for k, v in cost.items()})}  total ${sum(cost.values()):.2f}")
    parsed_rate = {FRONTIER: st.mean(s["parsed"] for ss in fr.values() for s in ss) if fr else None,
                   CROSS: st.mean(s["parsed"] for ss in cr.values() for s in ss) if cr else None,
                   C.GENESIS_ENGY: st.mean(s["parsed"] for ss in gen.values() for s in ss) if gen else None}
    P(f"parse rate (action found in the turn's dialect): {json.dumps({k: round(v, 3) for k, v in parsed_rate.items() if v is not None})}")
    report["parse_rate"] = parsed_rate

    # ---------------- definitions
    P("")
    P("DEFINITIONS (all similarities are dialect-aware: norm_action for exact, token-Jaccard of the action body otherwise)")
    P("  F_exact_any      frac of F samples whose normalised action equals >=1 of the 3 teacher refs")
    P("  F_j30 / F_j50    frac of F samples with best Jaccard vs refs >= 0.3 / 0.5")
    P("  contested_X      turn where EVERY F sample disagrees with ALL 3 refs under X (exact / j30 / j50)")
    P("  unanimous_agree  every F sample exactly matches >=1 ref")
    P("  T_self / F_self  teacher ref vs other refs / F sample 1 vs F sample 2 (same-model resample agreement)")
    P("  FD               F samples vs the deepseek sample (cross-frontier agreement)")
    P("  agree_T          per model: mean over its turns of best Jaccard(model action, 3 teacher refs)")
    P("  agree_F          per model: mean best Jaccard(model action, F samples)")
    P("  excess_F         agree_F(model) − mean_i best-Jaccard(ref_i, F) on the same turns  (turn-difficulty controlled)")
    P("  excess_T         agree_T(model) − mean_i best-Jaccard(ref_i, other refs): the frontier-free twin of excess_F (is the model as teacher-like as a teacher resample?)")
    P("  sides_with_F     frac of turns with agree_F − agree_T > 0.2 (model visibly nearer F than the teacher)")
    P("  beats_T→F        PROPOSED RULE STAT: on contested turns, frac where Jaccard(model, F) > mean_i Jaccard(ref_i, F)")
    P("  gap_F            on contested turns, mean signed Jaccard(model, F) − mean_i Jaccard(ref_i, F)")
    P("  teacher row      leave-one-out: ref_i is 'the model', the other two refs are 'the teacher' (k=2, so agree_T is biased low)")
    P("  bench axes       per cell: score − panel mean of that cell (cards have different cell sets); axis = mean over the row's available axis cells;")
    P("                   total = every non-constant bench cell; agentic = " + ", ".join(BENCH_AGENTIC)
      + "; agentic_no_tau2 = the same minus tau2-*; chat = " + ", ".join(BENCH_CHAT) + " (plain cap, else @budget variant)")

    # ---------------- disagreement tables
    def dis_block(title: str, sub: list[dict]) -> dict:
        keys = ("F_exact_any", "F_j30", "F_j50", "contested_exact", "contested_j30", "contested_j50",
                "unanimous_agree", "T_self_exact", "T_self_jac", "F_self_exact", "F_self_jac", "FD_exact", "FD_jac",
                "D_exact_any", "F_jac", "judge_same_any")
        row = {"n": len(sub)}
        for k in keys:
            xs = [float(m[k]) for m in sub if m.get(k) is not None]
            row[k] = st.mean(xs) if xs else None
            row[k + "_n"] = len(xs)
        P(f"{title:22} {row['n']:5d} " + " ".join(_fmt(row[k], 7) for k in keys[:15]) +
          (f" {_fmt(row['judge_same_any'], 7)}({row['judge_same_any_n']})" if row["judge_same_any"] is not None else ""))
        return row

    P("")
    P("===== DISAGREEMENT RATES: frontier vs the k=3 teacher refs (per turn, averaged) =====")
    hdr = ("F_exact", "F_j30", "F_j50", "cont_ex", "cont30", "cont50", "unanim", "Tself_x", "Tself_j",
           "Fself_x", "Fself_j", "FD_x", "FD_j", "D_exact", "F_jac")
    P(f"{'slice':22} {'n':>5} " + " ".join(f"{h:>7}" for h in hdr) + "  judge_same_any(n)")
    dis: dict = {"ALL": dis_block("ALL", mets)}
    for k in DIALECTS:
        sub = [m for m in mets if m["kind"] == k]
        if sub:
            dis["kind:" + k] = dis_block("kind " + k, sub)
    for g in sorted({m["group"] for m in mets}):
        sub = [m for m in mets if m["group"] == g]
        if sub:
            dis["group:" + g] = dis_block("group " + g, sub)
    depth_bins = [(0, 1, "depth 0-1"), (2, 4, "depth 2-4"), (5, 9, "depth 5-9"), (10, 19, "depth 10-19"), (20, 10**6, "depth 20+")]
    for lo, hi, name in depth_bins:
        sub = [m for m in mets if lo <= m["depth"] <= hi]
        if sub:
            dis[name] = dis_block(name, sub)
    for era, name in ((("min_rg",), "era min_rg (k11-15)"), (("sd_min_rga",), "era sd (k16-20)")):
        recs = {r for r, v in meta["records"].items() if v["score_mode"] in era}
        sub = [m for m in mets if m["record"] in recs]
        if sub:
            dis[name] = dis_block(name, sub)
    report["disagreement"] = dis

    # ---------------- semantic calibration
    P("")
    P("===== SEMANTIC JUDGE vs SURFACE METRICS (F sample 1 vs each ref; judge = glm-5.3-flash T=0 'same decision') =====")
    cal_rows = []
    for tid, j in judge.items():
        if j["same_any"] is None or tid not in turns_by_id:
            continue
        t = turns_by_id[tid]
        fs = [s for s in fr.get(tid, []) if s["parsed"]]
        if not fs:
            continue
        y = fs[0]["y"]
        for i, ref in enumerate(t["refs"]):
            if i in j["same"]:
                cal_rows.append({"kind": t["kind"], "sem": bool(j["same"][i]),
                                 "exact": C.exact(y, [ref["y"]], t["kind"]), "jac": C.jaccard(y, ref["y"])})
    cal: dict = {"n_pairs": len(cal_rows)}
    if cal_rows:
        sem = [r["sem"] for r in cal_rows]
        P(f"pairs judged {len(cal_rows)} ({len({tid for tid in judge if judge[tid]['same_any'] is not None})} turns); "
          f"semantic SAME rate {st.mean(sem):.3f}")
        P(f"{'surface rule':16} {'agree%':>7} {'kappa':>7} {'acc':>6} {'prec':>6} {'rec':>6}")
        best = None
        for name, fn in [("exact", lambda r: r["exact"])] + [(f"jac>={th:.2f}", (lambda th: lambda r: r["jac"] >= th)(th))
                                                              for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]:
            pred = [bool(fn(r)) for r in cal_rows]
            k = kappa(pred, sem)
            acc = st.mean(p == s for p, s in zip(pred, sem))
            tp = sum(p and s for p, s in zip(pred, sem))
            prec = tp / max(sum(pred), 1)
            rec = tp / max(sum(sem), 1)
            cal[name] = {"pred_rate": st.mean(pred), "kappa": k, "acc": acc, "prec": prec, "rec": rec}
            P(f"{name:16} {st.mean(pred):7.3f} {k:7.3f} {acc:6.3f} {prec:6.3f} {rec:6.3f}")
            if best is None or k > best[1]:
                best = (name, k)
        cal["best"] = {"rule": best[0], "kappa": best[1]}
        P(f"best surface proxy for semantic disagreement: {best[0]} (kappa {best[1]:.3f})")
        for kd in DIALECTS:
            sub = [r for r in cal_rows if r["kind"] == kd]
            if len(sub) >= 10:
                s2 = [r["sem"] for r in sub]
                P(f"  {kd:14} n={len(sub):4d} sem_same {st.mean(s2):.3f}  kappa exact {kappa([r['exact'] for r in sub], s2):.3f}"
                  f"  jac>=0.3 {kappa([r['jac'] >= 0.3 for r in sub], s2):.3f}  jac>=0.5 {kappa([r['jac'] >= 0.5 for r in sub], s2):.3f}")
                cal["kind:" + kd] = {"n": len(sub), "sem_same": st.mean(s2),
                                     "kappa_exact": kappa([r["exact"] for r in sub], s2),
                                     "kappa_j30": kappa([r["jac"] >= 0.3 for r in sub], s2),
                                     "kappa_j50": kappa([r["jac"] >= 0.5 for r in sub], s2)}
    else:
        P("no judge rows")
    report["calibration"] = cal

    # ---------------- per-model stats
    cards, cinfo = bench_axes()
    labels = {d: f"king{r}" for r, (d, _, _) in KINGS.items()}
    labels[TEACHER_KEY] = "teacher"
    labels[GENESIS_KEY] = "genesis"
    labels[PREV_KING[11]] = "king10"
    STATS = ("agree_T", "excess_T", "agree_F", "excess_F", "sides_with_F", "beats_T_toward_F", "gap_F", "exact_T", "exact_F",
             "agree_D", "excess_D")
    CONTESTED_STATS = ("beats_T_toward_F", "gap_F")
    contested_key = args.contested

    def model_table(sub: list[dict], title: str) -> dict:
        acc: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        for m in sub:
            for mk, s in m["sides"].items():
                for k in STATS:
                    if k not in s:
                        continue
                    if k in CONTESTED_STATS and not m.get(contested_key):
                        continue
                    acc[mk][k].append(float(s[k]))
                acc[mk]["_n"].append(1.0)
                acc[mk]["_contested"].append(float(bool(m.get(contested_key))))
        rows = {}
        for mk, d in acc.items():
            rows[mk] = {"n": len(d["_n"]), "n_contested": int(sum(d["_contested"]))}
            for k in STATS:
                mu, se = mean_se(d[k])
                rows[mk][k] = mu
                rows[mk][k + "_se"] = se
        P("")
        P(f"===== PER-MODEL STATS: {title} (contested = {contested_key}) =====")
        P(f"{'model':8} {'n':>5} {'nCont':>5} {'agree_T':>8} {'excess_T':>9} {'agree_F':>8} {'excess_F':>9} {'sides_F':>8} {'beats_T→F':>10} "
          f"{'gap_F':>8} {'exact_T':>8} {'exact_F':>8} {'agree_D':>8} {'excess_D':>9}  bench(centered): total agentic noTau2 chat")
        order = [TEACHER_KEY, GENESIS_KEY] + [KINGS[r][0] for r in sorted(KINGS, reverse=True)] + [PREV_KING[11]]
        for mk in order:
            if mk not in rows:
                continue
            r = rows[mk]
            b = axes.get(mk, {})
            P(f"{labels.get(mk, mk):8} {r['n']:5d} {r['n_contested']:5d} {_fmt(r['agree_T'])} {_fmt(r['excess_T'], 9, 4)} {_fmt(r['agree_F'])} "
              f"{_fmt(r['excess_F'], 9, 4)} {_fmt(r['sides_with_F'])} {_fmt(r['beats_T_toward_F'], 10)}±{_fmt(r['beats_T_toward_F_se'], 5, 3).strip()} "
              f"{_fmt(r['gap_F'], 8, 4)} {_fmt(r['exact_T'])} {_fmt(r['exact_F'])} {_fmt(r.get('agree_D'))} {_fmt(r.get('excess_D'), 9, 4)}"
              f"  {_fmt(b.get('total'), 6, 1)} {_fmt(b.get('agentic'), 7, 1)} {_fmt(b.get('agentic_no_tau2'), 6, 1)} {_fmt(b.get('chat'), 6, 1)}")
        return rows

    def align_table(rows: dict, title: str, only: list[str] | None = None) -> dict:
        keys = [k for k in rows if k in axes and axes[k].get("total") is not None and (only is None or k in only)]
        P(f"-- alignment ({title}): Spearman(stat, centered bench axis) over n={len(keys)} models "
          f"[{', '.join(labels.get(k, k) for k in keys)}]; perm p two-sided; 95% CI = bootstrap over models --")
        P(f"{'stat':18} " + " ".join(f"{ax:>26}" for ax in AXES))
        res = {}
        for k in STATS:
            line = f"{k:18} "
            res[k] = {}
            for ax in AXES:
                ks = [m for m in keys if axes[m].get(ax) is not None and rows[m].get(k) is not None
                      and not math.isnan(rows[m][k])]
                if len(ks) < 5:
                    line += f"{'-':>26} "
                    continue
                x = [rows[m][k] for m in ks]
                y = [axes[m][ax] for m in ks]
                rho = spearman(x, y)
                p = permutation_p(x, y, rho, 5000)
                lo, hi = boot_spearman_ci(x, y, 1000)
                res[k][ax] = {"rho": rho, "p": p, "ci": [lo, hi], "n": len(ks)}
                line += f"{rho:+.2f} p{p:.2f} [{lo:+.2f},{hi:+.2f}] "
            P(line)
        return res

    panel_keys = [TEACHER_KEY, GENESIS_KEY] + [KINGS[r][0] for r in KINGS]
    panel_keys = [k for k in panel_keys if k in cards]
    axes = axis_means(cards, panel_keys)
    report["bench_axes"] = {labels.get(k, k): v for k, v in axes.items() if not k.startswith("_")}
    report["bench_axis_cells"] = axes["_axes"]
    report["bench_cell_means"] = axes["_cell_mean"]
    P("")
    P("===== BENCHMARK AXES (matrix " + str(cinfo["generated_at"]) + ") =====")
    P("axis value = mean over the row's available axis cells of (score − panel mean of that cell); raw = plain mean of the same cells; n = cells used")
    P("kings 11–15 score 0.0 on tau2-* (pre-wvk-13 models do not survive the tool-calling harness) — real cells, kept; agentic_no_tau2 drops them")
    for ax, names in axes["_axes"].items():
        P(f"  {ax}: " + ",".join(names))
    P(f"  {'model':8} " + " ".join(f"{ax:>18}" for ax in AXES) + "  (centered / raw / n)")
    for k in panel_keys:
        P(f"  {labels.get(k, k):8} " + " ".join(
            f"{_fmt(axes[k][ax], 6, 1)}/{_fmt(axes[k][ax + '_raw'], 5, 1)}/{axes[k][ax + '_n']:2d}" for ax in AXES)
          + f"  cells {len(cards[k]['cells'])}")

    modern = [TEACHER_KEY, GENESIS_KEY] + [KINGS[r][0] for r in range(16, 21)]
    report["models"] = {}
    report["alignment"] = {}
    report["alignment_modern"] = {}
    for title, sub in (("ALL turns", mets),
                       ("agentic groups (coding/terminal)", [m for m in mets if m["group"] in AGENTIC_GROUPS]),
                       (f"deep turns (depth >= {DEEP_MIN_ASSISTANT})", [m for m in mets if m["depth"] >= DEEP_MIN_ASSISTANT]),
                       ("shell dialects (bash/tool_call/terminus_json)", [m for m in mets if m["kind"] in ("bash", "tool_call", "terminus_json")]),
                       ("text dialect", [m for m in mets if m["kind"] == "text"]),
                       ("boxed dialect", [m for m in mets if m["kind"] == "boxed"])):
        if len(sub) < 30:
            continue
        rows = model_table(sub, title)
        report["models"][title] = {labels.get(k, k): v for k, v in rows.items()}
        report["alignment"][title] = align_table(rows, title)
        if title == "ALL turns" or title.startswith("agentic"):
            report["alignment_modern"][title] = align_table(
                rows, title + " — sub-panel kings 16–20 + teacher + genesis (the post-wvk-13, tool-capable cluster)", modern)
    # the same with the alternative contested definition
    alt = "contested_j30" if contested_key == "contested_exact" else "contested_exact"
    args.contested, contested_key = alt, alt
    rows = model_table(mets, f"ALL turns, alternative contested = {alt}")
    report["models"]["ALL turns alt contested"] = {labels.get(k, k): v for k, v in rows.items()}
    report["alignment"]["ALL turns alt contested"] = align_table(rows, f"ALL, contested={alt}")
    args.contested = contested_key = "contested_exact"

    # ---------------- within-crown paired reads
    P("")
    P("===== WITHIN-CROWN PAIRED READS: challenger(new king) − king(old king) on the SAME crowning-duel turns vs bench change =====")
    P("(a positive Δstat with a positive Δbench = the stat moved with the benchmark; n=9 crowns with cards on both sides)")
    pairs = []
    for reign, (digest, crown_rec, _) in KINGS.items():
        sub = [m for m in mets if m["record"] == crown_rec]
        if not sub:
            continue
        old = meta["records"][crown_rec]["king"]
        if old not in axes or digest not in axes:
            continue
        row = {"reign": reign, "new": digest, "old": old, "n": len(sub)}
        for k in STATS:
            dn = [m["sides"][digest][k] for m in sub if k in m["sides"].get(digest, {})
                  and (k not in CONTESTED_STATS or m.get(contested_key))]
            do = [m["sides"][old][k] for m in sub if k in m["sides"].get(old, {})
                  and (k not in CONTESTED_STATS or m.get(contested_key))]
            row[k] = (st.mean(dn) - st.mean(do)) if dn and do else None
        for ax in AXES:
            a, b = axes[digest].get(ax), axes[old].get(ax)
            row["d_" + ax] = (a - b) if a is not None and b is not None else None
        pairs.append(row)
    P(f"{'crown':10} {'n':>4} {'Δagree_T':>9} {'Δagree_F':>9} {'Δexcess_F':>10} {'ΔsidesF':>8} {'Δbeats':>8} {'Δgap_F':>8}  {'Δtotal':>7} {'Δagent':>7} {'ΔnoTau2':>7} {'Δchat':>7}")
    for r in pairs:
        P(f"{r['reign'] - 1:>2}→{r['reign']:<7} {r['n']:4d} {_fmt(r['agree_T'], 9, 4)} {_fmt(r['agree_F'], 9, 4)} {_fmt(r['excess_F'], 10, 4)} "
          f"{_fmt(r['sides_with_F'], 8, 3)} {_fmt(r['beats_T_toward_F'], 8, 3)} {_fmt(r['gap_F'], 8, 4)}  "
          f"{_fmt(r['d_total'], 7, 1)} {_fmt(r['d_agentic'], 7, 1)} {_fmt(r['d_agentic_no_tau2'], 7, 1)} {_fmt(r['d_chat'], 7, 1)}")
    paired: dict = {"rows": pairs, "spearman": {}}
    for k in ("agree_T", "agree_F", "excess_F", "sides_with_F", "beats_T_toward_F", "gap_F"):
        line = f"  Δ{k:17}"
        paired["spearman"][k] = {}
        for ax in AXES:
            ps = [r for r in pairs if r.get(k) is not None and r.get("d_" + ax) is not None]
            if len(ps) < 5:
                line += f" {ax}: - "
                continue
            x = [r[k] for r in ps]
            y = [r["d_" + ax] for r in ps]
            rho = spearman(x, y)
            sign = sum((a > 0) == (b > 0) for a, b in zip(x, y))
            paired["spearman"][k][ax] = {"rho": rho, "sign_agree": sign, "n": len(ps)}
            line += f" {ax}: rho {rho:+.2f} sign {sign}/{len(ps)} "
        P(line)
    report["paired"] = paired

    # ---------------- hand-written reading of the numbers (kept next to the data, appended verbatim)
    verdict = OUT / "verdict.txt"
    if verdict.exists():
        P("")
        P("===== VERDICT (hand-written reading of the tables above) =====")
        L.extend(verdict.read_text().rstrip("\n").splitlines())

    # ---------------- turn-level dump for later reuse
    C.write_jsonl(OUT / "turn_metrics.jsonl", mets)
    text = "\n".join(L) + "\n"
    (OUT / "report.txt").write_text(text)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--seed", type=int, default=20260920)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("--budget", type=float, default=80.0, help="hard USD cap on cumulative Engy spend")
    s.add_argument("--concurrency", type=int, default=24)
    s.add_argument("--limit", type=int, default=0, help="max calls per model this run (pilot)")
    s.add_argument("--models", nargs="*", default=None)
    s = sub.add_parser("judge")
    s.add_argument("--concurrency", type=int, default=16)
    s = sub.add_parser("analyze")
    s.add_argument("--contested", default="contested_exact", choices=("contested_exact", "contested_j30", "contested_j50"))
    args = ap.parse_args()
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "sample":
        return asyncio.run(cmd_sample(args))
    if args.cmd == "judge":
        return asyncio.run(cmd_judge(args))
    return cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
