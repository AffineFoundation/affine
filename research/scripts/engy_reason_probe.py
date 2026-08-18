"""Engy-teacher Reason exploration: can a teacher the miners never optimized
against restore swe-bench isomorphism?

Motivation (RT-7): on the live board Spearman(Reason margin, swe_lite) = -0.42 —
miners Goodhart GLM-4.5-Air. This probe rescores *stored* thoughts with a
different, stronger teacher C' (engy `glm-5.2`, echo+logprobs supported) and
asks whether Reason_C' correlates positively with swe on:

  albedo : 10 kings x 200 shared turns (ekings_v2_all.jsonl, KING_BENCH swe)
  live   : published duel artifacts (evals/chal-*.json.gz) x benchmarks.json

Scoring mirrors production Reason v4 exactly:
  a_i  = lpC'(y_i | z_A) - lpC'(y_i | EMPTY)   per-byte, y_i ~ C' fresh refs
  turn = tau * log(mean_i exp(a_i / tau))       tau=0.03, k=3
  miner = mean(turn)

Prompts render through zai-org/GLM-5.2's own chat template (harness chat.py),
so injection and forcing are byte-exact; the action span is located with the
server's returned token strings (cumulative offsets), never local guesses.

Usage (from research/, venv active, .env sourced):
  python scripts/engy_reason_probe.py refs    --panel albedo --n-turns 60
  python scripts/engy_reason_probe.py score   --panel albedo --n-turns 60
  python scripts/engy_reason_probe.py analyze --panel albedo
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import math
import os
import random
import statistics as st
import sys
import urllib.request
from pathlib import Path

import httpx

from harness.chat import get_tokenizer, gen_prompt, force_text, split_rollout
from harness.config import KING_BENCH

ENGY_BASE = "https://api.engy.ai/v1"
TEACHER = "zai-org/GLM-5.2"        # tokenizer repo (chat template)
TEACHER_API = "glm-5.2"            # engy model id
K_REFS = 3
TAU = 0.03
SAMPLE_TEMP = 0.8
MAX_REF_TOKENS = 1024 + 768        # thought + action, harness defaults
SEED = 120

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
BENCH_URL = "https://s3.hippius.com/affine-sn120/data/benchmarks.json"
EVALS_URL = "https://s3.hippius.com/affine-sn120/evals"


# ---------------------------------------------------------------- engy client
class Engy:
    def __init__(self, key: str, concurrency: int = 24):
        self.cli = httpx.AsyncClient(
            base_url=ENGY_BASE, timeout=180.0,
            headers={"Authorization": f"Bearer {key}"})
        self.sem = asyncio.Semaphore(concurrency)

    async def _post(self, payload: dict) -> dict:
        last = None
        for attempt in range(6):
            async with self.sem:
                try:
                    r = await self.cli.post("/completions", json=payload)
                    if r.status_code == 200:
                        d = r.json()
                        if d.get("choices"):   # 200s without choices happen
                            return d
                        last = f"200 without choices: {r.text[:200]}"
                    else:
                        last = f"HTTP {r.status_code}: {r.text[:200]}"
                except (httpx.HTTPError, ValueError) as e:
                    last = repr(e)
            await asyncio.sleep(2 * (attempt + 1) + random.random() * 2)
        raise RuntimeError(f"engy call failed after retries: {last}")

    async def sample(self, prompt: str) -> str:
        d = await self._post({
            "model": TEACHER_API, "prompt": prompt,
            "max_tokens": MAX_REF_TOKENS, "temperature": SAMPLE_TEMP})
        return d["choices"][0]["text"]

    async def score_span(self, full: str, action: str) -> dict:
        """Sum of per-token logprobs over the trailing `action` span of `full`.

        Long prompts exceed the API's 1024-token full-echo cap, so we send
        `logprob_start_len` (token offset where scoring starts) computed with
        the local GLM-5.2 tokenizer — verified to match the server's ids
        exactly (no BOS shift). A 4-token margin absorbs the null the backend
        emits at the window's first position; returned token_ids are checked
        against the local ids so any drift fails loudly instead of silently
        mis-spanning.
        """
        tok = get_tokenizer(TEACHER)
        enc = tok(full, add_special_tokens=False,
                  return_offsets_mapping=True)
        ids = enc["input_ids"]
        action_start = len(full) - len(action)
        pos = [i for i, (s, _) in enumerate(enc["offset_mapping"])
               if s >= action_start]
        if not pos:
            return {"sum_lp": 0.0, "n_tokens": 0, "lp_per_byte": 0.0}
        start = max(0, pos[0] - 4)
        d = await self._post({
            "model": TEACHER_API, "prompt": ids, "max_tokens": 1,
            "temperature": 0, "echo": True, "logprobs": 1,
            "logprob_start_len": start})
        lp = d["choices"][0]["logprobs"]
        got_ids, lps = lp["token_ids"], lp["token_logprobs"]
        n_prompt = len(ids)
        want = ids[start:n_prompt]
        if got_ids[:len(want)] != want:
            raise RuntimeError("server token ids diverge from local tokenizer")
        span = [lps[i - start] for i in pos
                if i - start < len(lps) and lps[i - start] is not None]
        n_bytes = max(len(action.encode()), 1)
        return {"sum_lp": sum(span), "n_tokens": len(span),
                "lp_per_byte": sum(span) / n_bytes if span else 0.0}


# ------------------------------------------------------------------ data load
def load_turn_prefixes(turn_ids: set[str]) -> dict[str, list[dict]]:
    out = {}
    with open(DATA / "turns_minicoder.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tid = f"{t['traj_id']}:{t['turn_idx']}"
            if tid in turn_ids:
                out[tid] = t["prefix"]
    return out


def load_albedo_thoughts() -> dict[str, dict[str, str]]:
    """turn_id -> miner -> z_a (first rollout, mirrors production 1 sample)."""
    out: dict[str, dict[str, str]] = {}
    with open(RESULTS / "ekings_v2_all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("valid") or not r.get("pairs"):
                continue
            out.setdefault(r["turn_id"], {})[r["miner"]] = r["pairs"][0]["z_a"]
    return out


def pick_turns(turn_ids: list[str], n: int) -> list[str]:
    rng = random.Random(SEED)
    ids = sorted(turn_ids)
    rng.shuffle(ids)
    return ids[:n]


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def fetch_artifact(chal_id: str) -> dict:
    with urllib.request.urlopen(f"{EVALS_URL}/{chal_id}.json.gz",
                                timeout=300) as r:
        return json.loads(gzip.decompress(r.read()))


# ------------------------------------------------------------------ jsonl io
def load_done(path: Path, key_fields: tuple[str, ...]) -> dict:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[tuple(r[k] for k in key_fields)] = r
    return done


def append_jsonl(path: Path, rec: dict, lock: asyncio.Lock | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ ref stage
async def stage_refs(engy: Engy, prefixes: dict[str, list[dict]],
                     out_path: Path):
    """Sample k fresh teacher references per turn; cache to jsonl."""
    done = load_done(out_path, ("turn_id", "ref_idx"))
    todo = [(tid, i) for tid in prefixes for i in range(K_REFS)
            if (tid, i) not in done]
    print(f"refs: {len(done)} cached, {len(todo)} to sample")
    lock = asyncio.Lock()
    n_ok = n_fail = 0

    async def one(tid: str, i: int):
        nonlocal n_ok, n_fail
        prompt = gen_prompt(TEACHER, prefixes[tid])
        for _ in range(4):                    # resample until a valid action
            text = await engy.sample(prompt)
            z, y = split_rollout(text)
            if y:
                break
        else:
            z, y = "", ""
        rec = {"turn_id": tid, "ref_idx": i, "z": z, "y": y}
        async with lock:
            append_jsonl(out_path, rec)
            if y:
                n_ok += 1
            else:
                n_fail += 1
            if (n_ok + n_fail) % 20 == 0:
                print(f"  refs {n_ok + n_fail}/{len(todo)} (fail={n_fail})")

    await asyncio.gather(*[one(t, i) for t, i in todo])
    print(f"refs done: ok={n_ok} fail={n_fail}")


# ---------------------------------------------------------------- score stage
async def stage_score(engy: Engy, prefixes: dict[str, list[dict]],
                      thoughts: dict[str, dict[str, str]],
                      refs_path: Path, out_path: Path):
    """lpC'(y_i|z) for every (turn, ref, miner) plus the EMPTY baseline."""
    refs = load_done(refs_path, ("turn_id", "ref_idx"))
    done = {k: v for k, v in
            load_done(out_path, ("turn_id", "ref_idx", "miner")).items()
            if v.get("lp_per_byte") is not None}   # errored calls retry
    tasks = []
    for (tid, i), ref in refs.items():
        if tid not in prefixes or tid not in thoughts or not ref["y"]:
            continue
        for miner in list(thoughts[tid]) + ["__empty__"]:
            if (tid, i, miner) in done:
                continue
            z = "" if miner == "__empty__" else thoughts[tid][miner]
            tasks.append((tid, i, miner, z, ref["y"]))
    print(f"score: {len(done)} cached, {len(tasks)} calls to make")
    lock = asyncio.Lock()
    n = 0

    async def one(tid, i, miner, z, y):
        nonlocal n
        full = force_text(TEACHER, prefixes[tid], z, y)
        try:
            s = await engy.score_span(full, y)
        except RuntimeError as e:
            s = {"sum_lp": None, "n_tokens": 0, "lp_per_byte": None,
                 "error": str(e)[:200]}
        rec = {"turn_id": tid, "ref_idx": i, "miner": miner, **s}
        async with lock:
            append_jsonl(out_path, rec)
            n += 1
            if n % 50 == 0:
                print(f"  score {n}/{len(tasks)}")

    await asyncio.gather(*[one(*t) for t in tasks])
    print(f"score done: {n} new calls")


# -------------------------------------------------------------------- analyze
def turn_score(a: list[float], tau: float | None) -> float:
    if tau is None or tau <= 0:
        return st.mean(a)
    m = max(a)
    return m + tau * math.log(st.mean(math.exp((x - m) / tau) for x in a))


def rankdata(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra)
           * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def analyze(score_path: Path, swe_map: dict[str, float], label: str,
            out_prefix: Path | None = None):
    """Aggregate scored lp records into per-miner Reason and correlate."""
    by_tm: dict[tuple[str, str], dict[int, float]] = {}
    for line in open(score_path):
        r = json.loads(line)
        if r.get("lp_per_byte") is None:
            continue
        by_tm.setdefault((r["turn_id"], r["miner"]), {})[r["ref_idx"]] = \
            r["lp_per_byte"]

    empty = {t: v for (t, m), v in by_tm.items() if m == "__empty__"}
    per_miner: dict[str, dict[str, dict[str, float]]] = {}
    for (tid, miner), lps in by_tm.items():
        if miner == "__empty__" or tid not in empty:
            continue
        a = [lps[i] - empty[tid][i] for i in lps if i in empty[tid]]
        if not a:
            continue
        per_miner.setdefault(miner, {})[tid] = {
            "lme": turn_score(a, TAU), "mean": turn_score(a, None), "k": len(a)}

    rows = []
    for miner, turns in sorted(per_miner.items()):
        if miner not in swe_map:
            continue
        rows.append({
            "miner": miner, "swe": swe_map[miner], "n_turns": len(turns),
            "reason_lme": st.mean(t["lme"] for t in turns.values()),
            "reason_mean": st.mean(t["mean"] for t in turns.values()),
        })
    rows.sort(key=lambda r: -r["swe"])

    print(f"\n== {label}: engy glm-5.2 Reason vs swe (n={len(rows)}) ==")
    print(f"{'miner':<44} {'swe':>6} {'R_lme':>9} {'R_mean':>9} {'turns':>6}")
    for r in rows:
        print(f"{r['miner']:<44} {r['swe']:>6.2f} {r['reason_lme']:>9.5f} "
              f"{r['reason_mean']:>9.5f} {r['n_turns']:>6}")
    if len(rows) >= 4:
        swe = [r["swe"] for r in rows]
        rho_lme = spearman([r["reason_lme"] for r in rows], swe)
        rho_mean = spearman([r["reason_mean"] for r in rows], swe)
        print(f"\nSpearman(Reason_lme  tau={TAU}, swe) = {rho_lme:+.3f}")
        print(f"Spearman(Reason_mean v3-style, swe) = {rho_mean:+.3f}")
    if out_prefix:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out_prefix}.json", "w") as f:
            json.dump({"label": label, "tau": TAU, "k": K_REFS,
                       "teacher": TEACHER_API, "rows": rows}, f, indent=1)
    return rows


# -------------------------------------------------------------------- drivers
def albedo_inputs(n_turns: int):
    thoughts = load_albedo_thoughts()
    turn_ids = pick_turns(list(thoughts), n_turns)
    prefixes = load_turn_prefixes(set(turn_ids))
    thoughts = {t: thoughts[t] for t in turn_ids if t in prefixes}
    swe = {f"king-{k}": v for k, v in KING_BENCH.items()}
    return prefixes, thoughts, swe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["refs", "score", "analyze"])
    ap.add_argument("--panel", default="albedo", choices=["albedo", "live"])
    ap.add_argument("--n-turns", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    key = os.environ.get("ENGY", "")
    if not key and args.stage != "analyze":
        sys.exit("set ENGY env var (source .env)")

    tag = f"{args.panel}_n{args.n_turns}"
    refs_path = RESULTS / f"engy_refs_{tag}.jsonl"
    score_path = RESULTS / f"engy_scores_{tag}.jsonl"

    if args.panel == "albedo":
        prefixes, thoughts, swe = albedo_inputs(args.n_turns)
    else:
        sys.exit("live panel: use engy_reason_live.py (built separately)")

    if args.stage == "refs":
        engy = Engy(key, args.concurrency)
        asyncio.run(stage_refs(engy, prefixes, refs_path))
    elif args.stage == "score":
        engy = Engy(key, args.concurrency)
        asyncio.run(stage_score(engy, prefixes, thoughts, refs_path,
                                score_path))
    else:
        analyze(score_path, swe, f"{args.panel} panel",
                RESULTS / f"engy_reason_{tag}")


if __name__ == "__main__":
    main()
