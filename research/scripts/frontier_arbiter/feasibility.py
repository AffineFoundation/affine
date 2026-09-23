"""Frontier-arbiter feasibility probe (2026-09-20): is sampling a frontier
model F on live duel turns BUILDABLE, and at what cost?

Measures, on real wvk-22 duel turns re-materialised from the public corpus:
  shape        prompt-token distribution of live slices by dialect (teacher
               tokenizer) + D-wide totals + D growth from the manifest chain
  sample       n real turns per model at T=0.8 -> real per-call charge,
               completion tokens, latency, parse (feeds cost + compliance)
  throughput   200 turns at a fixed concurrency for one model -> wall, p50/p95,
               error rate
  greedy       40 fixed prompts x every model at T=0 (one pass; passes are
               spaced >= 10 min by the caller) -> determinism across passes
  stochastic   the same 40 prompts x 3 at T=0.8 -> self-agreement of actions
  topup        fill rare dialects (boxed / terminus_json) to >= 30 per model
  report       aggregate everything under results/frontier_arbiter/feasibility

Every Engy reply is appended to samples.jsonl (one row per call, with the
charge from x_engy.charged_micro) and the cumulative spend is kept in
ledger.json; the run aborts at BUDGET_USD. Materialised prefixes are cached
under /tmp/frontier_arbiter (they are large and reproducible).

Usage (venv: cd /workspace && source .venv/bin/activate):
  python research/scripts/frontier_arbiter/feasibility.py shape
  python research/scripts/frontier_arbiter/feasibility.py sample --model glm-5.3 --n 100
  python research/scripts/frontier_arbiter/feasibility.py throughput --model glm-5.3 --conc 32
  python research/scripts/frontier_arbiter/feasibility.py greedy --pass 1
  python research/scripts/frontier_arbiter/feasibility.py stochastic
  python research/scripts/frontier_arbiter/feasibility.py topup
  python research/scripts/frontier_arbiter/feasibility.py report
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402  (live tree, see common)

RESULTS = C.REPO / "research" / "results" / "frontier_arbiter" / "feasibility"
CACHE = Path("/tmp/frontier_arbiter")
POOL = CACHE / "pool.jsonl.gz"
SAMPLES = RESULTS / "samples.jsonl"
LEDGER = RESULTS / "ledger.json"
VERDICTS = ("chal-00614", "chal-00621", "chal-00623")
MODELS = list(C.FRONTIER_MODELS)
DIALECTS = ("bash", "tool_call", "text", "boxed", "terminus_json")
BUDGET_USD = 40.0
DUELS_PER_DAY = 30
N_TURNS_LIVE = 1000          # [duel].n_turns, wvk 22
SEED = 20260920

# Dialect share of the live slice (mean of the three verdicts, filled by shape).
SHAPE_JSON = RESULTS / "shape.json"


# ------------------------------------------------------------------ ledger
def ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"cost_usd": 0.0, "calls": 0, "by_model": {}}


def ledger_add(model: str, cost: float) -> float:
    L = ledger()
    L["cost_usd"] += cost
    L["calls"] += 1
    m = L["by_model"].setdefault(model, {"calls": 0, "cost_usd": 0.0})
    m["calls"] += 1
    m["cost_usd"] += cost
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(L, indent=1))
    return L["cost_usd"]


class Budget(Exception):
    pass


# ------------------------------------------------------------------ pool
def build_pool() -> list[dict]:
    """Every turn of the three verdicts, materialised: tid, kind, source,
    prompt tokens (teacher tokenizer, generation prompt incl. <think>),
    teacher-ref completion tokens, and the prefix messages (cached gz)."""
    if POOL.exists():
        return C.read_jsonl(POOL)
    tok = C.teacher_tokenizer()
    out: list[dict] = []
    for rec in VERDICTS:
        d = C.load_verdict(rec)
        tids = d["turn_ids"]
        refs = d.get("teacher_refs") or {}
        t0 = time.time()
        turns = C.materialize(d, tids)
        print(f"[pool] {rec}: materialised {len(turns)}/{len(tids)} in {time.time() - t0:.0f}s",
              flush=True)
        for tid in tids:
            tr = turns.get(tid)
            if tr is None:
                continue
            prompt = C.gen_prompt(tr["prefix"])
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            ref_toks = [len(tok(r["z"] + "\n</think>\n\n" + r["y"], add_special_tokens=False)["input_ids"])
                        for r in refs.get(tid, [])]
            out.append({"verdict": rec, "tid": tid, "kind": tr.get("action_kind") or "bash",
                        "source": tr.get("source"), "n_prefix_chars": tr.get("n_prefix_chars"),
                        "prompt_tokens": n_prompt, "ref_completion_tokens": ref_toks,
                        "prefix": tr["prefix"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(POOL, "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


def pick(pool: list[dict], n: int, seed: int, kinds: tuple[str, ...] | None = None,
         max_tokens: int | None = None, exclude: set[str] = frozenset()) -> list[dict]:
    rng = random.Random(seed)
    cand = [r for r in pool if (kinds is None or r["kind"] in kinds)
            and (max_tokens is None or r["prompt_tokens"] <= max_tokens)
            and r["tid"] not in exclude]
    rng.shuffle(cand)
    return cand[:n]


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": st.mean(xs), "p50": pct(xs, .5), "p90": pct(xs, .9),
            "p95": pct(xs, .95), "max": max(xs), "sum": sum(xs)}


# ------------------------------------------------------------------ step 1: shape
def manifest_chain(depth: int = 40) -> list[dict]:
    """(published_at, n_turns, corpus_epoch) walking prev_manifest back."""
    out = []
    url = "https://data.affine.io/corpus/manifest.json"
    with httpx.Client(timeout=60) as cli:
        for _ in range(depth):
            m = cli.get(url).json()
            if m.get("published_at") is None or m.get("n_turns") is None:
                break       # older manifests carry no timestamp
            out.append({"published_at": m.get("published_at"), "n_turns": m.get("n_turns"),
                        "corpus_epoch": m.get("corpus_epoch")})
            prev = m.get("prev_manifest")
            if not prev:
                break
            url = "https://data.affine.io/" + prev
    return out


def step_shape() -> dict:
    pool = build_pool()
    by_v: dict[str, dict] = {}
    for rec in VERDICTS:
        rows = [r for r in pool if r["verdict"] == rec]
        by_kind = {}
        for k in DIALECTS:
            ks = [r for r in rows if r["kind"] == k]
            if not ks:
                continue
            by_kind[k] = {"n_turns": len(ks),
                          "prompt_tokens": summarize([r["prompt_tokens"] for r in ks]),
                          "ref_completion_tokens": summarize(
                              [t for r in ks for t in r["ref_completion_tokens"]])}
        by_v[rec] = {"n_turns": len(rows),
                     "prompt_tokens": summarize([r["prompt_tokens"] for r in rows]),
                     "prompt_tokens_gt_128k": sum(r["prompt_tokens"] > 128_000 for r in rows),
                     "prompt_tokens_gt_64k": sum(r["prompt_tokens"] > 64_000 for r in rows),
                     "ref_completion_tokens": summarize(
                         [t for r in rows for t in r["ref_completion_tokens"]]),
                     "by_kind": by_kind}
    allrows = pool
    chars = sum(r["n_prefix_chars"] or 0 for r in allrows)
    toks = sum(r["prompt_tokens"] for r in allrows)
    tok_per_char = toks / chars if chars else None
    # D-wide from the live manifest's index (n_prefix_chars per turn).
    d = C.load_verdict(VERDICTS[-1])
    sl = d["verdict"]["slice"]
    corpus = C.corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
    idx = corpus.load_index_rows()
    d_by_kind = collections.Counter(r.get("action_kind") for r in idx)
    d_chars = sum(r.get("n_prefix_chars") or 0 for r in idx)
    chain = manifest_chain()
    growth = None
    if len(chain) >= 2:
        # turns/day over the chain window
        ts = [time.mktime(time.strptime(c["published_at"][:19], "%Y-%m-%dT%H:%M:%S"))
              for c in chain if c.get("published_at")]
        if len(ts) >= 2 and ts[0] > ts[-1]:
            growth = (chain[0]["n_turns"] - chain[-1]["n_turns"]) / ((ts[0] - ts[-1]) / 86400)
    shape = {
        "verdicts": by_v,
        "pooled": {"n_turns": len(allrows),
                   "prompt_tokens": summarize([r["prompt_tokens"] for r in allrows]),
                   "by_kind": {k: {"n_turns": sum(r["kind"] == k for r in allrows),
                                   "share": sum(r["kind"] == k for r in allrows) / len(allrows),
                                   "prompt_tokens": summarize([r["prompt_tokens"] for r in allrows if r["kind"] == k])}
                               for k in DIALECTS},
                   "tokens_per_1000_turn_slice": toks / len(allrows) * N_TURNS_LIVE,
                   "tok_per_char": tok_per_char,
                   "ref_completion_tokens": summarize([t for r in allrows for t in r["ref_completion_tokens"]])},
        "D": {"manifest_sha256": sl["manifest_sha256"], "n_turns": len(idx),
              "by_kind": dict(d_by_kind), "prefix_chars_total": d_chars,
              "prompt_tokens_total_est": d_chars * tok_per_char if tok_per_char else None,
              "mean_prompt_tokens_est": d_chars * tok_per_char / len(idx) if tok_per_char else None,
              "manifest_chain": chain, "growth_turns_per_day": growth},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    SHAPE_JSON.write_text(json.dumps(shape, indent=1))
    p = shape["pooled"]
    print(f"[shape] pooled n={p['n_turns']} prompt tokens p50={p['prompt_tokens']['p50']:.0f} "
          f"p90={p['prompt_tokens']['p90']:.0f} max={p['prompt_tokens']['max']} "
          f"tokens/1000-turn slice={p['tokens_per_1000_turn_slice']:.3e}")
    print(f"[shape] D n={shape['D']['n_turns']} est prompt tokens={shape['D']['prompt_tokens_total_est']:.3e} "
          f"growth/day={growth}")
    return shape


# ------------------------------------------------------------------ sampling
def think_closed(reply: dict) -> bool:
    """Engy returns reasoning separately; the model 'closed </think>' iff it
    produced a visible message (content or a tool call) after reasoning."""
    return bool((reply.get("content") or "").strip() or reply.get("tool_calls"))


async def one_call(engy: C.Engy, model: str, turn: dict, temperature: float, tag: str,
                   run: int, sem: asyncio.Semaphore, max_tokens: int = C.DUEL_MAX_TOKENS) -> dict:
    # Latency is timed INSIDE the concurrency gate (a call's own wall, not its
    # queue wait); the Engy client's inner semaphore is sized never to block.
    async with sem:
        t0 = time.monotonic()
        row = {"tag": tag, "model": model, "tid": turn["tid"], "kind": turn["kind"],
               "verdict": turn["verdict"], "temperature": temperature, "run": run,
               "prompt_tokens_teacher_tok": turn["prompt_tokens"], "at": time.time()}
        try:
            reply = await engy.chat(model, turn["prefix"], temperature=temperature,
                                    max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 — the error class is the finding
            row.update({"ok": False, "error": repr(e)[:300], "latency_s": time.monotonic() - t0})
            C.append_jsonl(SAMPLES, row)
            return row
        lat = time.monotonic() - t0
    ro = C.reply_to_rollout(reply, turn["kind"])
    z_live, y_live = split_rollout(
        (reply.get("reasoning") or "") + "\n" + C.THINK_CLOSE + "\n" + _content_with_tools(reply, turn["kind"]),
        turn["kind"], require_think_close=True, text_fallback_at_tool_turns=True)
    u = reply.get("usage") or {}
    cd = (u.get("completion_tokens_details") or {})
    row.update({
        "ok": True, "latency_s": lat, "worker": reply.get("worker"), "finish": reply.get("finish"),
        "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": cd.get("reasoning_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "cost_usd": reply.get("cost_usd"), "think_closed": think_closed(reply),
        "parsed": ro["parsed"], "repaired": ro["repaired"],
        "parsed_live": bool(y_live), "live_kind_used": (
            "text" if (y_live and turn["kind"] == "tool_call" and not ro["parsed"]) else turn["kind"] if y_live else None),
        "n_tool_calls": len(reply.get("tool_calls") or []),
        "reasoning_chars": ro["reasoning_chars"], "content_chars": ro["content_chars"],
        "z": ro["z"], "y": ro["y"], "reasoning_sha": hashlib.sha256((reply.get("reasoning") or "").encode()).hexdigest()[:16],
        "y_live": y_live,
    })
    total = ledger_add(model, reply.get("cost_usd") or 0.0)
    C.append_jsonl(SAMPLES, row)
    print(f"  {model:20s} {turn['kind']:13s} {lat:6.1f}s in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
          f"${reply.get('cost_usd') or 0:.4f} fin={reply.get('finish')} parsed={ro['parsed']} "
          f"| running ${total:.2f}", flush=True)
    if total > BUDGET_USD:
        raise Budget(f"budget cap ${BUDGET_USD} exceeded: ${total:.2f}")
    return row


def _content_with_tools(reply: dict, kind: str) -> str:
    content = reply.get("content") or ""
    if reply.get("tool_calls"):
        content = content.rstrip() + "\n" + "\n".join(C.render_tool_call(c) for c in reply["tool_calls"])
    if kind == "tool_call":
        content, _ = C.repair_tool_call(content)
    return content


async def run_calls(engy: C.Engy, jobs: list[tuple[str, dict, float, str, int]],
                    conc: int) -> list[dict]:
    sem = asyncio.Semaphore(conc)
    return list(await asyncio.gather(*[one_call(engy, m, t, temp, tag, run, sem)
                                       for m, t, temp, tag, run in jobs]))


def client(conc: int, retries: int = 6) -> C.Engy:
    return C.Engy(concurrency=conc * 2, retries=retries)


def done_keys() -> set[tuple]:
    return {(r["tag"], r["model"], r["tid"], r["run"]) for r in C.read_jsonl(SAMPLES) if r.get("ok")}


# ------------------------------------------------------------------ step 2: cost samples
def step_sample(model: str, n: int, conc: int) -> None:
    pool = build_pool()
    turns = pick(pool, n, SEED + 1)
    have = done_keys()
    jobs = [(model, t, 0.8, "cost", 0) for t in turns if ("cost", model, t["tid"], 0) not in have]
    print(f"[sample] {model}: {len(jobs)} calls (of {n}) at conc {conc}")
    engy = client(conc)
    asyncio.run(run_calls(engy, jobs, conc))
    print(f"[sample] {model} usage: {engy.usage}")


# ------------------------------------------------------------------ step 3: throughput
def step_throughput(model: str, conc: int, n: int, retries: int = 1, redo: int = 0) -> None:
    pool = build_pool()
    turns = pick(pool, n, SEED + 2)
    tag = f"tp{conc}" + (f"r{retries}" if retries != 1 else "") + (f"b{redo}" if redo else "")
    have = done_keys()
    jobs = [(model, t, 0.8, tag, 0) for t in turns if (tag, model, t["tid"], 0) not in have]
    print(f"[throughput] {model} conc={conc} retries={retries}: {len(jobs)} calls")
    engy = client(conc, retries=retries)   # retries=1 counts failures instead of hiding them
    t0 = time.monotonic()
    rows = asyncio.run(run_calls(engy, jobs, conc))
    wall = time.monotonic() - t0
    ok = [r for r in rows if r.get("ok")]
    lat = [r["latency_s"] for r in ok]
    rec = {"model": model, "conc": conc, "retries": retries, "redo": redo, "n": len(rows), "n_ok": len(ok), "wall_s": wall,
           "lat_p50": pct(lat, .5), "lat_p95": pct(lat, .95), "lat_max": max(lat) if lat else None,
           "calls_per_min": len(ok) / wall * 60 if wall else None,
           "errors": collections.Counter(r.get("error", "")[:60] for r in rows if not r.get("ok")),
           "prompt_tokens_sum": sum(r.get("prompt_tokens") or 0 for r in ok),
           "completion_tokens_sum": sum(r.get("completion_tokens") or 0 for r in ok),
           "cost_usd": sum(r.get("cost_usd") or 0 for r in ok),
           # wall for 1000 turns at this rate, counting only the calls that succeeded
           "wall_1000_turns_est_s": wall / max(len(ok), 1) * N_TURNS_LIVE, "at": time.time()}
    C.append_jsonl(RESULTS / "throughput.jsonl", rec)
    print(f"[throughput] {model} conc={conc}: wall={wall:.0f}s ok={len(ok)}/{len(rows)} "
          f"p50={rec['lat_p50']:.1f}s p95={rec['lat_p95']:.1f}s -> 1000 turns ≈ {rec['wall_1000_turns_est_s'] / 60:.1f} min")


# ------------------------------------------------------------------ step 4: determinism
def det_prompts(pool: list[dict]) -> list[dict]:
    """40 fixed prompts: 8 per dialect (fewer where the dialect is thin),
    prompt ≤ 40k tokens to bound the cost; the shortfall is filled from
    bash/tool_call/text."""
    out: list[dict] = []
    for k in DIALECTS:
        out += pick(pool, 8, SEED + 3, kinds=(k,), max_tokens=40_000)
    have = {r["tid"] for r in out}
    out += pick(pool, 40 - len(out), SEED + 4, kinds=("bash", "tool_call", "text"),
                max_tokens=40_000, exclude=have)
    return out[:40]


def step_greedy(pass_no: int, conc: int, models: list[str]) -> None:
    pool = build_pool()
    turns = det_prompts(pool)
    tag = "greedy"
    have = done_keys()
    jobs = [(m, t, 0.0, tag, pass_no) for m in models for t in turns
            if (tag, m, t["tid"], pass_no) not in have]
    print(f"[greedy] pass {pass_no}: {len(jobs)} calls")
    engy = client(conc)
    asyncio.run(run_calls(engy, jobs, conc))


def step_stochastic(conc: int, models: list[str]) -> None:
    pool = build_pool()
    turns = det_prompts(pool)
    tag = "stoch"
    have = done_keys()
    jobs = [(m, t, 0.8, tag, r) for m in models for t in turns for r in (1, 2, 3)
            if (tag, m, t["tid"], r) not in have]
    print(f"[stochastic] {len(jobs)} calls")
    engy = client(conc)
    asyncio.run(run_calls(engy, jobs, conc))


# ------------------------------------------------------------------ step 5: top-up rare dialects
def step_topup(conc: int, models: list[str], target: int) -> None:
    pool = build_pool()
    rows = [r for r in C.read_jsonl(SAMPLES) if r.get("ok")]
    jobs = []
    for m in models:
        for k in DIALECTS:
            have_t = {r["tid"] for r in rows if r["model"] == m and r["kind"] == k}
            need = target - len(have_t)
            if need <= 0:
                continue
            for t in pick(pool, need, SEED + 5, kinds=(k,), exclude=have_t):
                jobs.append((m, t, 0.8, "topup", 0))
    print(f"[topup] {len(jobs)} calls")
    engy = client(conc)
    asyncio.run(run_calls(engy, jobs, conc))


# ------------------------------------------------------------------ report
def fit_price(rows: list[dict]) -> dict:
    """charged ≈ p_in·prompt + p_out·completion (USD per token), least squares
    through the origin over this model's calls; also plain per-call mean."""
    xs = [(r["prompt_tokens"], r["completion_tokens"], r["cost_usd"]) for r in rows
          if r.get("prompt_tokens") and r.get("completion_tokens") is not None and r.get("cost_usd")]
    out = {"n": len(xs), "mean_cost_per_call": st.mean(x[2] for x in xs) if xs else None,
           "mean_prompt_tokens": st.mean(x[0] for x in xs) if xs else None,
           "mean_completion_tokens": st.mean(x[1] for x in xs) if xs else None}
    if len(xs) >= 5:
        # 2x2 normal equations
        a11 = sum(p * p for p, _, _ in xs); a12 = sum(p * c for p, c, _ in xs); a22 = sum(c * c for _, c, _ in xs)
        b1 = sum(p * y for p, _, y in xs); b2 = sum(c * y for _, c, y in xs)
        det = a11 * a22 - a12 * a12
        if det:
            p_in = (b1 * a22 - b2 * a12) / det
            p_out = (a11 * b2 - a12 * b1) / det
            resid = [y - p_in * p - p_out * c for p, c, y in xs]
            out.update({"usd_per_M_prompt": p_in * 1e6, "usd_per_M_completion": p_out * 1e6,
                        "fit_rmse_usd": (sum(r * r for r in resid) / len(resid)) ** 0.5})
    return out


def agreement_stats(groups: dict[tuple, list[dict]]) -> dict:
    """Per (model) determinism over runs of the same tid."""
    ex_all = []; jac_all = []; reason_same = []; worker_sets = []; ex_parsed = []
    by_kind: dict[str, list[float]] = collections.defaultdict(list)
    pair_by_worker: dict[str, list[float]] = {"same": [], "diff": []}
    n_prompts = 0
    for (model, tid), rs in groups.items():
        rs = [r for r in rs if r.get("ok")]
        if len(rs) < 2:
            continue
        n_prompts += 1
        kind = rs[0]["kind"]
        ys = [r["y"] for r in rs]
        parsed = [r for r in rs if r["parsed"]]
        norms = [C.norm_action(r["y"], kind) for r in parsed]
        ex_all.append(1.0 if len(parsed) == len(rs) and len(set(norms)) == 1 else 0.0)
        by_kind[kind].append(ex_all[-1])
        if len(parsed) == len(rs):
            ex_parsed.append(ex_all[-1])
        # mean pairwise jaccard of actions (unparsed = empty)
        pairs = [(a, b) for i, a in enumerate(ys) for b in ys[i + 1:]]
        jac_all.append(st.mean(C.jaccard(a, b) for a, b in pairs) if pairs else None)
        reason_same.append(1.0 if len({r["reasoning_sha"] for r in rs}) == 1 else 0.0)
        worker_sets.append(len({r.get("worker") for r in rs}))
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                same = (a["parsed"] and b["parsed"]
                        and C.norm_action(a["y"], kind) == C.norm_action(b["y"], kind))
                pair_by_worker["same" if a.get("worker") == b.get("worker") else "diff"].append(1.0 if same else 0.0)
    return {"n_prompts": n_prompts,
            "action_exact_all_runs": st.mean(ex_all) if ex_all else None,
            "action_exact_given_all_parsed": st.mean(ex_parsed) if ex_parsed else None,
            "n_all_parsed": len(ex_parsed),
            "exact_by_kind": {k: (st.mean(v), len(v)) for k, v in sorted(by_kind.items())},
            "pair_exact_same_worker": (st.mean(pair_by_worker["same"]), len(pair_by_worker["same"])) if pair_by_worker["same"] else None,
            "pair_exact_diff_worker": (st.mean(pair_by_worker["diff"]), len(pair_by_worker["diff"])) if pair_by_worker["diff"] else None,
            "action_mean_pairwise_jaccard": st.mean(j for j in jac_all if j is not None) if jac_all else None,
            "reasoning_identical_all_runs": st.mean(reason_same) if reason_same else None,
            "mean_distinct_workers": st.mean(worker_sets) if worker_sets else None}


def step_report() -> None:
    shape = json.loads(SHAPE_JSON.read_text()) if SHAPE_JSON.exists() else step_shape()
    rows = C.read_jsonl(SAMPLES)
    ok = [r for r in rows if r.get("ok")]
    L = ledger()
    slice_tokens = shape["pooled"]["tokens_per_1000_turn_slice"]
    d_tokens = shape["D"]["prompt_tokens_total_est"]
    d_n = shape["D"]["n_turns"]
    growth = shape["D"].get("growth_turns_per_day") or 15_000
    growth_tokens = growth * (d_tokens / d_n)
    out_lines: list[str] = []
    P = out_lines.append

    P("FRONTIER ARBITER — FEASIBILITY (2026-09-20)")
    P(f"spend so far: ${L['cost_usd']:.2f} over {L['calls']} Engy calls; by model: "
      + ", ".join(f"{m} ${v['cost_usd']:.2f}/{v['calls']}" for m, v in L["by_model"].items()))
    P("")
    P("1. LIVE SLICE SHAPE (teacher tokenizer Qwen/Qwen3.8-27B, prompt = chat template + <think>)")
    for rec, v in shape["verdicts"].items():
        pt = v["prompt_tokens"]
        P(f"  {rec}: n={v['n_turns']} prompt tokens p50={pt['p50']:.0f} p90={pt['p90']:.0f} max={pt['max']} "
          f"total={pt['sum']:.3e} (>64k: {v['prompt_tokens_gt_64k']}, >128k: {v['prompt_tokens_gt_128k']}); "
          f"teacher ref completion tokens p50={v['ref_completion_tokens'].get('p50')} p90={v['ref_completion_tokens'].get('p90')}")
    P(f"  {'dialect':14s} {'n(3 slices)':>11s} {'share':>6s} {'p50':>7s} {'p90':>7s} {'max':>7s} {'tokens/1000-slice':>18s}")
    for k, v in shape["pooled"]["by_kind"].items():
        pt = v["prompt_tokens"]
        if not pt.get("n"):
            continue
        P(f"  {k:14s} {v['n_turns']:11d} {v['share']:6.3f} {pt['p50']:7.0f} {pt['p90']:7.0f} {pt['max']:7d} "
          f"{v['share'] * N_TURNS_LIVE * pt['mean']:18.3e}")
    P(f"  pooled: prompt tokens p50={shape['pooled']['prompt_tokens']['p50']:.0f} mean={shape['pooled']['prompt_tokens']['mean']:.0f} "
      f"-> tokens per 1000-turn slice = {slice_tokens:.3e} (prompt only, ×n_F per frontier sample)")
    P(f"  D (manifest {shape['D']['manifest_sha256'][:12]}): {d_n} turns, {shape['D']['prefix_chars_total']:.3e} prefix chars "
      f"× {shape['pooled']['tok_per_char']:.3f} tok/char ≈ {d_tokens:.3e} prompt tokens; by kind {shape['D']['by_kind']}")
    P(f"  D growth (manifest chain, {len(shape['D']['manifest_chain'])} manifests): {growth:.0f} turns/day ≈ {growth_tokens:.3e} tokens/day")
    P("")

    # ---- cost
    P("2. COST (real x_engy.charged_micro; fit charge ≈ p_in·prompt + p_out·completion)")
    P(f"  {'model':20s} {'n':>4s} {'$/call':>8s} {'in_tok':>7s} {'out_tok':>7s} {'$/M in':>7s} {'$/M out':>8s} "
      f"| {'$/duel nF=1':>11s} {'nF=3':>8s} | {'$/day nF=1':>10s} {'nF=3':>8s} | {'D one-off nF=1':>14s} {'nF=3':>9s} | {'D daily nF=1':>12s}")
    cost_table = {}
    for m in MODELS:
        mr = [r for r in ok if r["model"] == m and r["tag"] in ("cost", "tp32", "tp64", "topup", "stoch")]
        f = fit_price(mr)
        if not f["n"]:
            continue
        # per-duel projection at the slice's prompt distribution
        p_in = f.get("usd_per_M_prompt", C.PRICE_PER_M.get(m, 0) ) / 1e6
        p_out = f.get("usd_per_M_completion", p_in * 3) / 1e6
        comp = f["mean_completion_tokens"] or 0
        per_turn = p_in * shape["pooled"]["prompt_tokens"]["mean"] + p_out * comp
        per_duel1 = per_turn * N_TURNS_LIVE
        d_oneoff1 = p_in * d_tokens + p_out * comp * d_n
        d_daily1 = p_in * growth_tokens + p_out * comp * growth
        cost_table[m] = {**f, "usd_per_turn_fit": per_turn, "usd_per_duel_nF1": per_duel1,
                         "usd_per_duel_nF3": per_duel1 * 3, "usd_per_day_nF1": per_duel1 * DUELS_PER_DAY,
                         "usd_per_day_nF3": per_duel1 * 3 * DUELS_PER_DAY,
                         "D_oneoff_nF1": d_oneoff1, "D_oneoff_nF3": d_oneoff1 * 3, "D_daily_nF1": d_daily1,
                         "D_daily_nF3": d_daily1 * 3}
        P(f"  {m:20s} {f['n']:4d} {f['mean_cost_per_call']:8.4f} {f['mean_prompt_tokens']:7.0f} {comp:7.0f} "
          f"{f.get('usd_per_M_prompt', float('nan')):7.3f} {f.get('usd_per_M_completion', float('nan')):8.3f} "
          f"| {per_duel1:11.2f} {per_duel1 * 3:8.2f} | {per_duel1 * DUELS_PER_DAY:10.0f} {per_duel1 * 3 * DUELS_PER_DAY:8.0f} "
          f"| {d_oneoff1:14.0f} {d_oneoff1 * 3:9.0f} | {d_daily1:12.0f}")
    P(f"  ($/duel = 1000 turns × (p_in × mean slice prompt {shape['pooled']['prompt_tokens']['mean']:.0f} tok + p_out × mean completion); "
      f"$/day at {DUELS_PER_DAY} duels/day; D one-off = every turn of D once; D daily = {growth:.0f} new turns/day)")
    P("")

    # ---- throughput
    P("3. LATENCY / THROUGHPUT (200 real turns, T=0.8, max_tokens 2816, retries=1)")
    # last record per (model, conc) wins: the first glm-5.3-flash conc-32 run
    # timed calls outside the concurrency gate (queue wait included).
    tps = list({(t["model"], t["conc"], t.get("retries", 1)): t for t in C.read_jsonl(RESULTS / "throughput.jsonl")}.values())
    P(f"  {'model':20s} {'conc':>4s} {'retry':>5s} {'ok/n':>8s} {'wall s':>7s} {'p50 s':>6s} {'p95 s':>6s} {'max s':>6s} {'calls/min':>9s} {'1000-turn wall':>14s} errors")
    for t in tps:
        P(f"  {t['model']:20s} {t['conc']:4d} {t.get('retries', 1):5d} {t['n_ok']:3d}/{t['n']:<4d} {t['wall_s']:7.0f} {t['lat_p50'] or 0:6.1f} {t['lat_p95'] or 0:6.1f} "
          f"{t['lat_max'] or 0:6.1f} {t['calls_per_min'] or 0:9.1f} {t['wall_1000_turns_est_s'] / 60:11.1f} min {dict(t['errors']) or ''}")
    P("  live duel: 43–45 min for 1000 turns (chal-00614/621/623 duel_seconds 2628/2586/2719 s); the frontier call "
      "runs in parallel with the teacher refs on the API side, so it is 'free' iff its 1000-turn wall ≤ the duel wall.")
    P("")

    # ---- determinism
    P("4. DETERMINISM / REPLAYABILITY (40 fixed prompts, ≤40k tokens, 8 per dialect where available)")
    gre = [r for r in ok if r["tag"] == "greedy"]
    sto = [r for r in ok if r["tag"] == "stoch"]
    for m in MODELS:
        gm = [r for r in gre if r["model"] == m]
        passes = sorted({r["run"] for r in gm})
        ts = {p: [r["at"] for r in gm if r["run"] == p] for p in passes}
        spacing = [(passes[i], (min(ts[passes[i]]) - max(ts[passes[i - 1]])) / 60)
                   for i in range(1, len(passes))] if len(passes) > 1 else []
        P(f"  {m}: greedy passes {passes}; gap from last call of pass i-1 to first of pass i (min): "
          + ", ".join(f"p{p}: {g:.1f}" for p, g in spacing))
    P(f"  {'model':20s} | {'T=0 ×3: exact':>13s} {'jaccard':>8s} {'reason=':>8s} {'workers':>7s} | {'T=0.8 ×3: exact':>15s} {'jaccard':>8s} {'reason=':>8s} | {'T0 vs T0.8 exact':>16s}")
    det_table = {}
    for m in MODELS:
        g = collections.defaultdict(list)
        for r in gre:
            if r["model"] == m:
                g[(m, r["tid"])].append(r)
        s = collections.defaultdict(list)
        for r in sto:
            if r["model"] == m:
                s[(m, r["tid"])].append(r)
        ga, sa = agreement_stats(g), agreement_stats(s)
        # does a fresh T=0.8 sample match the greedy action? (the memorisation payoff)
        cross = []
        for (mm, tid), rs in g.items():
            gp = [r for r in rs if r["parsed"]]
            if not gp:
                continue
            kind = gp[0]["kind"]
            greedy_norms = {C.norm_action(r["y"], kind) for r in gp}
            for r in s.get((mm, tid), []):
                cross.append(1.0 if r["parsed"] and C.norm_action(r["y"], kind) in greedy_norms else 0.0)
        det_table[m] = {"greedy": ga, "stochastic": sa,
                        "stoch_matches_greedy": st.mean(cross) if cross else None, "n_cross": len(cross)}
        P(f"  {m:20s} | {ga['action_exact_all_runs'] if ga['action_exact_all_runs'] is not None else float('nan'):13.2f} "
          f"{ga['action_mean_pairwise_jaccard'] if ga['action_mean_pairwise_jaccard'] is not None else float('nan'):8.2f} "
          f"{ga['reasoning_identical_all_runs'] if ga['reasoning_identical_all_runs'] is not None else float('nan'):8.2f} "
          f"{ga['mean_distinct_workers'] if ga['mean_distinct_workers'] is not None else float('nan'):7.2f} | "
          f"{sa['action_exact_all_runs'] if sa['action_exact_all_runs'] is not None else float('nan'):15.2f} "
          f"{sa['action_mean_pairwise_jaccard'] if sa['action_mean_pairwise_jaccard'] is not None else float('nan'):8.2f} "
          f"{sa['reasoning_identical_all_runs'] if sa['reasoning_identical_all_runs'] is not None else float('nan'):8.2f} | "
          f"{det_table[m]['stoch_matches_greedy'] if det_table[m]['stoch_matches_greedy'] is not None else float('nan'):16.2f}  (n prompts {ga['n_prompts']}/{sa['n_prompts']})")
    P("  T=0 exact given every run parsed (n): " + "; ".join(
        f"{m} {det_table[m]['greedy']['action_exact_given_all_parsed'] if det_table[m]['greedy'].get('action_exact_given_all_parsed') is not None else float('nan'):.2f} ({det_table[m]['greedy'].get('n_all_parsed')})"
        for m in det_table))
    P("  T=0 pairwise exact, same worker vs different worker (share, n pairs): " + "; ".join(
        f"{m} same {det_table[m]['greedy'].get('pair_exact_same_worker')} / diff {det_table[m]['greedy'].get('pair_exact_diff_worker')}"
        for m in det_table))
    P("  T=0 exact by dialect (share, n prompts): " + " | ".join(
        f"{m}: " + ", ".join(f"{k} {v[0]:.2f}/{v[1]}" for k, v in det_table[m]["greedy"].get("exact_by_kind", {}).items())
        for m in det_table))
    P("  exact = share of prompts whose action (norm_action, dialect-aware) is identical across all 3 runs; jaccard = mean pairwise "
      "token-Jaccard of the actions; reason= = share with byte-identical reasoning text; workers = mean distinct x_engy.worker ids "
      "per prompt; 'T0 vs T0.8' = share of fresh T=0.8 samples whose action equals a greedy action (what memorising a published greedy target buys).")
    P("")

    # ---- format compliance
    P("5. FORMAT COMPLIANCE per dialect per model (all T=0.8 samples; parsed = (z,y) via reply_to_rollout in the turn's dialect; "
      "parsed_live = live split_rollout with require_think_close + text fallback at tool turns)")
    P(f"  {'model':20s} {'dialect':14s} {'n':>4s} {'parsed':>6s} {'live':>6s} {'repaired':>8s} {'think_closed':>12s} {'len_cap':>7s} {'err':>4s} {'out_tok p50':>11s}")
    comp_table = {}
    for m in MODELS:
        for k in DIALECTS:
            # error rate over the retrying tags only (the retries=1 throughput
            # runs count their 429s in section 3, not here)
            rs = [r for r in rows if r["model"] == m and r["kind"] == k and r["temperature"] > 0
                  and not r["tag"].startswith("tp")]
            o = [r for r in rows if r["model"] == m and r["kind"] == k and r["temperature"] > 0 and r.get("ok")]
            if not o:
                continue
            e = {"n": len(rs), "n_ok": len(o), "parse_rate": st.mean(r["parsed"] for r in o) if o else None,
                 "parse_rate_live": st.mean(r["parsed_live"] for r in o) if o else None,
                 "repair_rate": st.mean(r["repaired"] for r in o) if o else None,
                 "think_close_rate": st.mean(r["think_closed"] for r in o) if o else None,
                 "length_cap_rate": st.mean(r["finish"] == "length" for r in o) if o else None,
                 "error_rate": (1 - sum(1 for r in rs if r.get("ok")) / len(rs)) if rs else None,
                 "completion_tokens_p50": pct([r["completion_tokens"] for r in o if r.get("completion_tokens") is not None], .5)}
            comp_table.setdefault(m, {})[k] = e
            P(f"  {m:20s} {k:14s} {len(o):4d} {e['parse_rate'] if o else float('nan'):6.2f} {e['parse_rate_live'] if o else float('nan'):6.2f} "
              f"{e['repair_rate'] if o else float('nan'):8.2f} {e['think_close_rate'] if o else float('nan'):12.2f} "
              f"{e['length_cap_rate'] if o else float('nan'):7.2f} {e['error_rate'] if e['error_rate'] is not None else float('nan'):4.2f} {e['completion_tokens_p50'] or 0:11.0f}")
    # teacher reference for compliance: from the verdicts' teacher by_dialect
    P("  teacher (live refs, same slices): mean refs/turn per dialect from the verdicts:")
    for rec in VERDICTS:
        d = C.load_verdict(rec)
        tb = d["verdict"]["teacher"].get("by_dialect") or {}
        P(f"    {rec}: " + ", ".join(f"{k} {v.get('mean_refs', 0):.2f}/3 (zero-ref {v.get('zero_ref_turns')}/{v.get('n_turns')})" for k, v in tb.items()))
    # error taxonomy
    errs = collections.Counter((r["model"], r.get("error", "")[:80]) for r in rows if not r.get("ok"))
    if errs:
        P("  errors:")
        for (m, e), n in errs.most_common(20):
            P(f"    {n:4d} {m:20s} {e}")
    P("")
    tables = {"shape": shape, "cost": cost_table, "throughput": tps, "determinism": det_table,
              "compliance": comp_table, "ledger": L}
    (RESULTS / "tables.json").write_text(json.dumps(tables, indent=1, default=str))
    (RESULTS / "measurements.txt").write_text("\n".join(out_lines) + "\n")
    print("\n".join(out_lines))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shape")
    s = sub.add_parser("sample"); s.add_argument("--model", required=True); s.add_argument("--n", type=int, default=100); s.add_argument("--conc", type=int, default=16)
    s = sub.add_parser("throughput"); s.add_argument("--model", required=True); s.add_argument("--conc", type=int, default=32); s.add_argument("--n", type=int, default=200); s.add_argument("--retries", type=int, default=1); s.add_argument("--redo", type=int, default=0)
    s = sub.add_parser("greedy"); s.add_argument("--pass", dest="pass_no", type=int, required=True); s.add_argument("--conc", type=int, default=16); s.add_argument("--models", default=",".join(MODELS))
    s = sub.add_parser("stochastic"); s.add_argument("--conc", type=int, default=16); s.add_argument("--models", default=",".join(MODELS))
    s = sub.add_parser("topup"); s.add_argument("--conc", type=int, default=16); s.add_argument("--models", default=",".join(MODELS)); s.add_argument("--target", type=int, default=30)
    sub.add_parser("report")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        if a.cmd == "shape":
            step_shape()
        elif a.cmd == "sample":
            step_sample(a.model, a.n, a.conc)
        elif a.cmd == "throughput":
            step_throughput(a.model, a.conc, a.n, a.retries, a.redo)
        elif a.cmd == "greedy":
            step_greedy(a.pass_no, a.conc, a.models.split(","))
        elif a.cmd == "stochastic":
            step_stochastic(a.conc, a.models.split(","))
        elif a.cmd == "topup":
            step_topup(a.conc, a.models.split(","), a.target)
        elif a.cmd == "report":
            step_report()
    except Budget as e:
        print(f"[budget] {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[ledger] ${ledger()['cost_usd']:.2f} spent so far")


if __name__ == "__main__":
    main()
