"""Frontier-rule probe (2026-09-06): can a frontier model's *actions* enter the
score without breaking what min(R,G) already guarantees?

Stand-in frontier F = engy `glm-5.2`. F only SAMPLES (z_F, y_F) on stored
duel turns; every new number is a Qwen3.8-27B teacher echo, so the ranked
quantity stays teacher-side (no lpF anywhere, no new attack surface).

    R_F  = centered tempered LME_j( lpC(y_F^j|z_A) − lpC(y_F^j|∅) )  frontier-target Reason
    A_F  = tempered LME_j( lpC(y_A|z_F^j) − lpC(y_A|∅) )              frontier-thought licence
    m_F  = lpC(z_F^j|x)                                              where F's thought sits in G's band

Stages (each resumable by done-key; all under --out):
    select   pick N turns per stored record (stratified by dialect), re-materialize
             prefixes from the record's pinned manifest      -> turns/<rec>.jsonl
    sample   engy glm-5.2: 3 rollouts @T=0.8 + 1 greedy per turn -> frontier/<rec>.jsonl (+ .done)
    echo     teacher echoes on the probe box                  -> echoes/<rec>.jsonl
    analyze  six tests over everything that landed            -> report.{json,txt}

    python research/scripts/frontier_rule_probe.py select --records chal-00286 chal-00287 ...
    python research/scripts/frontier_rule_probe.py sample --record chal-00286
    python research/scripts/frontier_rule_probe.py echo --record chal-00286 --pod swarm-t-eval-b200-8x-5
    python research/scripts/frontier_rule_probe.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))

import lium_api  # noqa: E402
from affine.config import load_config  # noqa: E402
from affine.score import leakage, reason  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402
from evalsrv.corpus import CorpusSync  # noqa: E402
from evalsrv.terms import EMPTY_THOUGHTS  # noqa: E402
from evalsrv.vllm_client import ModelPool, Served, VllmModel  # noqa: E402

OUT_DEFAULT = REPO / "research/results/frontier_rule_probe"
EVALS_DIR = REPO / "affine/state/evals"
SWARM_ENV = REPO / "ops/teacher-swarm/.swarm_env"
ENGY_BASE = "https://api.engy.ai/v1"
FRONTIER_MODEL = "glm-5.2"
N_FRONTIER = 3            # sampled frontier rollouts per turn (the "frontier refs")
FRONTIER_TEMP = 0.8       # same as the duel's miner/teacher sampling temperature
MAX_TOKENS = 1024 + 768   # duel cap: max_thought_tokens + max_action_tokens

# Attack variants (E6), applied to the king's real thought.
FILLER = ("\n\nBefore proceeding I will double-check every assumption, keep the "
          "change minimal and well-scoped, verify the result afterwards, and "
          "make sure nothing else in the environment is affected by this step.")
GENERIC_THOUGHT = ("Let me look at the relevant files to understand the current "
                   "state before making changes. I need to check the structure and "
                   "verify what has already been done, then decide the next step "
                   "carefully and keep the change small.")

RULE_NAMES = ("minRG", "RF", "minRGRF", "mix25", "mix50", "mix100",
              "add25", "add50", "pooled")


# ------------------------------------------------------------------ helpers
def lme(vals: list[float], tau: float) -> float:
    if not vals:
        return float("nan")
    if len(vals) == 1 or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def clme(vals: list[float], tau: float) -> float:
    """Centered tempered LME: exactly affine.score.centered_reason's rule."""
    if not vals:
        return float("nan")
    return lme(vals, tau) - st.mean(vals)


def band(ts: list[float], band_c: float, band_floor: float) -> tuple[float, float]:
    mu = st.mean(ts)
    sd = st.stdev(ts) if len(ts) >= 2 else 0.0
    return mu, max(band_c * sd, band_floor)


def g_leg(m: float, mu: float, w: float) -> float:
    return min(m - (mu - w), (mu + w) - m)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_swarm_key() -> str:
    for line in SWARM_ENV.read_text().splitlines():
        if line.startswith("SWARM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"SWARM_KEY missing from {SWARM_ENV}")


def probe_box_urls(pod_name: str) -> list[str]:
    """Replica base URLs of one swarm pod, straight from the Lium API
    (advertise=false boxes never appear in the router's state.json)."""
    sess = lium_api.session()
    pods = lium_api.pods(sess) or []
    pod = next((p for p in pods if lium_api.pod_name(p) == pod_name), None)
    if pod is None:
        raise SystemExit(f"pod {pod_name} not found on Lium")
    ip = lium_api.pod_ip(pod)
    ports = lium_api.data_ports(pod)
    urls = [f"http://{ip}:{ext}/v1" for internal, ext in sorted(ports.items())
            if 40000 <= internal < 40100]
    return healthy_urls(urls)


def healthy_urls(urls: list[str]) -> list[str]:
    """Keep replicas that answer /v1/models (Lium maps more ports than the
    bootstrap launches replicas; smoke 2026-09-06: 10 ports, 8 replicas, and
    the two dead URLs cost 5/20 turns to EngineUnreachable)."""
    headers = {"Authorization": f"Bearer {load_swarm_key()}"}
    out = []
    for u in urls:
        try:
            if httpx.get(f"{u}/models", headers=headers, timeout=6.0).status_code == 200:
                out.append(u)
        except httpx.HTTPError:
            pass
    return out


def not_forfeit(row: dict | None) -> bool:
    return bool(row and row.get("valid") and "pairs" in row)


def render_tool_call(call: dict) -> str:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    body = json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


# ------------------------------------------------------------------ select
def cmd_select(args) -> int:
    cfg = load_config()
    out = Path(args.out)
    (out / "turns").mkdir(parents=True, exist_ok=True)
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    rng = random.Random(args.seed)
    for rec in args.records:
        dst = out / "turns" / f"{rec}.jsonl"
        if dst.exists() and not args.force:
            print(f"{rec}: turns already selected ({sum(1 for _ in open(dst))}), skip")
            continue
        d = json.load(gzip.open(EVALS_DIR / f"{rec}.json.gz"))
        v = d["verdict"]
        refs = d["teacher_refs"]
        k_by = {r["turn_id"]: r for r in d["king_rows"]}
        c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
        tids = [t for t in d["turn_ids"]
                if len(refs.get(t) or []) == v["duel_params"]["n_teacher_samples"]
                and not_forfeit(k_by.get(t)) and not_forfeit(c_by.get(t))]
        sha = v["slice"]["manifest_sha256"]
        scratch = Path(f"/tmp/frontier_probe_corpus/{sha[:12]}")
        corpus = CorpusSync(cfg.dataset.corpus_base_url,
                            f"corpus/manifests/{sha}.json", scratch, lazy_chunks=True)
        if not corpus.ready:
            corpus.refresh()
        if not corpus.ready:
            raise SystemExit(f"{rec}: pinned manifest {sha[:12]} not syncable")
        rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
        by_kind: dict[str, list[str]] = {}
        for t in tids:
            by_kind.setdefault(rows[t]["action_kind"], []).append(t)
        # Proportional allocation (largest remainder), then seeded sampling.
        # Small dialects are floored at --min-per-kind (or all they have):
        # boxed turns forfeit often (token cap), so their eligible pool is
        # ~4% of a record and a proportional draw would leave ~15 turns.
        total = sum(len(x) for x in by_kind.values())
        quota = {k: args.n_per * len(x) / total for k, x in by_kind.items()}
        take = {k: int(q) for k, q in quota.items()}
        for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[
                : args.n_per - sum(take.values())]:
            take[k] += 1
        for k, pool in by_kind.items():
            floor = min(args.min_per_kind, len(pool))
            if take[k] < floor:
                big = max(take, key=take.get)
                take[big] -= floor - take[k]
                take[k] = floor
        picked: list[str] = []
        for k, pool in sorted(by_kind.items()):
            pool = sorted(pool)
            rng.shuffle(pool)
            picked += pool[: take[k]]
        turns = corpus.materialize_turns([rows[t] for t in picked])
        n_written = 0
        with open(dst, "w") as f:
            for t, tr in zip(picked, turns):
                f.write(json.dumps({
                    "record": rec, "turn_id": t, "kind": tr["action_kind"],
                    "source": tr.get("source"), "language": tr.get("language"),
                    "prefix": tr["prefix"], "refs": refs[t],
                    "king": k_by[t]["pairs"], "challenger": c_by[t]["pairs"],
                }) + "\n")
                n_written += 1
        meta[rec] = {
            "duel_params": v["duel_params"], "margin": v["margin"], "se": v["se"],
            "z": v["z"], "challenger_wins": v["challenger_wins"],
            "n_paired_turns": v["n_paired_turns"], "manifest_sha256": sha,
            "corpus_epoch": v["slice"]["corpus_epoch"],
            "king_repo": d["request"].get("king_repo"),
            "challenger_repo": d["request"].get("challenger_repo"),
            "eligible": len(tids), "picked": take,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{rec}: {n_written} turns picked {take} of {len(tids)} eligible "
              f"(manifest {sha[:12]}, epoch {v['slice']['corpus_epoch']})")
    return 0


# ------------------------------------------------------------------ sample
class Engy:
    def __init__(self, key: str, concurrency: int):
        self.cli = httpx.AsyncClient(base_url=ENGY_BASE, timeout=600.0,
                                     headers={"Authorization": f"Bearer {key}"})
        self.sem = asyncio.Semaphore(concurrency)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    async def chat(self, messages: list[dict], temperature: float) -> dict:
        """One chat completion -> the single choice. (engy ignores `n`, smoke
        2026-09-06: n=3 returned one choice; callers issue N separate calls.)"""
        payload = {"model": FRONTIER_MODEL, "messages": messages,
                   "max_tokens": MAX_TOKENS, "temperature": temperature}
        last = None
        for attempt in range(6):
            async with self.sem:
                try:
                    r = await self.cli.post("/chat/completions", json=payload)
                    if r.status_code == 200:
                        d = r.json()
                        if d.get("choices"):
                            u = d.get("usage") or {}
                            self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                            self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                            self.usage["calls"] += 1
                            return d["choices"][0]
                        last = f"200 without choices: {r.text[:200]}"
                    else:
                        last = f"HTTP {r.status_code}: {r.text[:200]}"
                except (httpx.HTTPError, ValueError) as e:
                    last = repr(e)
            await asyncio.sleep(3 * (attempt + 1) + random.random() * 2)
        raise RuntimeError(f"engy call failed after retries: {last}")


FUNC_OPEN_RE = re.compile(r"(?m)^function=")


def repair_tool_call(content: str) -> tuple[str, bool]:
    """Undo the engy server's tool-parser mangling of GLM's XML tool calls
    (smoke 2026-09-06, deterministic on every tool_call turn tried): the `<`
    of `<function=...>` is eaten and the closing `</tool_call>` is dropped,
    so the dialect parser sees no complete span. Repaired text is what the
    model emitted per the prompt's own format; flagged so the report can
    show the repair rate."""
    fixed = FUNC_OPEN_RE.sub("<function=", content)
    if fixed.count("<tool_call>") > fixed.count("</tool_call>") and \
            fixed.rstrip().endswith("</function>"):
        fixed = fixed.rstrip() + "\n</tool_call>"
    return fixed, fixed != content


def choice_to_rollout(choice: dict, kind: str) -> dict:
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if msg.get("tool_calls"):
        content = content.rstrip() + "\n" + "\n".join(
            render_tool_call(c) for c in msg["tool_calls"])
    repaired = False
    if kind == "tool_call":
        content, repaired = repair_tool_call(content)
    text = (msg.get("reasoning_content") or "") + "</think>\n" + content
    z, y = split_rollout(text, kind)
    return {"z": z, "y": y, "parsed": bool(y), "repaired": repaired,
            "finish": choice.get("finish_reason"),
            "structured_tool_calls": bool(msg.get("tool_calls")),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "content_chars": len(content)}


async def cmd_sample(args) -> int:
    out = Path(args.out)
    turns = read_jsonl(out / "turns" / f"{args.record}.jsonl")
    if not turns:
        raise SystemExit(f"no turns for {args.record}; run select first")
    if args.limit:
        turns = turns[: args.limit]
    dst = out / "frontier" / f"{args.record}.jsonl"
    done = {r["turn_id"] for r in read_jsonl(dst)}
    todo = [t for t in turns if t["turn_id"] not in done]
    key = os.environ.get(args.engy_env, "")
    if not key:
        raise SystemExit(f"{args.engy_env} not set (source .env)")
    engy = Engy(key, args.concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0
    print(f"{args.record}: sampling {len(todo)} turns ({len(done)} already done)")

    async def one(t: dict) -> None:
        nonlocal n_done
        try:
            *sampled, greedy = await asyncio.gather(
                *[engy.chat(t["prefix"], FRONTIER_TEMP) for _ in range(N_FRONTIER)],
                engy.chat(t["prefix"], 0.0))
            rec = {"turn_id": t["turn_id"], "record": args.record, "kind": t["kind"],
                   "samples": [choice_to_rollout(c, t["kind"]) for c in sampled],
                   "greedy": choice_to_rollout(greedy, t["kind"]),
                   "error": None}
        except Exception as e:  # noqa: BLE001 — one bad turn must not kill the run
            rec = {"turn_id": t["turn_id"], "record": args.record, "kind": t["kind"],
                   "samples": [], "greedy": None, "error": repr(e)[:300]}
        async with lock:
            append_jsonl(dst, rec)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(todo)} ({time.time() - t0:.0f}s) "
                      f"usage={engy.usage}", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    rows = read_jsonl(dst)
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(
            sum(1 for s in r["samples"] if s["parsed"]))
    print(f"{args.record}: {len(rows)} turns sampled in {time.time() - t0:.0f}s; "
          f"usage={engy.usage}; parsed of {N_FRONTIER} by kind: "
          + ", ".join(f"{k} {st.mean(v):.2f}" for k, v in sorted(by_kind.items())))
    if len(rows) >= len(turns):
        dst.with_suffix(".done").write_text(json.dumps(engy.usage))
    return 0


# ------------------------------------------------------------------ echo
async def cmd_echo(args) -> int:
    cfg = load_config()
    out = Path(args.out)
    turns = {t["turn_id"]: t for t in read_jsonl(out / "turns" / f"{args.record}.jsonl")}
    fr = {r["turn_id"]: r for r in read_jsonl(out / "frontier" / f"{args.record}.jsonl")}
    dst = out / "echoes" / f"{args.record}.jsonl"
    # Errored turns are NOT done: a rerun re-echoes them (analyze keeps the
    # successful line when both exist).
    done = {r["turn_id"] for r in read_jsonl(dst) if "lp" in r or r.get("skipped")}
    todo = [tid for tid in turns if tid in fr and tid not in done]
    if args.limit:
        todo = todo[: args.limit]
    urls = healthy_urls(args.urls.split(",")) if args.urls else probe_box_urls(args.pod)
    if not urls:
        raise SystemExit("no replica URLs")
    dp = json.loads((out / "meta.json").read_text())[args.record]["duel_params"]
    tau = dp["tau"]
    print(f"{args.record}: {len(todo)} turns to echo on {len(urls)} replicas "
          f"({len(done)} done)")
    headers = {"Authorization": f"Bearer {load_swarm_key()}"}
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0
    n_echo = 0

    async with httpx.AsyncClient(headers=headers) as http:
        per = max(1, math.ceil(args.concurrency / len(urls)))
        replicas = [VllmModel(Served(name="teacher", repo=cfg.teacher.repo, revision=None,
                                     port=0, base_url=u), http, asyncio.Semaphore(per))
                    for u in urls]
        pool = ModelPool(replicas)
        turn_sem = asyncio.Semaphore(max(2, args.concurrency // 6))

        async def lp_y(prefix, z, y, tid) -> float:
            nonlocal n_echo
            n_echo += 1
            return (await pool.score_action(prefix, z, y, sticky_key=tid))["lp_per_byte"]

        async def lp_z(prefix, z, tid) -> float:
            nonlocal n_echo
            n_echo += 1
            return (await pool.score_thought(prefix, z, sticky_key=tid))["lp_per_byte"]

        async def one(tid: str) -> None:
            nonlocal n_done
            t, f = turns[tid], fr[tid]
            F = [s for s in f["samples"] if s["parsed"]]
            G = f["greedy"] if f.get("greedy") and f["greedy"]["parsed"] else None
            if len(F) < 2:
                async with lock:
                    append_jsonl(dst, {"turn_id": tid, "record": args.record,
                                       "kind": t["kind"], "skipped": f"parsed={len(F)}"})
                    n_done += 1
                return
            F = F[:N_FRONTIER]
            x = t["prefix"]
            refs = t["refs"]
            zc = [r["z"] for r in refs]
            yc = [r["y"] for r in refs]
            kp, cp = t["king"][0], t["challenger"][0]
            zk, yk, zch, ych = kp["z_a"], kp["y_a"], cp["z_a"], cp["y_a"]
            yF = [s["y"] for s in F]
            zF = [s["z"] for s in F]
            variants = {"filler": zk + FILLER, "generic": GENERIC_THOUGHT,
                        "parrot": zk + "\n\n" + yF[0]}
            async with turn_sem:
                try:
                    jobs: dict[str, asyncio.Future] = {}

                    def add(name, coro):
                        jobs[name] = asyncio.ensure_future(coro)

                    for j, y in enumerate(yF):
                        add(f"yF_e.{j}", lp_y(x, EMPTY_THOUGHTS, y, tid))
                        add(f"king.f.{j}", lp_y(x, zk, y, tid))
                        add(f"chal.f.{j}", lp_y(x, zch, y, tid))
                        for i, z in enumerate(zc):
                            add(f"tref.{i}.f.{j}", lp_y(x, z, y, tid))
                        add(f"mF.{j}", lp_z(x, zF[j], tid))
                        add(f"AF.king.{j}", lp_y(x, zF[j], yk, tid))
                        add(f"AF.chal.{j}", lp_y(x, zF[j], ych, tid))
                        if j > 0:
                            add(f"fown.f.{j}", lp_y(x, zF[0], y, tid))
                    for i, y in enumerate(yc):
                        add(f"fown.a.{i}", lp_y(x, zF[0], y, tid))
                        if i > 0:
                            add(f"town.a.{i}", lp_y(x, zc[0], y, tid))
                    if G:
                        add("yG_e", lp_y(x, EMPTY_THOUGHTS, G["y"], tid))
                        add("king.fG", lp_y(x, zk, G["y"], tid))
                        add("chal.fG", lp_y(x, zch, G["y"], tid))
                    for vn, vz in variants.items():
                        add(f"var.{vn}.m", lp_z(x, vz, tid))
                        for i, y in enumerate(yc):
                            add(f"var.{vn}.a.{i}", lp_y(x, vz, y, tid))
                        for j, y in enumerate(yF):
                            add(f"var.{vn}.f.{j}", lp_y(x, vz, y, tid))
                    vals = dict(zip(jobs.keys(), await asyncio.gather(*jobs.values())))
                    rec = {"turn_id": tid, "record": args.record, "kind": t["kind"],
                           "n_frontier": len(F), "has_greedy": bool(G),
                           "lp": vals,
                           "len": {"yF": [len(y) for y in yF], "zF": [len(z) for z in zF],
                                   "yG": len(G["y"]) if G else None,
                                   "yk": len(yk), "ych": len(ych), "zk": len(zk),
                                   "zch": len(zch), "yc": [len(y) for y in yc]},
                           "leak": {"parrot_vs_yF0": leakage(variants["parrot"], yF[0]),
                                    "king_vs_yF0": leakage(zk, yF[0])},
                           "yF_all_same": len({y.strip() for y in yF}) == 1,
                           "exact": {"yF_eq_yC": [[a.strip() == b.strip() for b in yc] for a in yF],
                                     "yG_eq_yC": [G["y"].strip() == b.strip() for b in yc] if G else None}}
                except Exception as e:  # noqa: BLE001
                    rec = {"turn_id": tid, "record": args.record, "kind": t["kind"],
                           "error": repr(e)[:300]}
            async with lock:
                append_jsonl(dst, rec)
                n_done += 1
                if n_done % 20 == 0:
                    el = time.time() - t0
                    print(f"  {n_done}/{len(todo)} turns, {n_echo} echoes, "
                          f"{n_echo / max(el, 1):.1f}/s ({el:.0f}s)", flush=True)

        await asyncio.gather(*[one(tid) for tid in todo])
    print(f"{args.record}: echoed {n_done} turns / {n_echo} echoes in {time.time() - t0:.0f}s")
    return 0


# ------------------------------------------------------------------ analyze
def thought_scores(a: list[float], f: list[float], m: float, mu: float, w: float,
                   tau: float) -> dict[str, float]:
    """All candidate rules for one thought on one turn.
    a: per-teacher-ref Reason, f: per-frontier-ref Reason, m: lpC(z|x)."""
    R, RF, G = clme(a, tau), clme(f, tau), g_leg(m, mu, w)
    pooled = clme(a + f, tau)
    base = min(R, G)
    return {
        "R": R, "RF": RF, "G": G, "RF_u": lme(f, tau), "pooled_raw": pooled,
        "minRG": base,
        "RF_rule": RF,
        "minRGRF": min(R, G, RF),
        "mix25": min(0.75 * R + 0.25 * RF, G),
        "mix50": min(0.5 * R + 0.5 * RF, G),
        "mix100": min(RF, G),
        "add25": base + 0.25 * RF,
        "add50": base + 0.5 * RF,
        "pooled": min(pooled, G),
    }


RULES = {"minRG": "minRG", "RF": "RF_rule", "minRGRF": "minRGRF", "mix25": "mix25",
         "mix50": "mix50", "mix100": "mix100", "add25": "add25", "add50": "add50",
         "pooled": "pooled"}


def build_turn(t: dict, e: dict, tau: float, band_c: float, band_floor: float) -> dict:
    lp = e["lp"]
    refs = t["refs"]
    ts = [r["lp_thought"] for r in refs]
    mu, w = band(ts, band_c, band_floor)
    nF = e["n_frontier"]
    yF_e = [lp[f"yF_e.{j}"] for j in range(nF)]
    kp, cp = t["king"], t["challenger"]

    def f_of(prefix_key: str) -> list[float]:
        return [lp[f"{prefix_key}.{j}"] - yF_e[j] for j in range(nF)]

    # Per family: (a over teacher refs, f over frontier refs, m). The teacher's
    # own thought z_C^0 only has refs 1.. (leave-one-out); the frontier's own
    # thought z_F^0 only has frontier refs 1... `sides` scores every family on
    # all the refs it has (live rule for king/challenger); `sides_loo` drops
    # ref 0 on BOTH axes for every family so Test 1 compares like with like
    # (k−1 refs everywhere).
    fam: dict[str, tuple[list[float], list[float], float]] = {
        "king": ([reason(p) for p in kp], f_of("king.f"), kp[0]["lpC_za_x"]),
        "challenger": ([reason(p) for p in cp], f_of("chal.f"), cp[0]["lpC_za_x"]),
        "teacher_own": ([None] + [lp[f"town.a.{i}"] - refs[i]["lp_empty"] for i in range(1, len(refs))],
                        f_of("tref.0.f"), refs[0]["lp_thought"]),
        "frontier_own": ([lp[f"fown.a.{i}"] - refs[i]["lp_empty"] for i in range(len(refs))],
                         [None] + [lp[f"fown.f.{j}"] - yF_e[j] for j in range(1, nF)], lp["mF.0"]),
    }
    for vn in ("filler", "generic", "parrot"):
        fam[vn] = ([lp[f"var.{vn}.a.{i}"] - refs[i]["lp_empty"] for i in range(len(refs))],
                   [lp[f"var.{vn}.f.{j}"] - yF_e[j] for j in range(nF)], lp[f"var.{vn}.m"])
    sides, sides_loo = {}, {}
    for name, (a, f, m) in fam.items():
        a_full = [x for x in a if x is not None]
        f_full = [x for x in f if x is not None]
        sides[name] = thought_scores(a_full, f_full, m, mu, w, tau)
        sides_loo[name] = thought_scores(a[1:], f[1:], m, mu, w, tau)
    # A_F: does the frontier's thought license the miner's action
    AF = {"king": lme([lp[f"AF.king.{j}"] - kp[0]["lpC_ya_e"] for j in range(nF)], tau),
          "challenger": lme([lp[f"AF.chal.{j}"] - cp[0]["lpC_ya_e"] for j in range(nF)], tau)}
    # how well does *any* teacher thought predict the frontier action (mean over refs)
    tref_RF = st.mean(clme([lp[f"tref.{i}.f.{j}"] - yF_e[j] for j in range(nF)], tau)
                      for i in range(len(refs)))
    surprise = st.mean(yF_e) - st.mean(r["lp_empty"] for r in refs)   # <0: teacher finds y_F less likely than its own
    mF_in_band = st.mean(1.0 if g_leg(lp[f"mF.{j}"], mu, w) > 0 else 0.0 for j in range(nF))
    agree = any(any(row) for row in e["exact"]["yF_eq_yC"])
    yF_same = len(set(e["len"]["yF"])) == 1 and e.get("yF_all_same", None) is not False
    return {"turn_id": t["turn_id"], "record": t["record"], "kind": t["kind"],
            "sides": sides, "sides_loo": sides_loo, "AF": AF, "tref_RF": tref_RF,
            "surprise": surprise, "mF_in_band": mF_in_band,
            "mF_G": [g_leg(lp[f"mF.{j}"], mu, w) for j in range(nF)],
            "yF_exact_teacher": agree, "yF_all_same": e.get("yF_all_same", yF_same),
            "B_king": kp[0]["lpC_ya_za"] - kp[0]["lpC_ya_e"],
            "B_chal": cp[0]["lpC_ya_za"] - cp[0]["lpC_ya_e"],
            "leak": e["leak"], "len": e["len"], "has_greedy": e["has_greedy"],
            "greedy_f": ({"king": lp["king.fG"] - lp["yG_e"], "chal": lp["chal.fG"] - lp["yG_e"]}
                         if e["has_greedy"] else None)}


def margin_stats(rows: list[dict], rule: str, rng: random.Random, n_boot: int = 300) -> dict:
    key = RULES[rule]
    d = [r["sides"]["challenger"][key] - r["sides"]["king"][key] for r in rows]
    n = len(d)
    if n < 3:
        return {"n": n}
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(n)
    boots = []
    for _ in range(n_boot):
        boots.append(st.mean(d[rng.randrange(n)] for _ in range(n)))
    sign_stable = sum(1 for b in boots if (b > 0) == (m > 0)) / n_boot
    # split-half sign agreement
    agree = 0
    idx = list(range(n))
    for _ in range(200):
        rng.shuffle(idx)
        a = st.mean(d[i] for i in idx[: n // 2])
        b = st.mean(d[i] for i in idx[n // 2:])
        agree += (a > 0) == (b > 0)
    return {"n": n, "margin": m, "se": se, "z": m / se if se else float("nan"),
            "boot_se": st.pstdev(boots), "sign_stable": sign_stable,
            "split_half_agree": agree / 200}


def bind_fracs(rows: list[dict], side: str) -> dict:
    out = {"R": 0, "G": 0, "RF": 0}
    for r in rows:
        s = r["sides"][side]
        legs = {"R": s["R"], "G": s["G"], "RF": s["RF"]}
        out[min(legs, key=legs.get)] += 1
    n = max(len(rows), 1)
    return {k: v / n for k, v in out.items()}


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def cmd_analyze(args) -> int:
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    rows: list[dict] = []
    frontier_stats: dict[str, dict] = {}
    skipped = errors = 0
    for rec in sorted(meta):
        dp = meta[rec]["duel_params"]
        turns = {t["turn_id"]: t for t in read_jsonl(out / "turns" / f"{rec}.jsonl")}
        for f in read_jsonl(out / "frontier" / f"{rec}.jsonl"):
            s = frontier_stats.setdefault(f["kind"], {"turns": 0, "parsed": 0, "samples": 0,
                                                       "greedy_parsed": 0, "len_cap": 0,
                                                       "structured": 0, "repaired": 0,
                                                       "errors": 0})
            s["turns"] += 1
            if f.get("error"):
                s["errors"] += 1
                continue
            for smp in f["samples"]:
                s["samples"] += 1
                s["parsed"] += smp["parsed"]
                s["len_cap"] += smp["finish"] == "length"
                s["structured"] += smp["structured_tool_calls"]
                s["repaired"] += smp.get("repaired", False)
            s["greedy_parsed"] += bool(f["greedy"] and f["greedy"]["parsed"])
        by_tid: dict[str, dict] = {}
        for e in read_jsonl(out / "echoes" / f"{rec}.jsonl"):
            prev = by_tid.get(e["turn_id"])
            if prev is None or ("lp" in e and "lp" not in prev):
                by_tid[e["turn_id"]] = e
        for e in by_tid.values():
            if e.get("skipped"):
                skipped += 1
                continue
            if e.get("error"):
                errors += 1
                continue
            rows.append(build_turn(turns[e["turn_id"]], e, dp["tau"], dp["band_c"],
                                   dp["band_floor"]))
    if not rows:
        raise SystemExit("nothing to analyze yet")
    rng = random.Random(7)
    L: list[str] = []
    P = L.append
    P(f"Frontier-rule probe — {len(rows)} turns from {len(meta)} records "
      f"(skipped {skipped} with <2 parsed frontier samples, {errors} echo errors)")
    P(f"frontier = engy {FRONTIER_MODEL}; teacher echoes only; tau={meta[next(iter(meta))]['duel_params']['tau']}")
    P("")
    P("== Frontier sampling (Job A) ==")
    P(f"{'kind':10} {'turns':>6} {'parsed':>7} {'greedy ok':>10} {'len cap':>8} {'repaired':>9} {'errors':>7}")
    for k, s in sorted(frontier_stats.items()):
        P(f"{k:10} {s['turns']:6d} {s['parsed'] / max(s['samples'], 1):7.1%} "
          f"{s['greedy_parsed'] / max(s['turns'], 1):10.1%} {s['len_cap'] / max(s['samples'], 1):8.1%} "
          f"{s['repaired'] / max(s['samples'], 1):9.1%} {s['errors']:7d}")
    P("")

    def section(sub: list[dict], title: str) -> dict:
        res: dict = {"n": len(sub)}
        P(f"===== {title} (n={len(sub)}) =====")
        if len(sub) < 10:
            P("  too few turns")
            P("")
            return res
        # Test 1: ordering
        P("-- Test 1: ordering of thought families (mean per-turn score; every family on k−1 refs, leave-one-out) --")
        fams = ("frontier_own", "teacher_own", "king", "challenger", "generic", "filler", "parrot")
        P(f"{'rule':9} " + " ".join(f"{f[:11]:>11}" for f in fams))
        res["ordering"] = {}
        for rule, key in RULES.items():
            means = {f: st.mean(r["sides_loo"][f][key] for r in sub) for f in fams}
            res["ordering"][rule] = means
            P(f"{rule:9} " + " ".join(f"{means[f]:11.4f}" for f in fams))
        P("legs:     " + " ".join(f"{f[:11]:>11}" for f in fams))
        for leg in ("R", "RF", "G"):
            P(f"{leg:9} " + " ".join(f"{st.mean(r['sides_loo'][f][leg] for r in sub):11.4f}" for f in fams))
        P(f"frontier refs all identical on {sum(1 for r in sub if r['yF_all_same']) / len(sub):.0%} of turns "
          f"(centered R_F is 0 there by construction)")
        wins = {}
        for rule, key in RULES.items():
            wins[rule] = {
                "F>T": sum(1 for r in sub if r["sides_loo"]["frontier_own"][key] > r["sides_loo"]["teacher_own"][key]) / len(sub),
                "T>king": sum(1 for r in sub if r["sides_loo"]["teacher_own"][key] > r["sides_loo"]["king"][key]) / len(sub),
                "F>king": sum(1 for r in sub if r["sides_loo"]["frontier_own"][key] > r["sides_loo"]["king"][key]) / len(sub),
            }
        res["ordering_wins"] = wins
        P("paired win fractions: " + "; ".join(
            f"{rule}: F>T {w['F>T']:.0%} T>king {w['T>king']:.0%} F>king {w['F>king']:.0%}"
            for rule, w in wins.items()))
        P("")
        # Test 2 + 3: live pair margins, noise
        P("-- Test 2/3: challenger − king margin per rule (same turns), noise --")
        P(f"{'rule':9} {'margin':>9} {'SE':>8} {'z':>7} {'boot SE':>8} {'sign stab':>9} {'split agree':>11} "
          f"{'rho vs minRG':>12} {'SE ratio':>8}")
        res["margins"] = {}
        base = margin_stats(sub, "minRG", rng)
        d0 = [r["sides"]["challenger"]["minRG"] - r["sides"]["king"]["minRG"] for r in sub]
        for rule, key in RULES.items():
            ms = margin_stats(sub, rule, rng)
            d1 = [r["sides"]["challenger"][key] - r["sides"]["king"][key] for r in sub]
            ms["rho_vs_minRG"] = spearman(d0, d1)
            ms["se_ratio"] = ms["se"] / base["se"] if base.get("se") else float("nan")
            res["margins"][rule] = ms
            P(f"{rule:9} {ms['margin']:+9.5f} {ms['se']:8.5f} {ms['z']:+7.2f} {ms['boot_se']:8.5f} "
              f"{ms['sign_stable']:9.1%} {ms['split_half_agree']:11.1%} {ms['rho_vs_minRG']:12.3f} "
              f"{ms['se_ratio']:8.2f}")
        res["binds"] = {s: bind_fracs(sub, s) for s in ("king", "challenger")}
        for s in ("king", "challenger"):
            b = res["binds"][s]
            P(f"binding leg under min(R,G,RF) {s:10}: R {b['R']:.0%}  G {b['G']:.0%}  RF {b['RF']:.0%}")
        P("")
        # Test 4: attacks
        P("-- Test 4: attack variants of the king's thought (mean score; king real for reference) --")
        P(f"{'rule':9} {'king':>9} {'filler':>9} {'generic':>9} {'parrot':>9} {'filler<=0':>9} {'generic<=0':>10} {'parrot<king':>11}")
        res["attacks"] = {}
        for rule, key in RULES.items():
            vals = {f: [r["sides"][f][key] for r in sub] for f in ("king", "filler", "generic", "parrot")}
            row = {f: st.mean(v) for f, v in vals.items()}
            row["filler_le0"] = sum(1 for v in vals["filler"] if v <= 0) / len(sub)
            row["generic_le0"] = sum(1 for v in vals["generic"] if v <= 0) / len(sub)
            row["parrot_lt_king"] = sum(1 for a, b in zip(vals["parrot"], vals["king"]) if a < b) / len(sub)
            res["attacks"][rule] = row
            P(f"{rule:9} {row['king']:+9.4f} {row['filler']:+9.4f} {row['generic']:+9.4f} {row['parrot']:+9.4f} "
              f"{row['filler_le0']:9.0%} {row['generic_le0']:10.0%} {row['parrot_lt_king']:11.0%}")
        leak_parrot = sum(1 for r in sub if r["leak"]["parrot_vs_yF0"]) / len(sub)
        P(f"parrot flagged by B-leakage check: {leak_parrot:.0%}; parrot G mean "
          f"{st.mean(r['sides']['parrot']['G'] for r in sub):+.4f} (king G mean "
          f"{st.mean(r['sides']['king']['G'] for r in sub):+.4f})")
        P("")
        # Test 5: disagreement
        P("-- Test 5: teacher/frontier disagreement --")
        sur = [r["surprise"] for r in sub]
        q = pct(sur, 0.25)
        hi = [r for r in sub if r["surprise"] <= q]      # teacher most surprised by y_F
        lo = [r for r in sub if r["surprise"] > q]
        exact = sum(1 for r in sub if r["yF_exact_teacher"]) / len(sub)
        P(f"surprise = mean_j lpC(y_F^j|∅) − mean_i lpC(y_C^i|∅): median {st.median(sur):+.4f}, "
          f"p25 {q:+.4f}; frontier action exactly equals a teacher ref on {exact:.0%} of turns")
        P(f"frontier thought inside G band: {st.mean(r['mF_in_band'] for r in sub):.0%} of samples "
          f"(mean m_F G {st.mean(g for r in sub for g in r['mF_G']):+.4f})")
        P(f"teacher's own thoughts on frontier actions (tref R_F mean): {st.mean(r['tref_RF'] for r in sub):+.4f}")
        res["disagreement"] = {"exact_match": exact, "median_surprise": st.median(sur),
                               "mF_in_band": st.mean(r["mF_in_band"] for r in sub)}
        for name, part in (("most-surprising quartile", hi), ("rest", lo)):
            if len(part) < 5:
                continue
            k = RULES["RF"]
            P(f"  {name:24} n={len(part):4d}  R_F: F_own {st.mean(r['sides']['frontier_own'][k] for r in part):+.4f} "
              f"T_own {st.mean(r['sides']['teacher_own'][k] for r in part):+.4f} "
              f"king {st.mean(r['sides']['king'][k] for r in part):+.4f} "
              f"chal {st.mean(r['sides']['challenger'][k] for r in part):+.4f} | "
              f"m_F in band {st.mean(r['mF_in_band'] for r in part):.0%} | "
              f"A_F king {st.mean(r['AF']['king'] for r in part):+.4f} chal {st.mean(r['AF']['challenger'] for r in part):+.4f}")
            res["disagreement"][name] = {"n": len(part)}
        P(f"A_F overall: king {st.mean(r['AF']['king'] for r in sub):+.4f}, "
          f"challenger {st.mean(r['AF']['challenger'] for r in sub):+.4f}; "
          f"R_F vs action length rho (king): "
          f"{spearman([r['len']['yk'] for r in sub], [r['sides']['king']['RF'] for r in sub]):+.3f}")
        P("")
        return res

    report = {"n": len(rows), "frontier_sampling": frontier_stats,
              "all": section(rows, "ALL DIALECTS")}
    for kind in sorted({r["kind"] for r in rows}):
        report[kind] = section([r for r in rows if r["kind"] == kind], f"dialect {kind}")
    for rec in sorted(meta):
        sub = [r for r in rows if r["record"] == rec]
        if sub:
            m = meta[rec]
            P(f"record {rec}: live verdict margin {m['margin']:+.5f} z {m['z']:+.2f} "
              f"(challenger_wins={m['challenger_wins']})")
            report[rec] = section(sub, f"record {rec}")

    # Morning decision rule
    P("== Decision (per plan): pass = test1 ordering F>=T>miners, SE ratio <= 1.30, "
      "filler & generic <= 0, |z| >= 0.7·|z(minRG)| ==")
    verdict = {}
    A = report["all"]
    z0 = abs(A["margins"]["minRG"]["z"])
    for rule in RULES:
        o = A["ordering"][rule]
        ok_order = o["frontier_own"] >= o["teacher_own"] >= max(o["king"], o["challenger"]) \
            if rule != "minRG" else o["teacher_own"] >= max(o["king"], o["challenger"])
        ok_noise = A["margins"][rule]["se_ratio"] <= 1.30
        ok_attack = A["attacks"][rule]["filler"] <= 0 and A["attacks"][rule]["generic"] <= 0
        ok_z = abs(A["margins"][rule]["z"]) >= 0.7 * z0
        verdict[rule] = {"ordering": ok_order, "noise": ok_noise, "attacks": ok_attack,
                         "z_kept": ok_z, "pass": ok_order and ok_noise and ok_attack and ok_z}
        P(f"{rule:9} ordering {str(ok_order):5} noise {str(ok_noise):5} attacks {str(ok_attack):5} "
          f"z_kept {str(ok_z):5} -> {'PASS' if verdict[rule]['pass'] else 'fail'}")
    report["decision"] = verdict
    text = "\n".join(L) + "\n"
    (out / "report.txt").write_text(text)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "turn_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(text)
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--records", nargs="+", required=True)
    s.add_argument("--n-per", type=int, default=400)
    s.add_argument("--min-per-kind", type=int, default=60)
    s.add_argument("--seed", type=int, default=6)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("--record", required=True)
    s.add_argument("--concurrency", type=int, default=24)
    s.add_argument("--engy-env", default="ENGY_2")
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("echo")
    s.add_argument("--record", required=True)
    s.add_argument("--pod", default="swarm-t-eval-b200-8x-5")
    s.add_argument("--urls", default="", help="comma-separated replica base URLs (overrides --pod)")
    s.add_argument("--concurrency", type=int, default=48)
    s.add_argument("--limit", type=int, default=0)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "sample":
        return asyncio.run(cmd_sample(args))
    if args.cmd == "echo":
        return asyncio.run(cmd_echo(args))
    return cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
