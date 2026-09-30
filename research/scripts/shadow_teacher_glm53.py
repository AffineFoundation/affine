"""Shadow re-score of stored wvk-22/23 verdicts under a candidate teacher
served by Engy (default ``glm-5.3-flash``), with the LIVE sd-meter rule.

Question (Jacob, 2026-09-23): if the teacher were GLM-5.3-Flash instead of
Qwen3.8-27B, (a) how do the recent duels' margins / z move, (b) does the
teacher-vs-king control turn positive again (headroom), (c) how do the
kings rank against the env totals?

Method: the stored artifacts freeze both sides' rollouts (z_a, y_a) and the
turn ids. For a seeded subset of turns per verdict we sample k fresh
references from the candidate teacher, run the production echo set (own /
empty / thought / cross / unconditioned / R / B / A) through the production
``evalsrv.terms`` code against an Engy-backed client, and score with the
production ``evalsrv.sdmeter`` functions. The live numbers are recomputed
from the stored rows on the SAME turns so both columns are like for like.

Only the transport is new: Engy scores spans from ``logprob_start_len`` on a
token-id prompt with a 1,024-token per-call cap, so long spans are echoed in
windows (logprobs of a token depend only on earlier tokens; windows compose
exactly). Rendering, splitting, and all math are the production modules.

Cost is read from Engy's ``x_engy.charged_micro`` per response; the run
stops cleanly at ``--budget-usd``.

Usage (repo root, venv active):
  python research/scripts/shadow_teacher_glm53.py run --smoke
  python research/scripts/shadow_teacher_glm53.py run --first 640 --last 670 \
      --turns-per-verdict 20 --budget-usd 45
  python research/scripts/shadow_teacher_glm53.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import math
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from affine.config import load_config  # noqa: E402
from affine.score import DEFAULT_CAUSALITY_TAU, b_gate_pass, teacher_causality  # noqa: E402
from evalsrv import sdmeter  # noqa: E402
from evalsrv.chat import (  # noqa: E402
    gen_prompt, get_tokenizer, set_thought_rendering, split_rollout,
    think_closed,
)
from evalsrv.corpus import CorpusSync  # noqa: E402
from evalsrv.dueling import ref_token_caps, token_caps, turn_id  # noqa: E402
from evalsrv.terms import miner_terms, score_teacher_rollouts, teacher_reference  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

log = logging.getLogger("shadow_glm53")

ENGY_BASE = "https://api.engy.ai/v1"
ECHO_MAX_TOKENS = 1024
EVALS_URL = "https://s3.hippius.com/affine-sn120/evals"
WORK = Path("/tmp/shadow")
EVALS_DIR = WORK / "evals"
OUT_DIR = REPO / "research/results/shadow_glm53"
CORPUS_DIR = WORK / "corpus_cache"

TEACHERS = {
    "glm-5.3-flash": "zai-org/GLM-5.3-Flash",
    "glm-5.3": "zai-org/GLM-5.3",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
}


class BudgetExceeded(RuntimeError):
    pass


class FatalRequestError(RuntimeError):
    """400s are deterministic for this prompt; do not retry."""


# -- Engy-backed teacher -------------------------------------------------------------

class EngyTeacher(VllmModel):
    """Production client semantics over Engy's completions API.

    Differences from a vLLM replica, all in transport: bearer auth, per-call
    cost accounting, token-id prompts, and span scoring via
    ``logprob_start_len`` in <= ECHO_MAX_TOKENS windows.
    """

    def __init__(self, engy_model: str, repo: str, key: str, concurrency: int,
                 budget_usd: float):
        cfg = Served(name="teacher", repo=repo, revision=None, port=0,
                     base_url=ENGY_BASE, model_name=engy_model)
        client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(600.0, connect=15.0),
            limits=httpx.Limits(max_connections=concurrency + 8,
                                max_keepalive_connections=concurrency + 8))
        super().__init__(cfg, client, asyncio.Semaphore(concurrency))
        self.budget_micro = int(budget_usd * 1e6)
        self.charged_micro = 0
        self.n_requests = 0
        self.n_429 = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.finish = {"stop": 0, "length": 0, "other": 0}

    def cost_usd(self) -> float:
        return self.charged_micro / 1e6

    async def _post(self, payload: dict) -> dict:
        if self.charged_micro >= self.budget_micro:
            raise BudgetExceeded(f"spent ${self.cost_usd():.2f}")
        payload = dict(payload)
        payload.pop("add_special_tokens", None)
        payload.pop("vllm_xargs", None)
        async with self.sem:
            self.in_flight += 1
            try:
                for attempt in range(6):
                    try:
                        r = await self.http.post(f"{self.base}/completions", json=payload)
                        if r.status_code == 400:
                            raise FatalRequestError(r.text[:300])
                        if r.status_code == 429:
                            self.n_429 += 1
                            await asyncio.sleep(3 * 2 ** attempt + random.random())
                            continue
                        r.raise_for_status()
                        d = r.json()
                        if not d.get("choices"):
                            raise httpx.HTTPError(f"200 without choices: {r.text[:200]}")
                        self._account(d)
                        return d
                    except FatalRequestError:
                        raise
                    except (httpx.HTTPError, json.JSONDecodeError):
                        if attempt == 5:
                            raise
                        await asyncio.sleep(3 * 2 ** attempt + random.random())
            finally:
                self.in_flight -= 1
        raise RuntimeError("engy retries exhausted")

    def _account(self, d: dict) -> None:
        self.n_requests += 1
        u = d.get("usage") or {}
        self.prompt_tokens += int(u.get("prompt_tokens") or 0)
        self.completion_tokens += int(u.get("completion_tokens") or 0)
        self.cached_tokens += int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.charged_micro += int((d.get("x_engy") or {}).get("charged_micro") or 0)

    async def sample(self, prefix_messages: list[dict], temperature: float,
                     max_tokens: int, *, sticky_key: str | None = None,
                     action_kind: str | None = None) -> tuple[str, str]:
        del sticky_key
        tok = get_tokenizer(self.cfg.repo, self.cfg.revision)
        prompt = gen_prompt(self.cfg.repo, self.cfg.revision, prefix_messages)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        d = await self._post({"model": self.cfg.request_model, "prompt": ids,
                              "max_tokens": max_tokens, "temperature": temperature})
        ch = d["choices"][0]
        fr = ch.get("finish_reason") or "other"
        self.finish[fr if fr in self.finish else "other"] += 1
        text = ch["text"]
        self.n_samples += 1
        self.n_think_closed += int(think_closed(text))
        return split_rollout(text, action_kind,
                             require_think_close=self.require_think_close,
                             text_fallback_at_tool_turns=self.text_fallback_at_tool_turns)

    async def _echo_span(self, full: str, span_start: int, span_bytes: int, *,
                         tokens: bool = False,
                         spans: list[tuple[int, int]] | None = None) -> dict:
        tok = get_tokenizer(self.cfg.repo, self.cfg.revision)
        enc = tok([full], add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"][0], enc["offset_mapping"][0]
        n_prompt = sum(1 for s, _ in offsets if s < span_start)
        raw: list[float | None] = []
        start = n_prompt
        while start < len(ids):
            end = min(start + ECHO_MAX_TOKENS - 1, len(ids))
            req_start = max(0, start - 1)
            d = await self._post({
                "model": self.cfg.request_model, "prompt": ids[:end],
                "max_tokens": 1, "temperature": 0, "echo": True,
                "logprobs": 1, "logprob_start_len": req_start})
            lp = d["choices"][0]["logprobs"]["token_logprobs"]
            window = lp[start - req_start:]
            got = [x for x in window if x is not None]
            expect = end - start
            if len(got) == expect + 1:     # generated token appended
                got = got[:expect]
            if len(got) != expect:
                raise RuntimeError(
                    f"echo window misaligned: expected {expect}, got {len(got)} "
                    f"(start={start} end={end} n={len(ids)})")
            raw.extend(got)
            start = end
        span_offsets = offsets[n_prompt:]
        raw_span: list[float | None] = list(raw)
        if spans is not None:
            keep = [any(a <= s < b for a, b in spans) for s, _ in span_offsets]
            raw_span = [x if k else None for x, k in zip(raw_span, keep)]
        span = [x for x in raw_span if x is not None]
        n_bytes = max(span_bytes, 1)
        out = {"sum_lp": sum(span), "n_tokens": len(span), "n_bytes": n_bytes,
               "lp_per_byte": sum(span) / n_bytes if span else 0.0}
        if tokens:
            out["tokens"] = [(s - span_start, e - span_start, x)
                             for (s, e), x in zip(span_offsets, raw_span) if x is not None]
        return out

    def stats(self) -> dict:
        return {"cost_usd": self.cost_usd(), "requests": self.n_requests,
                "http_429": self.n_429, "prompt_tokens": self.prompt_tokens,
                "cached_tokens": self.cached_tokens,
                "completion_tokens": self.completion_tokens,
                "samples": self.n_samples, "think_closed": self.n_think_closed,
                "finish": dict(self.finish)}


class LocalTeacher(VllmModel):
    """A vLLM replica (e.g. the GLM probe box) through the production client;
    same counters as EngyTeacher so run() can print them (cost = 0)."""

    def __init__(self, base_url: str, repo: str, model_name: str | None, concurrency: int):
        cfg = Served(name="teacher", repo=repo, revision=None, port=0,
                     base_url=base_url, model_name=model_name)
        client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=15.0),
                                   limits=httpx.Limits(max_connections=concurrency + 8,
                                                       max_keepalive_connections=concurrency + 8))
        super().__init__(cfg, client, asyncio.Semaphore(concurrency))
        self.n_429 = 0
        self.finish = {"stop": 0, "length": 0, "other": 0}

    def cost_usd(self) -> float:
        return 0.0

    def stats(self) -> dict:
        return {"cost_usd": 0.0, "requests": sum(v.get("requests", 0) for v in self.echo_stats().values()),
                "samples": self.n_samples, "think_closed": self.n_think_closed,
                "echo_stats": self.echo_stats()}


# -- data -------------------------------------------------------------------------------

def load_key() -> str:
    for name in ("ENGY_2", "ENGY", "ENGY_EVAL"):
        if os.environ.get(name):
            return os.environ[name]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ENGY"):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no Engy key (ENGY_2 / ENGY / ENGY_EVAL env, or .env)")


def fetch_artifact(chal: str) -> dict:
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    p = EVALS_DIR / f"{chal}.json.gz"
    if not p.exists():
        r = httpx.get(f"{EVALS_URL}/{chal}.json.gz", timeout=300, follow_redirects=True)
        r.raise_for_status()
        p.write_bytes(r.content)
    return json.load(gzip.open(p))


def pick_turns(art: dict, n: int, seed: int) -> list[str]:
    """Seeded subset of turn ids with a row on both sides (forfeits kept)."""
    k = {r["turn_id"] for r in art["king_rows"]}
    c = {r["turn_id"] for r in art["challenger_rows"]}
    both = sorted(k & c)
    rng = random.Random(seed)
    rng.shuffle(both)
    return both[:n]


def rows_by_tid(rows: list[dict]) -> dict[str, dict]:
    return {r["turn_id"]: r for r in rows}


# -- run --------------------------------------------------------------------------------

async def score_turn(teacher, rec: dict, duel_cfg: dict, sd: dict,
                     sides: dict[str, tuple[str, str] | None],
                     fixed_refs: list[tuple[str, str]] | None = None) -> dict:
    """Fresh refs (or `fixed_refs` re-echoed) + production echo set for the
    stored rollouts of each side. fixed_refs = the (z, y) of a previous pass:
    every echo is recomputed, nothing is resampled — the repeat-echo probe."""
    tid = turn_id(rec)
    kind = rec.get("action_kind")
    caps = token_caps(duel_cfg)
    ref_caps = ref_token_caps(duel_cfg)
    max_thought, max_action = caps(kind)
    ref_thought, ref_action = ref_caps(kind)
    temperature = float(duel_cfg["temperature"])
    k = int(duel_cfg["n_teacher_samples"])
    t0 = time.monotonic()
    if fixed_refs is not None:
        ref = await score_teacher_rollouts(
            teacher, rec["prefix"], fixed_refs, thought_echo=True, cross_echo=True,
            content_echo=True, content_lift_nats=sd["content_lift_nats"], sticky_key=tid)
    else:
        ref = await teacher_reference(
            teacher, rec["prefix"], k, temperature, ref_thought, ref_action,
            thought_echo=True, cross_echo=True, content_echo=True,
            content_lift_nats=sd["content_lift_nats"], sticky_key=tid,
            action_kind=kind)
    out = {"turn_id": tid, "action_kind": kind or dialects.DEFAULT_KIND,
           "refs": ref, "n_refs": len(ref), "rows": {}, "seconds": 0.0}
    if not ref:
        out["seconds"] = time.monotonic() - t0
        return out
    for side, ro in sides.items():
        if ro is None:
            out["rows"][side] = {"turn_id": tid, "valid": False}
            continue
        t = await miner_terms(
            teacher, None, rec["prefix"], ref, 1, temperature, max_thought,
            max_action, reason_only=True, causality_gate=True,
            thought_echo=True, action_echo=False, content_echo=True,
            content_lift_nats=sd["content_lift_nats"],
            content_prefix=sd["content_prefix"], shadow_action_echo=True,
            sticky_key=tid, action_kind=kind, rollouts=[ro])
        t["turn_id"] = tid
        out["rows"][side] = t
    out["seconds"] = time.monotonic() - t0
    return out


def side_rollout(row: dict) -> tuple[str, str] | None:
    if not (row.get("valid") and row.get("pairs")):
        return None
    p = row["pairs"][0]
    return (p.get("z_a") or "", p.get("y_a") or "")


async def run(args: argparse.Namespace) -> None:
    cfg = load_config()
    duel_cfg = dict(cfg.raw["duel"])
    sd = sdmeter.settings(duel_cfg)
    set_thought_rendering(str(duel_cfg.get("thought_rendering", "canonical")))
    repo = TEACHERS[args.teacher]
    tok = get_tokenizer(repo, None)
    if tok.chat_template is None:
        raise SystemExit(f"{repo} ships no chat_template")
    if args.base_url:
        teacher = LocalTeacher(args.base_url, repo, args.model_name, args.concurrency)
    else:
        teacher = EngyTeacher(args.teacher, repo, load_key(), args.concurrency, args.budget_usd)
    fixed: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if args.refs_from:
        for line in Path(args.refs_from).open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("refs"):
                fixed[(r["chal"], r["turn_id"])] = [(x["z"], x["y"]) for x in r["refs"]]
        log.info("re-echo mode: %d turns with fixed references from %s", len(fixed), args.refs_from)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_path = OUT_DIR / f"turns_{args.teacher}{tag}.jsonl"
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done.add((r["chal"], r["turn_id"]))
            except json.JSONDecodeError:
                continue

    corpus = CorpusSync(cfg.dataset.corpus_base_url, cfg.dataset.manifest_key,
                        CORPUS_DIR, lazy_chunks=True)
    if not corpus.ready:
        log.info("syncing corpus index from %s", cfg.dataset.corpus_base_url)
        corpus.refresh()
    index_rows = corpus.load_index_rows()
    by_tid = {turn_id(r): r for r in index_rows}
    log.info("corpus epoch %s, %d index rows", corpus.info()["corpus_epoch"], len(index_rows))

    chals = [f"chal-{i:05d}" for i in range(args.first, args.last + 1)]
    if args.extra:
        chals += [f"chal-{int(x):05d}" for x in args.extra.split(",")]
    jobs: list[tuple[str, str, dict, dict]] = []
    for chal in chals:
        try:
            art = fetch_artifact(chal)
        except httpx.HTTPError as e:
            log.warning("%s: no artifact (%s)", chal, e)
            continue
        if "king_rows" not in art or "challenger_rows" not in art:
            log.warning("%s: no duel rows (%s); skipped", chal,
                        art.get("rejection_reason") or "rejected/infra")
            continue
        n = args.turns_per_verdict if not args.smoke else 2
        seed = int(art["slice"]["seed"]) ^ 0x5A5A
        tids = pick_turns(art, n, seed)
        krows, crows = rows_by_tid(art["king_rows"]), rows_by_tid(art["challenger_rows"])
        for tid in tids:
            if (chal, tid) in done:
                continue
            if tid not in by_tid:
                log.warning("%s %s: turn not in current corpus index; skipped", chal, tid)
                continue
            if fixed and (chal, tid) not in fixed:
                continue
            jobs.append((chal, tid, krows[tid], crows[tid]))
        if args.smoke and len(jobs) >= 2:
            break
    log.info("%d turns to score (%d already done)", len(jobs), len(done))

    lock = asyncio.Lock()
    n_done = 0
    t_start = time.monotonic()

    async def one(chal: str, tid: str, krow: dict, crow: dict) -> None:
        nonlocal n_done
        rec = corpus.materialize_turns([by_tid[tid]])[0]
        rec["action_kind"] = by_tid[tid].get("action_kind") or rec.get("action_kind")
        try:
            res = await score_turn(teacher, rec, duel_cfg, sd,
                                   {"king": side_rollout(krow),
                                    "challenger": side_rollout(crow)},
                                   fixed_refs=fixed.get((chal, tid)))
        except BudgetExceeded:
            raise
        except FatalRequestError as e:
            res = {"turn_id": tid, "error": f"400:{e}", "refs": [], "rows": {}}
        except Exception as e:  # keep the run alive; report per turn
            log.warning("%s %s failed: %s: %s", chal, tid, type(e).__name__, str(e)[:200])
            res = {"turn_id": tid, "error": f"{type(e).__name__}:{str(e)[:200]}",
                   "refs": [], "rows": {}}
        res.update({"chal": chal, "teacher": args.teacher,
                    "king_digest": art_king(chal), "cost_usd_so_far": teacher.cost_usd()})
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(res) + "\n")
            n_done += 1
            if n_done % 5 == 0 or args.smoke:
                el = time.monotonic() - t_start
                log.info("%d/%d turns  $%.2f  %.1f s/turn  429s=%d  finish=%s",
                         n_done, len(jobs), teacher.cost_usd(), el / max(n_done, 1),
                         teacher.n_429, teacher.finish)

    sem = asyncio.Semaphore(args.turn_concurrency)

    async def guarded(job):
        async with sem:
            await one(*job)

    try:
        await asyncio.gather(*[guarded(j) for j in jobs])
    except BudgetExceeded as e:
        log.warning("budget reached: %s", e)
    stats = teacher.stats()
    stats.update({"turns_scored": n_done, "wall_s": time.monotonic() - t_start})
    (OUT_DIR / f"run_stats_{args.teacher}{tag}.json").write_text(json.dumps(stats, indent=1, default=str))
    log.info("done: %s", json.dumps(stats))
    await teacher.http.aclose()


_KING_CACHE: dict[str, str] = {}


def art_king(chal: str) -> str:
    if chal not in _KING_CACHE:
        art = fetch_artifact(chal)
        _KING_CACHE[chal] = str(art["request"].get("king_revision") or "")[:12]
    return _KING_CACHE[chal]


# -- analyze ------------------------------------------------------------------------------

def _paired_stats(c_scores: dict, k_scores: dict, cfg: dict) -> dict:
    return sdmeter._paired(c_scores, k_scores, cfg)


def teacher_control(turn_refs: dict[str, list[dict]], sigma_by_kind: dict,
                    kind_by_tid: dict, tau: float, cfg: dict) -> dict[str, dict]:
    """Held-out reference j scored as a miner against the other k-1 refs
    (production sdmeter.shadow_verdict.teacher_scores, loo anchor)."""
    out = {}
    a_norm = cfg["a_norm_bytes"]
    for tid, refs in turn_refs.items():
        t = sdmeter.ref_loo_terms(refs, tau, a_norm)
        if t is None:
            continue
        sig = sigma_by_kind.get(kind_by_tid.get(tid) or dialects.DEFAULT_KIND)
        per = []
        for j in range(len(refs)):
            legs = {"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                    "n_content": refs[j].get("n_content_thought"),
                    "n_tokens": refs[j].get("n_tokens_thought")}
            others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None]
                      for leg in ("R", "A", "Mc")}
            mu = {leg: st.mean(v) for leg, v in others.items() if v}
            s = sdmeter.turn_score(legs, mu, sig, cfg)
            if s["score"] is not None:
                per.append(s)
        if per:
            out[tid] = {"score": st.mean(p["score"] for p in per),
                        "bind": max(("R", "Gc", "A"), key=lambda b: sum(1 for p in per if p["bind"] == b)),
                        "z_R": sdmeter._mean([p["z_R"] for p in per]),
                        "typ_c": sdmeter._mean([p["typ_c"] for p in per]),
                        "z_A": sdmeter._mean([p["z_A"] for p in per]), "dropped": []}
    return out


def score_group(turn_refs: dict, rows_c: dict, rows_k: dict, kind_by_tid: dict,
                tau: float, cfg: dict, sigma_override: dict | None = None) -> dict:
    """One 'duel' worth of turns: anchors, side scores, paired stats, control."""
    a_norm = cfg["a_norm_bytes"]
    loo = sdmeter.loo_anchors(turn_refs, kind_by_tid, tau, a_norm)
    sigma = sigma_override or loo.sigma

    def side_scores(rows: dict) -> dict:
        out = {}
        for tid, r in rows.items():
            legs = sdmeter.side_legs(r, tau, a_norm)
            kind = kind_by_tid.get(tid) or dialects.DEFAULT_KIND
            out[tid] = sdmeter.turn_score(legs, loo.mu.get(tid), sigma.get(kind), cfg)
        return out

    c_scores, k_scores = side_scores(rows_c), side_scores(rows_k)
    t_scores = teacher_control(turn_refs, sigma, kind_by_tid, tau, cfg)
    paired = _paired_stats(c_scores, k_scores, cfg)
    t_vs_k = _paired_stats(t_scores, {t: s for t, s in k_scores.items() if t in t_scores}, cfg)
    # The held-out reference scores forfeit_sd when it carries < content_min_tokens
    # content tokens (a teacher that answers a wrap-up turn without thinking).
    # That floor is a rule artefact, not headroom: report the control on the
    # turns where neither side sits at the floor as well.
    floor = cfg["forfeit_sd"] + 1.0
    nf = {t: s for t, s in t_scores.items()
          if s["score"] > floor and t in k_scores and k_scores[t]["score"] is not None
          and k_scores[t]["score"] > floor}
    t_vs_k_nf = _paired_stats(nf, {t: k_scores[t] for t in nf}, cfg)
    n_t_floor = sum(1 for s in t_scores.values() if s["score"] <= floor)
    t_vs_c = _paired_stats(t_scores, {t: s for t, s in c_scores.items() if t in t_scores}, cfg)

    def leg_means(scores: dict) -> dict:
        valid = [s for s in scores.values() if s["bind"] != "forfeit" and s["score"] is not None]
        return {"mean": sdmeter._mean([s["score"] for s in valid]),
                "z_R": sdmeter._mean([s["z_R"] for s in valid]),
                "typ_c": sdmeter._mean([s["typ_c"] for s in valid]),
                "z_A": sdmeter._mean([s["z_A"] for s in valid]),
                "n_valid": len(valid),
                "bind": {b: sum(1 for s in valid if s["bind"] == b) / len(valid) if valid else None
                         for b in ("R", "Gc", "A")}}

    def paired_leg(t_scores: dict, k_scores: dict, leg: str) -> dict:
        d = [t_scores[t][leg] - k_scores[t][leg] for t in t_scores
             if t in k_scores and t_scores[t].get(leg) is not None and k_scores[t].get(leg) is not None]
        if len(d) < 2:
            return {"margin": None, "z": None, "n": len(d)}
        se = st.stdev(d) / math.sqrt(len(d))
        return {"margin": st.mean(d), "z": st.mean(d) / se if se > 0 else None, "n": len(d)}

    def b_pass(rows: dict) -> dict:
        """B licence (causality_tau per byte, leakage check) on the valid rows —
        the one per-byte knob the sd-meter still carries."""
        vals = [b_gate_pass(r["pairs"][0], DEFAULT_CAUSALITY_TAU) for r in rows.values()
                if r.get("valid") and r.get("pairs")]
        vals = [v for v in vals if v is not None]
        bs = [teacher_causality(r["pairs"][0]) for r in rows.values() if r.get("valid") and r.get("pairs")]
        bs = [b for b in bs if b is not None]
        return {"pass_rate": (sum(vals) / len(vals)) if vals else None, "n": len(vals),
                "mean_b": (st.mean(bs) if bs else None)}

    return {
        "n_turns": len(turn_refs), "n_loo_turns": len(loo.mu),
        "sigma_by_dialect": sigma, "mu_mean_by_dialect": loo.mu_mean,
        "b_gate": {"challenger": b_pass(rows_c), "king": b_pass(rows_k)},
        "challenger": leg_means(c_scores), "king": leg_means(k_scores),
        "teacher": leg_means(t_scores),
        "paired": paired,
        "teacher_vs_king": {**{k: t_vs_k.get(k) for k in ("margin", "se", "z", "n_paired_turns")},
                            "typ_c": paired_leg(t_scores, k_scores, "typ_c"),
                            "z_R": paired_leg(t_scores, k_scores, "z_R"),
                            "z_A": paired_leg(t_scores, k_scores, "z_A"),
                            "n_teacher_floor": n_t_floor,
                            "excl_floor": {k: t_vs_k_nf.get(k) for k in ("margin", "se", "z", "n_paired_turns")}},
        "teacher_vs_challenger": {k: t_vs_c.get(k) for k in ("margin", "se", "z", "n_paired_turns")},
        "_scores": {"c": c_scores, "k": k_scores, "t": t_scores},
    }


def analyze(args: argparse.Namespace) -> None:
    import tomllib
    duel_cfg = tomllib.loads((REPO / "affine/affine.toml").read_text())["duel"]
    cfg = sdmeter.settings(duel_cfg)
    tau = float(duel_cfg["tau"])
    path = OUT_DIR / f"turns_{args.teacher}.jsonl"
    recs = [json.loads(l) for l in path.open() if l.strip()]
    recs = [r for r in recs if not r.get("error") and r.get("refs")]
    by_chal: dict[str, list[dict]] = {}
    for r in recs:
        by_chal.setdefault(r["chal"], []).append(r)
    arts = {chal: fetch_artifact(chal) for chal in by_chal}

    # Shadow (candidate teacher) and live (stored) inputs on the same turns.
    shadow_refs, shadow_c, shadow_k = {}, {}, {}
    live_refs, live_c, live_k = {}, {}, {}
    kind_by_tid: dict[str, str] = {}
    chal_of: dict[str, str] = {}
    ref_yield = []
    for chal, rs in by_chal.items():
        art = arts[chal]
        lk, lc = rows_by_tid(art["king_rows"]), rows_by_tid(art["challenger_rows"])
        for r in rs:
            tid = r["turn_id"]
            key = f"{chal}|{tid}"
            chal_of[key] = chal
            kind_by_tid[key] = r["action_kind"]
            ref_yield.append(len(r["refs"]))
            shadow_refs[key] = r["refs"]
            shadow_c[key] = {**r["rows"].get("challenger", {"valid": False}), "turn_id": key}
            shadow_k[key] = {**r["rows"].get("king", {"valid": False}), "turn_id": key}
            if tid in art["teacher_refs"]:
                live_refs[key] = art["teacher_refs"][tid]
                live_c[key] = {**lc[tid], "turn_id": key}
                live_k[key] = {**lk[tid], "turn_id": key}

    pooled_shadow = score_group(shadow_refs, shadow_c, shadow_k, kind_by_tid, tau, cfg)
    pooled_live = score_group(live_refs, live_c, live_k, kind_by_tid, tau, cfg)

    # Per verdict, σ pooled over the whole run (30 turns is too few for a per-duel σ).
    per_verdict = []
    for chal in sorted(by_chal):
        keys = [k for k in shadow_refs if chal_of[k] == chal]
        sub = lambda d: {k: d[k] for k in keys if k in d}  # noqa: E731
        g_s = score_group(sub(shadow_refs), sub(shadow_c), sub(shadow_k), kind_by_tid, tau, cfg,
                          sigma_override=pooled_shadow["sigma_by_dialect"])
        g_l = score_group(sub(live_refs), sub(live_c), sub(live_k), kind_by_tid, tau, cfg,
                          sigma_override=pooled_live["sigma_by_dialect"])
        v = arts[chal]["verdict"]
        per_verdict.append({
            "chal": chal, "king_digest": str(arts[chal]["request"].get("king_revision"))[:12],
            "challenger_digest": str(arts[chal]["request"].get("challenger_revision"))[:12],
            "n_turns": len(keys),
            "live_full": {"margin": v.get("margin"), "z": v.get("z"), "n": v.get("n_paired_turns"),
                          "challenger_wins": v.get("challenger_wins"),
                          "teacher_vs_king_z": (((v.get("shadow") or {}).get("sd_meter") or {})
                                                .get("by_anchor", {}).get("loo", {})
                                                .get("teacher_vs_king", {}) or {}).get("z")},
            "live_subset": {"margin": g_l["paired"]["margin"], "z": g_l["paired"]["z"],
                            "teacher_vs_king_z": g_l["teacher_vs_king"]["z"]},
            "shadow": {"margin": g_s["paired"]["margin"], "z": g_s["paired"]["z"],
                       "teacher_vs_king_z": g_s["teacher_vs_king"]["z"],
                       "teacher_vs_king_typ_z": g_s["teacher_vs_king"]["typ_c"]["z"]},
        })

    # King ranking: mean sd-meter turn score of each king digest (as king) under both teachers.
    def king_means(group: dict, rows_k: dict) -> dict:
        by_king: dict[str, list[float]] = {}
        for key, s in group["_scores"]["k"].items():
            if s["score"] is None:
                continue
            kd = str(arts[chal_of[key]]["request"].get("king_revision"))[:12]
            by_king.setdefault(kd, []).append(s["score"])
        return {kd: {"mean": st.mean(v), "se": (st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None),
                     "n": len(v)} for kd, v in by_king.items()}

    kings_shadow = king_means(pooled_shadow, shadow_k)
    kings_live = king_means(pooled_live, live_k)

    stats_path = OUT_DIR / f"run_stats_{args.teacher}.json"
    run_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    def strip(g: dict) -> dict:
        return {k: v for k, v in g.items() if k != "_scores"}

    report = {
        "teacher": args.teacher, "n_turns": len(shadow_refs), "n_verdicts": len(by_chal),
        "ref_yield": {"mean_refs": st.mean(ref_yield) if ref_yield else None,
                      "frac_3": (sum(1 for x in ref_yield if x == 3) / len(ref_yield)) if ref_yield else None,
                      "frac_lt2": (sum(1 for x in ref_yield if x < 2) / len(ref_yield)) if ref_yield else None},
        "pooled": {"shadow": strip(pooled_shadow), "live_same_turns": strip(pooled_live)},
        "per_verdict": per_verdict,
        "kings": {"shadow": kings_shadow, "live_same_turns": kings_live},
        "run_stats": run_stats,
    }
    (OUT_DIR / f"report_{args.teacher}.json").write_text(json.dumps(report, indent=1, default=str))
    print(render(report))
    (OUT_DIR / f"report_{args.teacher}.txt").write_text(render(report))


def _f(x, w=7, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x is not None and math.isfinite(x) else f"{'-':>{w}}"


def render(rep: dict) -> str:
    L = []
    L.append(f"Shadow teacher {rep['teacher']} vs live Qwen3.8-27B — sd-meter min(z_R, typ_c, z_A), "
             f"{rep['n_turns']} turns from {rep['n_verdicts']} verdicts")
    rs = rep.get("run_stats") or {}
    L.append(f"cost ${rs.get('cost_usd', 0):.2f}  requests {rs.get('requests')}  prompt tok {rs.get('prompt_tokens')}  "
             f"cached {rs.get('cached_tokens')}  completion {rs.get('completion_tokens')}  wall {rs.get('wall_s', 0)/60:.0f} min  "
             f"finish {rs.get('finish')}")
    ry = rep["ref_yield"]
    L.append(f"reference yield: mean refs/turn {_f(ry['mean_refs'],5,2)}  all-3 {_f(ry['frac_3'],5,2)}  <2 {_f(ry['frac_lt2'],5,2)}")
    L.append("")
    for name, g in (("SHADOW " + rep["teacher"], rep["pooled"]["shadow"]),
                    ("LIVE Qwen3.8-27B (same turns)", rep["pooled"]["live_same_turns"])):
        L.append(f"== {name}: {g['n_turns']} turns, {g['n_loo_turns']} with LOO anchors")
        for side in ("teacher", "king", "challenger"):
            s = g[side]
            L.append(f"  {side:10s} mean {_f(s['mean'])}  z_R {_f(s['z_R'])}  typ_c {_f(s['typ_c'])}  z_A {_f(s['z_A'])}  "
                     f"n {s['n_valid']}  bind R/Gc/A {_f(s['bind']['R'],4,2)}/{_f(s['bind']['Gc'],4,2)}/{_f(s['bind']['A'],4,2)}")
        tk = g["teacher_vs_king"]
        L.append(f"  teacher − king (control): margin {_f(tk['margin'])} sd  z {_f(tk['z'],6,2)}  n {tk['n_paired_turns']}  |  "
                 f"typ_c leg margin {_f(tk['typ_c']['margin'])} z {_f(tk['typ_c']['z'],6,2)}  |  z_R leg {_f(tk['z_R']['margin'])} z {_f(tk['z_R']['z'],6,2)}  |  "
                 f"z_A leg {_f(tk['z_A']['margin'])} z {_f(tk['z_A']['z'],6,2)}")
        ef = tk["excl_floor"]
        L.append(f"  teacher − king excluding floor turns ({tk['n_teacher_floor']} teacher refs at the content floor): "
                 f"margin {_f(ef['margin'])} sd  z {_f(ef['z'],6,2)}  n {ef['n_paired_turns']}")
        p = g["paired"]
        L.append(f"  challenger − king (all verdicts pooled): margin {_f(p['margin'])}  z {_f(p['z'],6,2)}  n {p['n_paired_turns']}")
        bg = g["b_gate"]
        L.append(f"  B licence (tau {DEFAULT_CAUSALITY_TAU}/byte, gate 0.30): king pass {_f(bg['king']['pass_rate'],5,2)} mean B {_f(bg['king']['mean_b'],7,4)}  |  "
                 f"challenger pass {_f(bg['challenger']['pass_rate'],5,2)} mean B {_f(bg['challenger']['mean_b'],7,4)}")
        L.append(f"  sigma by dialect: " + ", ".join(f"{k}: R {_f(v.get('R'),6,4)} A {_f(v.get('A'),6,2)} Mc {_f(v.get('Mc'),6,3)}"
                                                    for k, v in sorted(g["sigma_by_dialect"].items())))
        L.append("")
    L.append("== per verdict (σ pooled over the run; live_full = the published 1000-turn verdict)")
    L.append(f"{'chal':11} {'king':12} {'n':>3} | {'live margin':>11} {'z':>6} | {'live subset':>11} {'z':>6} | {'GLM margin':>11} {'z':>6} | {'T−K z live':>10} {'T−K z GLM':>9} {'typ z GLM':>9}")
    for v in rep["per_verdict"]:
        lf, ls, sh = v["live_full"], v["live_subset"], v["shadow"]
        L.append(f"{v['chal']:11} {v['king_digest']:12} {v['n_turns']:>3} | {_f(lf['margin'],11)} {_f(lf['z'],6,2)} | "
                 f"{_f(ls['margin'],11)} {_f(ls['z'],6,2)} | {_f(sh['margin'],11)} {_f(sh['z'],6,2)} | "
                 f"{_f(ls['teacher_vs_king_z'],10,2)} {_f(sh['teacher_vs_king_z'],9,2)} {_f(sh['teacher_vs_king_typ_z'],9,2)}")
    L.append("")
    L.append("== kings as king: mean sd-meter turn score (sd units) — shadow vs live on the same turns")
    for kd in sorted(set(rep["kings"]["shadow"]) | set(rep["kings"]["live_same_turns"])):
        s = rep["kings"]["shadow"].get(kd, {}); l = rep["kings"]["live_same_turns"].get(kd, {})
        L.append(f"  {kd}: GLM {_f(s.get('mean'))} ± {_f(s.get('se'),5,3)} (n {s.get('n')})   live {_f(l.get('mean'))} ± {_f(l.get('se'),5,3)} (n {l.get('n')})")
    return "\n".join(L)


def repeat(args: argparse.Namespace) -> None:
    """Repeat-echo probe: two passes over the same turns with the SAME
    references (pass B re-echoed with --refs-from pass A). Reports the
    run-to-run repeat sd of every leg and of the turn score in teacher-sd
    units, the paired margin / z of each pass, and the extrapolated margin
    noise at n = 1000 against the verdict SE and δ."""
    import tomllib
    duel_cfg = tomllib.loads((REPO / "affine/affine.toml").read_text())["duel"]
    cfg = sdmeter.settings(duel_cfg)
    tau = float(duel_cfg["tau"])

    def load(path: str) -> dict[str, dict]:
        out = {}
        for line in Path(path).open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") or not r.get("refs"):
                continue
            out[f"{r['chal']}|{r['turn_id']}"] = r
        return out

    A, B = load(args.pass_a), load(args.pass_b)
    keys = sorted(set(A) & set(B))
    kinds = {k: A[k]["action_kind"] for k in keys}

    def score_pass(P: dict) -> dict:
        refs = {k: P[k]["refs"] for k in keys}
        rows_c = {k: {**P[k]["rows"].get("challenger", {"valid": False}), "turn_id": k} for k in keys}
        rows_k = {k: {**P[k]["rows"].get("king", {"valid": False}), "turn_id": k} for k in keys}
        g = score_group(refs, rows_c, rows_k, kinds, tau, cfg)
        return g

    ga, gb = score_pass(A), score_pass(B)

    def legs(g: dict, side: str) -> dict[str, dict]:
        return g["_scores"][side]

    def rep_sd(vals_a: list, vals_b: list) -> float | None:
        d = [b - a for a, b in zip(vals_a, vals_b) if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]
        return (st.stdev(d) / math.sqrt(2)) if len(d) > 2 else None  # per-pass sd of one measurement

    out = {"n_turns": len(keys)}
    for leg in ("score", "z_R", "typ_c", "z_A"):
        va, vb = [], []
        for side in ("k", "c"):
            sa, sb = legs(ga, side), legs(gb, side)
            for k in keys:
                a, b = sa.get(k, {}).get(leg), sb.get(k, {}).get(leg)
                # floor / forfeit rows are identical by construction; keep only scored rows
                if a is None or b is None or a <= cfg["forfeit_sd"] + 1 or b <= cfg["forfeit_sd"] + 1:
                    continue
                va.append(a); vb.append(b)
        out[f"repeat_sd_{leg}"] = rep_sd(va, vb)
        out[f"n_{leg}"] = len(va)
    # per-turn paired diffs (challenger - king) under each pass
    da, db = {}, {}
    for k in keys:
        for g, dst in ((ga, da), (gb, db)):
            c, kk = g["_scores"]["c"].get(k, {}).get("score"), g["_scores"]["k"].get(k, {}).get("score")
            if c is not None and kk is not None:
                dst[k] = c - kk
    common = sorted(set(da) & set(db))
    d_a = [da[k] for k in common]; d_b = [db[k] for k in common]
    delta = [db[k] - da[k] for k in common]
    n = len(common)
    sd_d = st.stdev(d_a + d_b) if n > 2 else None
    sd_delta = st.stdev(delta) if n > 2 else None
    out.update({
        "paired_pass_a": {"margin": st.mean(d_a), "se": sd_d / math.sqrt(n), "z": st.mean(d_a) / (st.stdev(d_a) / math.sqrt(n))} if n > 2 else None,
        "paired_pass_b": {"margin": st.mean(d_b), "se": sd_d / math.sqrt(n), "z": st.mean(d_b) / (st.stdev(d_b) / math.sqrt(n))} if n > 2 else None,
        "margin_diff_between_passes_n": (st.mean(delta) if n else None),
        "sd_per_turn_paired_diff": sd_d,
        "sd_per_turn_pass_delta": sd_delta,
        # echo noise carried by a 1000-turn margin (one pass): sd(delta)/sqrt(2)/sqrt(1000)
        "echo_noise_se_margin_n1000": (sd_delta / math.sqrt(2) / math.sqrt(1000)) if sd_delta else None,
        "verdict_se_n1000": (sd_d / math.sqrt(1000)) if sd_d else None,
        "delta_sd": cfg["min_margin_sd"],
        # z shift a re-run would show at n=1000, in SE units
        "z_shift_n1000": ((sd_delta / math.sqrt(2)) / sd_d) if (sd_delta and sd_d) else None,
        "per_verdict": {},
    })
    for chal in sorted({k.split("|")[0] for k in common}):
        ks = [k for k in common if k.startswith(chal + "|")]
        if len(ks) < 3:
            continue
        ma, mb = st.mean(da[k] for k in ks), st.mean(db[k] for k in ks)
        out["per_verdict"][chal] = {"n": len(ks), "margin_a": ma, "margin_b": mb, "same_sign": (ma > 0) == (mb > 0)}
    rep_path = OUT_DIR / "repeat_echo_report.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(out, indent=1, default=str))
    L = [f"Repeat-echo probe: {n} turns, same references re-echoed twice",
         f"per-measurement repeat sd (teacher-sd units): turn score {_f(out['repeat_sd_score'])}  z_R {_f(out['repeat_sd_z_R'])}  typ_c {_f(out['repeat_sd_typ_c'])}  z_A {_f(out['repeat_sd_z_A'])}  (n {out['n_score']} scored side-turns)",
         f"paired challenger−king: pass A margin {_f(out['paired_pass_a']['margin'])} z {_f(out['paired_pass_a']['z'],6,2)} | pass B margin {_f(out['paired_pass_b']['margin'])} z {_f(out['paired_pass_b']['z'],6,2)} | mean shift {_f(out['margin_diff_between_passes_n'])} at n={n}",
         f"per-turn paired-diff sd {_f(out['sd_per_turn_paired_diff'])} | per-turn pass-to-pass delta sd {_f(out['sd_per_turn_pass_delta'])}",
         f"at n=1000: verdict SE {_f(out['verdict_se_n1000'],7,4)} | echo-noise SE on the margin {_f(out['echo_noise_se_margin_n1000'],7,4)} | z shift between re-runs ≈ {_f(out['z_shift_n1000'],5,2)} SE | δ {cfg['min_margin_sd']}",
         "per verdict (n≈20): " + ", ".join(f"{c}: {v['margin_a']:+.2f}/{v['margin_b']:+.2f}{'' if v['same_sign'] else ' SIGN'}" for c, v in out["per_verdict"].items())]
    print("\n".join(L))
    (OUT_DIR / "repeat_echo_report.txt").write_text("\n".join(L) + "\n")


def self_check(args: argparse.Namespace) -> None:
    """Re-derive a published verdict's margin / z / control from its stored
    rows with the same functions analyze() uses (σ per duel, as production).
    Validates the scoring side of this script without any teacher calls."""
    import tomllib
    duel_cfg = tomllib.loads((REPO / "affine/affine.toml").read_text())["duel"]
    cfg = sdmeter.settings(duel_cfg)
    tau = float(duel_cfg["tau"])
    for chal in [f"chal-{int(x):05d}" for x in args.extra.split(",") if x]:
        art = fetch_artifact(chal)
        kinds = {}
        for r in art["king_rows"]:
            kinds[r["turn_id"]] = dialects.DEFAULT_KIND
        # dialect per turn is not stored on rows; recover it from the corpus index
        cfgm = load_config()
        corpus = CorpusSync(cfgm.dataset.corpus_base_url, cfgm.dataset.manifest_key,
                            CORPUS_DIR, lazy_chunks=True)
        if not corpus.ready:
            corpus.refresh()
        idx = {turn_id(r): r for r in corpus.load_index_rows()}
        for tid in kinds:
            kinds[tid] = (idx.get(tid) or {}).get("action_kind") or dialects.DEFAULT_KIND
        g = score_group(art["teacher_refs"], rows_by_tid(art["challenger_rows"]),
                        rows_by_tid(art["king_rows"]), kinds, tau, cfg)
        v = art["verdict"]
        pub = (((v.get("shadow") or {}).get("sd_meter") or {}).get("by_anchor", {}).get("loo", {}))
        print(f"{chal}: published margin {v['margin']:+.5f} z {v['z']:+.3f} | recomputed margin "
              f"{g['paired']['margin']:+.5f} z {g['paired']['z']:+.3f} | control T−K z published "
              f"{(pub.get('teacher_vs_king') or {}).get('z')} recomputed {g['teacher_vs_king']['z']:+.3f} "
              f"(typ_c leg z {g['teacher_vs_king']['typ_c']['z']:+.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyze", "check", "repeat"])
    ap.add_argument("--teacher", default="glm-5.3-flash", choices=sorted(TEACHERS))
    ap.add_argument("--first", type=int, default=640)
    ap.add_argument("--last", type=int, default=670)
    ap.add_argument("--extra", default="", help="comma list of extra chal numbers")
    ap.add_argument("--turns-per-verdict", type=int, default=20)
    ap.add_argument("--budget-usd", type=float, default=45.0)
    ap.add_argument("--concurrency", type=int, default=12, help="Engy requests in flight")
    ap.add_argument("--turn-concurrency", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base-url", default="", help="score against a vLLM replica instead of Engy (e.g. http://127.0.0.1:8000/v1)")
    ap.add_argument("--model-name", default=None, help="served model name on that replica (default: the repo)")
    ap.add_argument("--refs-from", default="", help="re-echo mode: reuse the (z, y) references of this pass's jsonl")
    ap.add_argument("--out-tag", default="", help="suffix for the output files, e.g. pass1 / pass2")
    ap.add_argument("--pass-a", default="", help="repeat stage: first pass jsonl")
    ap.add_argument("--pass-b", default="", help="repeat stage: second pass jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "run":
        asyncio.run(run(args))
    elif args.stage == "check":
        self_check(args)
    elif args.stage == "repeat":
        repeat(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
