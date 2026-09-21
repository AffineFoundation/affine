"""N1 stage-2 scorer — implicit process reward term on Set A / Set B.

    term(y | x) = lpC+(y | x) - lpC(y | x)     summed over the ACTION BODY tokens

C  = frozen teacher Qwen/Qwen3.8-27B (base endpoint), C+ = the success-conditioned
fine-tune (plus endpoint, or the same endpoint with a LoRA adapter name).
Rows come from setA_build.py / setB_build.py (prompt + candidate suffixes +
absolute span offsets); every unique full text is echoed ONCE per model
through an OpenAI-compatible /v1/completions call with the prompt sent as
TOKEN IDS (echo=True, logprobs=1, max_tokens=1) and the logprobs summed over
the tokens whose START offset lies inside the span (live evalsrv rule). All
three span levels (action / inner / body) are read off the same echo.

Stages (resumable; echoes are cached under research/results/frontier_arbiter/iprm/):
  echo     score_sets_echo_<set>_<side>.jsonl   one line per (text key, model)
  terms    setX_terms.jsonl                     one line per (row, candidate)
  report   setX_report.txt / .json

Set A report: paired teacher-held-out (each ref, and their mean) vs king vs
challenger vs `ls -la` vs repeat-last per turn — mean / SE / z / exact sign
test, overall and per dialect; AUC teacher-vs-king; Spearman of the term with
the live sd-meter legs (z_R, typ_c, z_A, score) turn by turn; the paired
challenger-king term margin next to the verdict's live z; base-side parity
against the stored live echoes under --rendering with_thought.
Set B report: per-model mean term and excess over the teacher refs' mean on
the same turns (teacher row = leave-one-out over its refs), Spearman vs the
centered bench axes total / agentic / agentic_no_tau2 / chat with permutation
p and bootstrap CI, exactly as panel.py did, on ALL turns and per dialect group.

Usage (once C+ is served):
  python research/scripts/frontier_arbiter/iprm_probe/score_sets.py \
      --set A --endpoint-base http://HOST:8000/v1 --endpoint-plus http://HOST:8001/v1
  python ... --set B --endpoint-base http://HOST:8000/v1 --lora-name cplus   # one vLLM, LoRA adapter
Dry run against the Engy teacher for the base side, plus side mocked:
  python ... --set A --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import panel as P  # noqa: E402  (spearman, permutation_p, boot_spearman_ci, mean_se, KINGS)

OUT = C.REPO / "research" / "results" / "frontier_arbiter" / "iprm"
LEVELS = ("action", "inner", "body")
SIDES = ("base", "plus")
A_ROLES = ("king", "challenger", "ref_mean", "ref_0", "ref_1", "ref_2", "attack_ls", "attack_repeat")
SET_FILES = {"A": ("setA_turns.jsonl", "setA_turns.jsonl.gz"), "B": ("setB_turns.jsonl", "setB_turns.jsonl.gz")}


# ------------------------------------------------------------------ io
def load_set(name: str, limit: int = 0, stride: int = 1) -> list[dict]:
    for fn in SET_FILES[name]:
        p = OUT / fn
        if p.exists():
            rows = C.read_jsonl(p)[::max(stride, 1)]
            return rows[:limit] if limit else rows
    raise SystemExit(f"set {name} not built: run set{name}_build.py")


def text_key(full: str) -> str:
    return hashlib.sha1(full.encode()).hexdigest()[:20]


# ------------------------------------------------------------------ echo backends
class Backend:
    """echo(ids, start) -> logprobs of ids[start:] (entry for token `start` is None)."""
    name = "?"

    async def echo(self, ids: list[int], start: int) -> list[float | None]:
        raise NotImplementedError


class EngyBackend(Backend):
    def __init__(self, model: str = C.TEACHER_ENGY, concurrency: int = 8):
        self.engy = C.Engy(concurrency=concurrency)
        self.model = model
        self.name = f"engy:{model}"

    async def echo(self, ids: list[int], start: int) -> list[float | None]:
        return await self.engy.echo_ids(ids, start, self.model)

    @property
    def cost_usd(self) -> float:
        return self.engy.cost_usd


class VllmBackend(Backend):
    """OpenAI-compatible vLLM: prompt as token ids, echo=True, logprobs=1,
    max_tokens=1 -> choices[0].logprobs.token_logprobs covers every prompt
    token (first None) plus the one sampled token, which is dropped."""

    def __init__(self, base_url: str, model: str | None, concurrency: int = 4, timeout: float = 1800.0,
                 api_key: str | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.cli = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.name = f"vllm:{base_url}:{model}"
        self.cost_usd = 0.0

    async def resolve_model(self) -> str:
        if self.model:
            return self.model
        r = await self.cli.get("/models")
        r.raise_for_status()
        self.model = r.json()["data"][0]["id"]
        self.name = f"vllm:{self.cli.base_url}:{self.model}"
        return self.model

    async def echo(self, ids: list[int], start: int) -> list[float | None]:
        payload = {"model": await self.resolve_model(), "prompt": ids, "max_tokens": 1, "echo": True,
                   "logprobs": 1, "temperature": 0.0}
        last = None
        for attempt in range(5):
            async with self.sem:
                try:
                    r = await self.cli.post("/completions", json=payload)
                    if r.status_code == 200:
                        lp = (r.json()["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
                        if len(lp) < len(ids):
                            raise RuntimeError(f"echo returned {len(lp)} logprobs for {len(ids)} prompt tokens "
                                               f"(server without prompt echo?)")
                        return lp[start:len(ids)]
                    if r.status_code in (400, 404, 413, 422):
                        raise RuntimeError(f"vllm {r.status_code}: {r.text[:300]}")
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                except (httpx.HTTPError, ValueError, KeyError) as e:
                    last = repr(e)[:200]
            await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"vllm echo failed: {last}")


class MockPlusBackend(Backend):
    """Dry-run stand-in for C+: the base logprobs plus seeded N(0, sigma) noise
    per token (so the pipeline's differencing, spans and stats are exercised
    end to end without a second model)."""

    def __init__(self, base_results: dict[str, list[float | None]], sigma: float = 0.05):
        self.base_results = base_results
        self.sigma = sigma
        self.name = f"mock:base+N(0,{sigma})"
        self.cost_usd = 0.0
        self.key: str | None = None

    async def echo(self, ids: list[int], start: int) -> list[float | None]:
        base = self.base_results[self.key]
        rng = random.Random(self.key)
        return [None if v is None else v + rng.gauss(0, self.sigma) for v in base]


# ------------------------------------------------------------------ jobs
def candidate_jobs(rows: list[dict], rendering: str, tok, max_tokens: int) -> tuple[dict[str, dict], dict]:
    """Unique full texts to echo. job = {key, ids, start, tok_levels: {level: [token idx]}}
    with token index lists per span level (token scored iff its start offset is
    inside a span). Candidates referencing each job are recorded on the rows."""
    jobs: dict[str, dict] = {}
    stats = collections.Counter()
    for row in rows:
        prompt = row["prompt"]
        for cand in row["candidates"]:
            full = prompt + cand["render"][rendering]["suffix"]
            key = text_key(full)
            cand["_key"] = key
            if key in jobs:
                stats["dedup"] += 1
                continue
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if len(ids) > max_tokens:
                cand["_skip"] = f"too_long:{len(ids)}"
                stats["too_long"] += 1
                continue
            spans = cand["abs"][rendering]
            levels = {"action": [tuple(spans["action"])], "inner": [tuple(spans["inner"])],
                      "body": [tuple(b) for b in spans["body"]]}
            tok_levels = {lv: [i for i, (a, _) in enumerate(offs) if any(s <= a < e for s, e in sp)]
                          for lv, sp in levels.items()}
            if not tok_levels["body"]:
                # tiny body swallowed by a token starting on the preceding byte: overlap rule
                tok_levels["body"] = [i for i, (a, b) in enumerate(offs)
                                      if any(a < e and b > s for s, e in levels["body"])]
                stats["body_overlap_fallback"] += 1
            first = min(i for lv in LEVELS for i in tok_levels[lv]) if tok_levels["action"] else len(ids) - 1
            jobs[key] = {"key": key, "ids": ids, "start": max(first - 1, 0), "tok_levels": tok_levels,
                         "n_bytes": {lv: sum(len(full[s:e].encode()) for s, e in sp) for lv, sp in levels.items()},
                         "n_tokens": len(ids)}
            stats["jobs"] += 1
    return jobs, dict(stats)


def echo_cache_path(set_name: str, side: str) -> Path:
    return OUT / f"score_sets_echo_{set_name}_{side}.jsonl"


async def run_echoes(jobs: dict[str, dict], backend: Backend, cache: Path, label: str,
                     concurrency: int = 8) -> dict[str, dict]:
    """{key: {"lps": [...], "start": int, "model": name}} for every job, from
    cache where present (same key + model), else echoed and appended."""
    done = {r["key"]: r for r in C.read_jsonl(cache) if r.get("model") == backend.name}
    todo = [j for k, j in jobs.items() if k not in done]
    print(f"[{label}] {backend.name}: {len(done)} cached, {len(todo)} to echo "
          f"({sum(j['n_tokens'] for j in todo) / 1e6:.2f}M prompt tokens)", flush=True)
    t0 = time.time()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    n = 0

    async def one(j: dict) -> None:
        nonlocal n
        async with sem:
            if isinstance(backend, MockPlusBackend):
                backend.key = j["key"]
            try:
                lps = await backend.echo(j["ids"], j["start"])
                rec = {"key": j["key"], "model": backend.name, "start": j["start"], "lps": lps, "error": None}
            except Exception as e:  # noqa: BLE001 — one failed echo must not kill the run
                rec = {"key": j["key"], "model": backend.name, "start": j["start"], "lps": None, "error": repr(e)[:300]}
        async with lock:
            C.append_jsonl(cache, rec)
            done[j["key"]] = rec
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  [{label}] {n}/{len(todo)} ({time.time() - t0:.0f}s) cost ${getattr(backend, 'cost_usd', 0.0):.4f}",
                      flush=True)

    await asyncio.gather(*[one(j) for j in todo])
    return done


def level_sums(job: dict, rec: dict) -> dict | None:
    """{level: {"lp": sum, "n_tok": k, "n_bytes": b}} from an echo record."""
    if not rec or rec.get("lps") is None:
        return None
    lps, start = rec["lps"], rec["start"]
    out = {}
    for lv in LEVELS:
        vals = []
        for i in job["tok_levels"][lv]:
            k = i - start
            if 0 <= k < len(lps) and lps[k] is not None:
                vals.append(lps[k])
        out[lv] = {"lp": sum(vals), "n_tok": len(vals), "n_bytes": job["n_bytes"][lv]}
    return out


# ------------------------------------------------------------------ terms
def compute_terms(rows: list[dict], jobs: dict[str, dict], echoes: dict[str, dict[str, dict]],
                  rendering: str, level: str) -> list[dict]:
    """One record per (row, candidate) with base/plus sums per level and the
    term at the chosen level (sum, per byte, per token)."""
    out = []
    for row in rows:
        for cand in row["candidates"]:
            key = cand.get("_key")
            if not key or key not in jobs:
                continue
            sums = {side: level_sums(jobs[key], echoes[side].get(key)) for side in SIDES}
            rec = {"set": row["set"], "turn_id": row["turn_id"], "record": row.get("record"),
                   "dialect": row["dialect"], "group": row.get("group"), "depth": row.get("depth"),
                   "model": cand.get("model") if row["set"] == "A" else row["model"],
                   "row_model": row.get("model"), "role": cand["role"], "y_kind": cand.get("y_kind"),
                   "body_kind": cand["rel"]["body_kind"], "rendering": rendering, "level": level, "key": key,
                   "sums": sums, "live": cand.get("live")}
            b, p = sums["base"], sums["plus"]
            if b and p and b[level]["n_tok"]:
                rec["lp_base"], rec["lp_plus"] = b[level]["lp"], p[level]["lp"]
                rec["term"] = p[level]["lp"] - b[level]["lp"]
                rec["term_per_byte"] = rec["term"] / max(b[level]["n_bytes"], 1)
                rec["term_per_tok"] = rec["term"] / b[level]["n_tok"]
                rec["n_tok"], rec["n_bytes"] = b[level]["n_tok"], b[level]["n_bytes"]
            else:
                rec["term"] = None
            out.append(rec)
    return out


# ------------------------------------------------------------------ stats
def sign_test_p(diffs: list[float]) -> float:
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def _fmt(x, w=8, p=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (w - 1) + "-"
    return f"{x:{w}.{p}f}"


def paired_line(name: str, diffs: list[float]) -> tuple[str, dict]:
    if len(diffs) < 2:
        return f"{name:34} n={len(diffs):4d}  -", {"n": len(diffs)}
    mu, se = P.mean_se(diffs)
    z = mu / se if se else float("nan")
    p = sign_test_p(diffs)
    frac = st.mean(d > 0 for d in diffs)
    d = {"n": len(diffs), "mean": mu, "se": se, "z": z, "sign_p": p, "frac_pos": frac}
    return (f"{name:34} n={len(diffs):4d}  mean {mu:+.4f} ± {se:.4f}  z {z:+5.2f}  frac>0 {frac:.3f}  sign p {p:.3g}"), d


def report_A(terms: list[dict], rows: list[dict], metric: str) -> tuple[str, dict]:
    L: list[str] = []
    Pn = L.append
    by_turn: dict[str, dict[str, float]] = collections.defaultdict(dict)
    live: dict[tuple[str, str], dict] = {}
    for t in terms:
        if t.get("term") is None:
            continue
        by_turn[t["turn_id"]][t["role"]] = t[metric]
        if t.get("live"):
            live[(t["turn_id"], t["role"])] = t["live"]
    for tid, d in by_turn.items():
        refs = [d[r] for r in ("ref_0", "ref_1", "ref_2") if r in d]
        if refs:
            d["ref_mean"] = st.mean(refs)
            d["ref_min"] = min(refs)
            d["ref_max"] = max(refs)
    dialect_of = {r["turn_id"]: r["dialect"] for r in rows}
    record_of = {r["turn_id"]: r["record"] for r in rows}
    rep: dict = {"metric": metric, "n_turns": len(by_turn)}
    Pn(f"===== SET A — N1 term ({metric}) per candidate; {len(by_turn)} turns with terms =====")
    Pn("term = lpC+(y|x) − lpC(y|x) over the chosen span; ref_mean = mean of the 3 teacher held-out actions on the turn")
    Pn("")
    Pn("-- per-role mean (unpaired) --")
    Pn(f"{'slice':16} {'n':>5} " + " ".join(f"{r:>13}" for r in A_ROLES))
    rep["role_means"] = {}

    def role_block(name: str, tids: list[str]) -> None:
        cells = {}
        for r in A_ROLES:
            xs = [by_turn[t][r] for t in tids if r in by_turn[t]]
            cells[r] = P.mean_se(xs) if xs else (None, None)
        rep["role_means"][name] = {r: {"mean": m, "se": s} for r, (m, s) in cells.items()}
        Pn(f"{name:16} {len(tids):5d} " + " ".join(f"{_fmt(cells[r][0], 7)}±{_fmt(cells[r][1], 5, 3).strip():>5}" for r in A_ROLES))

    all_tids = list(by_turn)
    role_block("ALL", all_tids)
    for k in sorted({dialect_of[t] for t in all_tids}):
        role_block("kind " + k, [t for t in all_tids if dialect_of[t] == k])

    PAIRS = (("ref_mean", "king"), ("ref_mean", "challenger"), ("challenger", "king"), ("ref_min", "king"),
             ("ref_mean", "attack_ls"), ("king", "attack_ls"), ("challenger", "attack_ls"),
             ("ref_mean", "attack_repeat"), ("king", "attack_repeat"), ("challenger", "attack_repeat"),
             ("ref_0", "ref_1"))
    Pn("")
    Pn("-- paired differences X − Y over turns where both exist (mean ± SE, z = mean/SE, exact two-sided sign test) --")
    rep["paired"] = {}
    for name, tids in [("ALL", all_tids)] + [("kind " + k, [t for t in all_tids if dialect_of[t] == k])
                                             for k in sorted({dialect_of[t] for t in all_tids})]:
        Pn(f"[{name}]")
        rep["paired"][name] = {}
        for x, y in PAIRS:
            diffs = [by_turn[t][x] - by_turn[t][y] for t in tids if x in by_turn[t] and y in by_turn[t]]
            line, d = paired_line(f"{x} − {y}", diffs)
            Pn("  " + line)
            rep["paired"][name][f"{x}-{y}"] = d
    Pn("")
    Pn("-- AUC (unpaired, pooled over turns): P(term_X > term_Y) --")
    rep["auc"] = {}
    for x, y in (("ref_0", "king"), ("ref_0", "challenger"), ("ref_0", "attack_ls"), ("king", "attack_ls"),
                 ("ref_0", "attack_repeat"), ("king", "attack_repeat")):
        xs = [by_turn[t][r] for t in all_tids for r in ("ref_0", "ref_1", "ref_2") if x == "ref_0" and r in by_turn[t]] \
            if x == "ref_0" else [by_turn[t][x] for t in all_tids if x in by_turn[t]]
        ys = [by_turn[t][y] for t in all_tids if y in by_turn[t]]
        a = auc(xs, ys)
        rep["auc"][f"{x if x != 'ref_0' else 'refs'}>{y}"] = {"auc": a, "n_x": len(xs), "n_y": len(ys)}
        Pn(f"  {(x if x != 'ref_0' else 'refs(all 3)'):14} > {y:14}  AUC {a:.3f}   (n {len(xs)} vs {len(ys)})")
    Pn("")
    Pn("-- attack rows vs the labelled candidates: fraction of turns where the attack beats the MEDIAN of {king, challenger, refs} --")
    rep["attack_beats_median"] = {}
    for atk in ("attack_ls", "attack_repeat"):
        beats = []
        for t in all_tids:
            d = by_turn[t]
            lab = [d[r] for r in ("king", "challenger", "ref_0", "ref_1", "ref_2") if r in d]
            if atk in d and lab:
                beats.append(d[atk] > st.median(lab))
        rep["attack_beats_median"][atk] = {"frac": st.mean(beats) if beats else None, "n": len(beats)}
        Pn(f"  {atk:14} beats median labelled action on {st.mean(beats) if beats else float('nan'):.3f} of {len(beats)} turns")

    Pn("")
    Pn("-- correlation with the LIVE sd-meter legs (Spearman over (turn, side) for king+challenger; refs LOO) --")
    rep["live_corr"] = {}
    for who, roles in (("miners (king+challenger)", ("king", "challenger")), ("teacher refs (LOO legs)", ("ref_0", "ref_1", "ref_2"))):
        Pn(f"[{who}]")
        rep["live_corr"][who] = {}
        for leg in ("z_R", "typ_c", "z_A", "score"):
            xs, ys = [], []
            for (tid, role), lv in live.items():
                if role not in roles or role not in by_turn.get(tid, {}):
                    continue
                sd = (lv or {}).get("sd") or {}
                v = sd.get(leg)
                if v is None or not math.isfinite(v):
                    continue
                xs.append(by_turn[tid][role])
                ys.append(v)
            if len(xs) >= 10:
                rho = P.spearman(xs, ys)
                p = P.permutation_p(xs, ys, rho, 2000)
                rep["live_corr"][who][leg] = {"rho": rho, "p": p, "n": len(xs)}
                Pn(f"  term vs live {leg:6}  rho {rho:+.3f}  perm p {p:.3f}  n {len(xs)}")
        for raw in ("lpC_ya_za", "lpC_ya_e"):
            xs, ys = [], []
            for (tid, role), lv in live.items():
                if role not in roles or role not in by_turn.get(tid, {}):
                    continue
                v = ((lv or {}).get("lp") or {}).get(raw) if role in ("king", "challenger") else \
                    ((lv or {}).get("lp") or {}).get({"lpC_ya_za": "lp_own", "lpC_ya_e": "lp_empty"}[raw])
                if v is None:
                    continue
                xs.append(by_turn[tid][role])
                ys.append(v)
            if len(xs) >= 10:
                rho = P.spearman(xs, ys)
                rep["live_corr"][who]["raw:" + raw] = {"rho": rho, "n": len(xs)}
                Pn(f"  term vs live {raw:10} (per-byte teacher lp)  rho {rho:+.3f}  n {len(xs)}")
    # paired diff vs live paired diff
    xs, ys = [], []
    for tid, d in by_turn.items():
        lk, lc = live.get((tid, "king")), live.get((tid, "challenger"))
        if "king" in d and "challenger" in d and lk and lc:
            sk, sc = (lk.get("sd") or {}).get("score"), (lc.get("sd") or {}).get("score")
            if sk is not None and sc is not None:
                xs.append(d["challenger"] - d["king"])
                ys.append(sc - sk)
    if len(xs) >= 10:
        rho = P.spearman(xs, ys)
        rep["live_corr"]["paired_chal_minus_king"] = {"rho": rho, "n": len(xs)}
        Pn(f"  (challenger − king) term vs (challenger − king) live sd score: rho {rho:+.3f}  n {len(xs)}")

    Pn("")
    Pn("-- per record: paired (challenger − king) term margin next to the verdict's live z --")
    meta_p = OUT / "setA_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"records": {}}
    rep["per_record"] = {}
    for rec in sorted({record_of[t] for t in all_tids}):
        diffs = [by_turn[t]["challenger"] - by_turn[t]["king"] for t in all_tids
                 if record_of[t] == rec and "king" in by_turn[t] and "challenger" in by_turn[t]]
        tref = [by_turn[t]["ref_mean"] - by_turn[t]["king"] for t in all_tids
                if record_of[t] == rec and "king" in by_turn[t] and "ref_mean" in by_turn[t]]
        m = meta["records"].get(rec, {})
        mu, se = P.mean_se(diffs)
        mu2, se2 = P.mean_se(tref)
        rep["per_record"][rec] = {"n": len(diffs), "chal_minus_king": mu, "se": se, "z": mu / se if se else None,
                                  "ref_minus_king": mu2, "se_ref": se2, "live_z": m.get("z"),
                                  "live_margin": m.get("margin"), "challenger_wins": m.get("challenger_wins")}
        Pn(f"  {rec}  n {len(diffs):3d}  chal−king {mu:+.4f} ± {se:.4f} (z {mu / se if se else float('nan'):+.2f})   "
           f"refs−king {mu2:+.4f} ± {se2:.4f}   live z {m.get('z', float('nan')):+.2f} wins={m.get('challenger_wins')}")
    return "\n".join(L) + "\n", rep


def parity_A(terms: list[dict]) -> str:
    """Base-side action logprob per byte vs the stored live echo (only under
    with_thought rendering, where the two condition on the same thought)."""
    xs, ys = [], []
    for t in terms:
        lv = t.get("live") or {}
        b = (t.get("sums") or {}).get("base")
        if not b or t["rendering"] != "with_thought":
            continue
        live_v = (lv.get("lp") or {}).get("lpC_ya_za") if t["role"] in ("king", "challenger") else (lv.get("lp") or {}).get("lp_own")
        if live_v is None or not b["action"]["n_bytes"]:
            continue
        xs.append(b["action"]["lp"] / b["action"]["n_bytes"])
        ys.append(live_v)
    if len(xs) < 3:
        return "parity: not applicable (needs --rendering with_thought and live lp fields)\n"
    d = [x - y for x, y in zip(xs, ys)]
    return (f"parity (base action lp/byte vs live lpC(y|z) echo): n {len(xs)}  mean |Δ| {st.mean(abs(v) for v in d):.5f}  "
            f"max |Δ| {max(abs(v) for v in d):.5f}  Spearman {P.spearman(xs, ys):+.3f}\n")


def report_B(terms: list[dict], rows: list[dict], metric: str) -> tuple[str, dict]:
    L: list[str] = []
    Pn = L.append
    meta = json.loads((OUT / "setB_meta.json").read_text()) if (OUT / "setB_meta.json").exists() else {}
    bench_by_label = meta.get("bench_axes", {})
    label_of = {r["model"]: r["label"] for r in rows}
    bench = {m: bench_by_label.get(lab) for m, lab in label_of.items()}
    # ref terms per turn (shared by every model row on the turn)
    ref_terms: dict[str, dict[str, float]] = collections.defaultdict(dict)
    model_terms: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for t in terms:
        if t.get("term") is None:
            continue
        if t["role"].startswith("ref_"):
            ref_terms[t["turn_id"]][t["role"]] = t[metric]
        elif t["role"] == "model":
            model_terms[t["row_model"]][t["turn_id"]] = t[metric]
    dialect_of = {r["turn_id"]: r["dialect"] for r in rows}
    group_of = {r["turn_id"]: r["group"] for r in rows}
    rep: dict = {"metric": metric}

    def per_model(tids_ok) -> dict[str, dict]:
        out = {}
        for mk, d in model_terms.items():
            vals, exc = [], []
            for tid, v in d.items():
                if not tids_ok(tid):
                    continue
                vals.append(v)
                refs = list(ref_terms.get(tid, {}).values())
                if refs:
                    exc.append(v - st.mean(refs))
            if vals:
                m, s = P.mean_se(vals)
                me, se_ = P.mean_se(exc)
                out[mk] = {"n": len(vals), "mean": m, "se": s, "excess": me, "excess_se": se_}
        # teacher LOO
        vals, exc = [], []
        for tid, d in ref_terms.items():
            if not tids_ok(tid) or len(d) < 2:
                continue
            rs = list(d.values())
            for i, v in enumerate(rs):
                vals.append(v)
                exc.append(v - st.mean(rs[:i] + rs[i + 1:]))
        if vals:
            m, s = P.mean_se(vals)
            me, se_ = P.mean_se(exc)
            out[P.TEACHER_KEY] = {"n": len(vals), "mean": m, "se": s, "excess": me, "excess_se": se_}
        return out

    def align(tab: dict[str, dict], title: str) -> dict:
        keys = [k for k in tab if bench.get(k) and bench[k].get("total") is not None]
        Pn(f"-- alignment ({title}): Spearman(stat, centered bench axis) over n={len(keys)} models "
           f"[{', '.join(label_of.get(k, k) for k in keys)}]; perm p two-sided; 95% CI bootstrap over models --")
        res = {}
        for stat in ("mean", "excess"):
            line = f"{stat:10} "
            res[stat] = {}
            for ax in P.AXES:
                ks = [k for k in keys if bench[k].get(ax) is not None and tab[k].get(stat) is not None]
                if len(ks) < 5:
                    line += f"{ax}: {'-':>24} "
                    continue
                x = [tab[k][stat] for k in ks]
                y = [bench[k][ax] for k in ks]
                rho = P.spearman(x, y)
                p = P.permutation_p(x, y, rho, 5000)
                lo, hi = P.boot_spearman_ci(x, y, 1000)
                res[stat][ax] = {"rho": rho, "p": p, "ci": [lo, hi], "n": len(ks)}
                line += f"{ax}: {rho:+.2f} p{p:.2f} [{lo:+.2f},{hi:+.2f}]  "
            Pn(line)
        return res

    order = [P.TEACHER_KEY, P.GENESIS_KEY] + [P.KINGS[r][0] for r in sorted(P.KINGS, reverse=True)]
    Pn(f"===== SET B — N1 term ({metric}) per model vs benchmark cards =====")
    Pn("mean = mean term over the model's turns; excess = mean over turns of (model term − mean of the 3 ref terms on the same turn);")
    Pn("teacher row = leave-one-out (ref_i − mean of the other two; its mean is identically 0 for k=3, so the teacher anchors `excess`);")
    Pn("bench axes centered as panel.py (score − panel mean per cell)")
    rep["tables"] = {}
    rep["alignment"] = {}
    for title, ok in (("ALL turns", lambda t: True),
                      ("shell dialects (bash/tool_call/terminus_json)", lambda t: dialect_of[t] in ("bash", "tool_call", "terminus_json")),
                      ("text dialect", lambda t: dialect_of[t] == "text"),
                      ("boxed dialect", lambda t: dialect_of[t] == "boxed"),
                      ("agentic groups (coding/terminal)", lambda t: group_of[t] in ("coding", "terminal"))):
        tab = per_model(ok)
        if sum(v["n"] for v in tab.values()) < 30:
            continue
        Pn("")
        Pn(f"===== PER-MODEL: {title} =====")
        Pn(f"{'model':8} {'n':>5} {'mean':>9} {'se':>7} {'excess':>9} {'se':>7}   bench(centered): total agentic noTau2 chat")
        for mk in order:
            if mk not in tab:
                continue
            r = tab[mk]
            b = bench.get(mk) or {}
            Pn(f"{label_of.get(mk, mk):8} {r['n']:5d} {_fmt(r['mean'], 9)} {_fmt(r['se'], 7)} {_fmt(r['excess'], 9)} {_fmt(r['excess_se'], 7)}"
               f"   {_fmt(b.get('total'), 6, 1)} {_fmt(b.get('agentic'), 7, 1)} {_fmt(b.get('agentic_no_tau2'), 6, 1)} {_fmt(b.get('chat'), 6, 1)}")
        rep["tables"][title] = {label_of.get(k, k): v for k, v in tab.items()}
        rep["alignment"][title] = align(tab, title)
        if title == "ALL turns":
            modern = {k: v for k, v in tab.items() if k in (P.TEACHER_KEY, P.GENESIS_KEY) or k in [P.KINGS[r][0] for r in range(16, 21)]}
            rep["alignment"]["ALL turns — kings 16–20 + teacher + genesis"] = align(
                modern, "ALL turns — sub-panel kings 16–20 + teacher + genesis")
    return "\n".join(L) + "\n", rep


# ------------------------------------------------------------------ main
async def amain(a) -> int:
    rows = load_set(a.set, a.limit, a.stride)
    tok = C.teacher_tokenizer()
    jobs, jstats = candidate_jobs(rows, a.rendering, tok, a.max_tokens)
    n_cands = sum(len(r["candidates"]) for r in rows)
    print(f"set {a.set}: {len(rows)} rows, {n_cands} candidates -> {len(jobs)} unique echoes per side "
          f"({sum(j['n_tokens'] for j in jobs.values()) / 1e6:.2f}M prompt tokens/side); {jstats}", flush=True)
    tag = f"{a.set}_{a.rendering}"
    if a.dry_run:
        base = EngyBackend(concurrency=a.concurrency)
    else:
        base = VllmBackend(a.endpoint_base, a.model_base, a.concurrency, api_key=a.api_key)
        await base.resolve_model()
    echoes = {"base": await run_echoes(jobs, base, echo_cache_path(tag, "base"), "base", a.concurrency)}
    if a.dry_run or a.mock_plus:
        plus = MockPlusBackend({k: v["lps"] for k, v in echoes["base"].items() if v.get("lps") is not None}, a.mock_sigma)
    elif a.lora_name:
        plus = VllmBackend(a.endpoint_plus or a.endpoint_base, a.lora_name, a.concurrency, api_key=a.api_key)
    else:
        if not a.endpoint_plus:
            raise SystemExit("--endpoint-plus or --lora-name required (or --mock-plus)")
        plus = VllmBackend(a.endpoint_plus, a.model_plus, a.concurrency, api_key=a.api_key)
        await plus.resolve_model()
    echoes["plus"] = await run_echoes(jobs, plus, echo_cache_path(tag, "plus"), "plus", a.concurrency)
    n_err = {s: sum(1 for r in e.values() if r.get("error")) for s, e in echoes.items()}
    terms = compute_terms(rows, jobs, echoes, a.rendering, a.level)
    suffix = ("_dryrun" if a.dry_run else "") + ("" if a.rendering == "no_thought" else "_" + a.rendering)
    C.write_jsonl(OUT / f"set{a.set}_terms{suffix}.jsonl", terms)
    header = (f"N1 stage-2 scorer — set {a.set}; rendering {a.rendering}; span level {a.level}; metric {a.metric}\n"
              f"base = {base.name}; plus = {plus.name}; rows {len(rows)}; candidates {n_cands}; unique echoes {len(jobs)}; "
              f"echo errors {n_err}; terms {sum(1 for t in terms if t.get('term') is not None)}/{len(terms)}; "
              f"engy cost ${getattr(base, 'cost_usd', 0.0):.4f}\n"
              + ("!! DRY RUN: plus side is base + noise; every statistic below is pipeline exercise only !!\n" if a.dry_run or a.mock_plus else ""))
    text, rep = (report_A if a.set == "A" else report_B)(terms, rows, a.metric)
    if a.set == "A":
        text = parity_A(terms) + text
    body = header + "\n" + text
    (OUT / f"set{a.set}_report{suffix}.txt").write_text(body)
    (OUT / f"set{a.set}_report{suffix}.json").write_text(json.dumps(
        {"header": header, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
         "job_stats": jstats, "echo_errors": n_err, **rep}, indent=1, default=str))
    print(body)
    print(f"-> {OUT / f'set{a.set}_report{suffix}.txt'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=("A", "B"), required=True)
    ap.add_argument("--endpoint-base", help="OpenAI-compatible vLLM base URL for the frozen teacher C (…/v1)")
    ap.add_argument("--endpoint-plus", help="…for C+ (omit with --lora-name to reuse --endpoint-base)")
    ap.add_argument("--lora-name", help="served LoRA adapter name for C+ on the base endpoint")
    ap.add_argument("--model-base", help="served model name for C (default: first of /v1/models)")
    ap.add_argument("--model-plus", help="served model name for C+ (default: first of /v1/models)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rendering", choices=("no_thought", "with_thought"), default="no_thought",
                    help="no_thought = the N1 term as defined (y|x); with_thought = each candidate under its own z (parity with live echoes)")
    ap.add_argument("--level", choices=LEVELS, default="body", help="span summed for the term (default body = chosen bytes only)")
    ap.add_argument("--metric", choices=("term", "term_per_byte", "term_per_tok"), default="term")
    ap.add_argument("--max-tokens", type=int, default=60000, help="skip candidates whose full text exceeds this many tokens")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only")
    ap.add_argument("--stride", type=int, default=1, help="take every k-th row (with --limit: a spread pilot)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="base side via Engy qwen3.8-27b echo, plus side mocked")
    ap.add_argument("--mock-plus", action="store_true", help="plus side = base + noise (pipeline test with a real base endpoint)")
    ap.add_argument("--mock-sigma", type=float, default=0.05)
    a = ap.parse_args()
    if not a.dry_run and not a.endpoint_base:
        ap.error("--endpoint-base required unless --dry-run")
    return asyncio.run(amain(a))


if __name__ == "__main__":
    raise SystemExit(main())
