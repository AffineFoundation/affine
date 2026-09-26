"""Routed references (Alan's multi-teacher router, 2026-09-21) — keep Qwen as
the LIKELIHOOD function (every echo) and as the TYPICALITY anchor (typ_c from
the stored plain Qwen refs), but draw the z_R / z_A REFERENCES from a pool
member (Genesis = Qwen/Qwen3.6-35B-A3B via Engy, Occamy-1.0 via a rented
vLLM pod). Does that open headroom above the teacher where the environment
says the pool member is better (τ² telecom), does teacher-vs-king separation
survive, and is uploading the pool member the binding risk (RT-14)?

    python routed_refs.py turns                  # select + materialize (~120 τ² + ~60 bash)
    python routed_refs.py sample --family gen    # k=4 Genesis samples / turn (Engy qwen3.6-35b-a3b)
    python routed_refs.py sample --family occ    # k=4 Occamy samples / turn (pod endpoint json)
    python routed_refs.py echo                   # full candidate grid under the Qwen teacher (Engy echo)
    python routed_refs.py report

Work files under research/results/frontier_arbiter/routed_refs/ (jsonl,
resumable). Cost ledger cost.jsonl; running $ printed at every checkpoint.

Terms (one line each):
  x / z / y        turn prefix / thought / action (dialect action span; text = whole visible reply).
  Q refs (q0..q2)  the 3 stored Qwen teacher refs of the wvk-22 verdict (today's references AND the typ_c anchor).
  G refs (g0..g2)  3 fresh Genesis samples at T0.8; g3 = a 4th fresh Genesis sample used only as a miner.
  O refs (o0..o2)  3 fresh Occamy samples; o3 = a 4th, miner only.
  arm              which refs feed z_R / z_A: Q3 = Q refs, G3 = G refs, MIX = q0,q1,g0, O3 = O refs.
  a_i(z)           lpC(y_i|x,z) − lpC(y_i|x,∅) per byte (ref i's action under thought z).
  R(z|S)           τ·log mean_{i∈S} exp(a_i/τ) − mean_{i∈S} a_i, τ = 0.03 (centred tempered LME).
  b_i(y)           [lpC(y|x,z_i) − lpC(y|x,∅)] · bytes(y) (summed nats, a_norm = 1).
  A(y|S)           τ·log mean_{i∈S} exp(b_i/τ).
  LOO              ref j scored against the other refs of S (2 refs); μ_leg = mean_j of the LOO values
                   (over the OTHER refs when the miner is itself a ref); σ_leg = pooled within-turn sd
                   of the LOO values per dialect (per arm).
  z_R / z_A        (R − μ_R)/σ_R, (A − μ_A)/σ_A.
  m_c              content-masked mean token logprob of z under x: tokens with |lpC(tok|x) − lpC(tok|∅)| > 1 nat.
  typ_c            2 − |m_c − μ_c|/σ_c, μ_c/σ_c ALWAYS from the Q refs' thoughts (LOO for a Q ref itself);
                   < 10 content tokens → −2.4.
  turn             min(z_R, typ_c, z_A); bind = the leg attaining the min.
  families         teacher = mean over q0..q2; genesis = mean over g0..g2 (LOO under G3); genesis_fresh = g3;
                   occamy / occamy_fresh likewise; king = stored (z_a, y_a).
  in-band          typ_c >= 0.
  headroom         paired mean over turns of family − teacher (z = mean/SE).
  separation       paired teacher − king (the positive control).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    REPO, BOX, BOX_KEY, BOX_REPO, GENESIS_ENGY, Engy, TeacherEcho, agree, box_ssh, clme, exact,
    jaccard, lme, load_verdict, materialize, norm_action, read_jsonl, append_jsonl, write_jsonl,
    verdict_index, corpus_for,
)
from privileged_refs import parse_reply  # noqa: E402
from evalsrv.sdmeter import content_stats  # noqa: E402

RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "routed_refs"
TURNS = RESULTS / "turns.jsonl"
SAMPLES = RESULTS / "samples.jsonl"
ECHOES = RESULTS / "echoes.jsonl"
COST = RESULTS / "cost.jsonl"
OCC_ENDPOINT = Path("/tmp/occamy/endpoint.json")

CHAL_RANGE = range(600, 631)
TAU2_SOURCES = ("affine_tau2", "affine_tau2_synth")
TAU2_DIALECTS = ("text", "tool_call")
BASH_SOURCES = ("swesmith", "scaleswe", "multiswe", "swerebench_v2", "terminal_lego",
                "terminal_bench_2", "r2e_gym", "swelego", "nl2repobench")
N_TAU2 = 120
N_BASH = 60
MAX_PREFIX_CHARS = 80_000
K = 3
K_SAMPLE = 4
TAU = 0.03
REF_TEMPERATURE = 0.8
REF_MAX_TOKENS = 4096
BUDGET_USD = 70.0
THETA = 1.0
WIDTH = 2.0
FLOOR = -2.4
CONTENT_MIN = 10
FAMILY_MODEL = {"gen": GENESIS_ENGY, "occ": "occamy-1.0"}
ARMS = {"Q3": ("q0", "q1", "q2"), "G3": ("g0", "g1", "g2"), "MIX": ("q0", "q1", "g0"), "O3": ("o0", "o1", "o2")}
FAMILIES = {"teacher": ("q0", "q1", "q2"), "genesis": ("g0", "g1", "g2"), "genesis_fresh": ("g3",),
            "occamy": ("o0", "o1", "o2"), "occamy_fresh": ("o3",), "king": ("king",),
            "hybrid_qz_ga": ("h0", "h1", "h2")}   # virtual: teacher thought q_j + Genesis action g3
LEGS = ("z_R", "typ_c", "z_A", "score")


# ------------------------------------------------------------------ cost ledger
def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "stage": stage, "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | Engy total so far ${tot:.2f} {note}", flush=True)
    if tot > BUDGET_USD:
        raise SystemExit(f"budget exceeded: ${tot:.2f} > ${BUDGET_USD}")


def total_cost() -> float:
    """Sum of the max cumulative cost per (stage, run); a cumulative value
    dropping within a group marks a fresh process."""
    groups: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in read_jsonl(COST):
        groups[(r["stage"], str(r.get("note", "")).split("/")[-1])].append(r["cost_usd"])
    total = 0.0
    for vals in groups.values():
        run_max = 0.0
        for c in vals:
            if c < run_max:
                total += run_max
                run_max = 0.0
            run_max = max(run_max, c)
        total += run_max
    return total


# ------------------------------------------------------------------ stage: turns
def eligible_turns(chal: str) -> list[dict]:
    d = load_verdict(chal)
    sl = d["verdict"]["slice"]
    corpus = corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    sigma = ((d["verdict"].get("shadow") or {}).get("sd_meter") or {}).get("sigma_by_dialect") or {}
    king = {r["turn_id"]: r for r in d["king_rows"]}
    out = []
    for tid in d["turn_ids"]:
        row = rows.get(tid)
        if not row:
            continue
        src, kind = row["source"], row["action_kind"]
        if src in TAU2_SOURCES and kind in TAU2_DIALECTS:
            group = "tau2"
        elif src in BASH_SOURCES and kind == "bash":
            group = "bash"
        else:
            continue
        if int(row["n_prefix_chars"]) > MAX_PREFIX_CHARS:
            continue
        refs = d["teacher_refs"].get(tid)
        if not refs or len(refs) != K:
            continue
        if any(r.get(k) is None for r in refs for k in
               ("z", "y", "lp_own", "lp_empty", "lp_thought", "lp_thought_e", "mc_thought", "lp_cross", "n_bytes_y")):
            continue
        if any(len(r["lp_cross"]) != K or sum(v is None for v in r["lp_cross"]) != 1 for r in refs):
            continue
        kr = king.get(tid)
        if not kr or not kr.get("valid") or len(kr.get("pairs") or []) != K:
            continue
        if any(p.get("mc_za") is None or p.get("lpC_ya_zc") is None for p in kr["pairs"]):
            continue
        if not kr["pairs"][0].get("z_a") or not kr["pairs"][0].get("y_a"):
            continue
        out.append({"turn_id": tid, "chal": chal, "group": group, "dialect": kind, "source": src,
                    "turn_idx": int(row["turn_idx"]), "rollout_id": row["rollout_id"],
                    "n_prefix_chars": int(row["n_prefix_chars"]), "stratum": row["stratum"],
                    "phase": row.get("phase"),
                    "refs": [{k: r[k] for k in ("z", "y", "lp_own", "lp_empty", "lp_thought", "lp_thought_e",
                                                "mc_thought", "lp_cross", "n_bytes_y", "n_content_thought",
                                                "n_tokens_thought")} for r in refs],
                    "king": {"z_a": kr["pairs"][0]["z_a"], "y_a": kr["pairs"][0]["y_a"],
                             "pairs": [{k: p.get(k) for k in ("lpC_yc_za", "lpC_ya_za", "lpC_ya_e", "n_bytes_ya",
                                                              "lpC_yc_zc", "lpC_yc_e", "lpC_ya_zc", "mc_za",
                                                              "n_content_za", "n_tokens_za")} for p in kr["pairs"]]},
                    "sigma_stored": sigma.get(kind)})
    return out


def task_join(rollout_ids: list[str]) -> dict:
    """rollouts.sqlite on the box: the D turn's own rollout (seat / outcome /
    task) and, for its task, every seat's outcome (teacher / king labels)."""
    ids = ",".join(f"'{r}'" for r in rollout_ids)
    q1 = (f"select rollout_id, task_sid, task_uid, seat, model_label, outcome, score from rollouts "
          f"where rollout_id in ({ids});")
    raw = box_ssh(f"cd {BOX_REPO} && sqlite3 -json ops/kingboard/state/rollouts.sqlite \"{q1}\"")
    own = {r["rollout_id"]: r for r in (json.loads(raw) if raw.strip() else [])}
    sids = sorted({r["task_sid"] for r in own.values() if r.get("task_sid")})
    by_task: dict[str, list] = collections.defaultdict(list)
    for i in range(0, len(sids), 200):
        chunk = ",".join(f"'{s}'" for s in sids[i:i + 200])
        q2 = (f"select task_sid, seat, model_label, outcome from rollouts where task_sid in ({chunk}) "
              f"and outcome in ('solved','failed');")
        raw = box_ssh(f"cd {BOX_REPO} && sqlite3 -json ops/kingboard/state/rollouts.sqlite \"{q2}\"")
        for r in (json.loads(raw) if raw.strip() else []):
            by_task[r["task_sid"]].append(r)
    out = {}
    for rid, r in own.items():
        rows = by_task.get(r.get("task_sid") or "", [])
        t_rows = [x for x in rows if x["seat"] == "teacher"]
        k_rows = [x for x in rows if x["seat"] == "king"]
        out[rid] = {"task_sid": r.get("task_sid"), "task_uid": r.get("task_uid"), "own_seat": r.get("seat"),
                    "own_model": r.get("model_label"), "own_outcome": r.get("outcome"),
                    "teacher_n": len(t_rows), "teacher_solved": sum(x["outcome"] == "solved" for x in t_rows),
                    "king_n": len(k_rows), "king_solved": sum(x["outcome"] == "solved" for x in k_rows),
                    "king_by_model": {m: [sum(x["outcome"] == "solved" for x in k_rows if x["model_label"] == m),
                                          sum(1 for x in k_rows if x["model_label"] == m)]
                                      for m in sorted({x["model_label"] for x in k_rows})}}
    return out


def fault_class(task_uid: str | None) -> str | None:
    if not task_uid:
        return None
    a, b = task_uid.find("["), task_uid.find("]")
    return task_uid[a + 1:b] if 0 <= a < b else None


def cmd_turns(args: argparse.Namespace) -> None:
    rng = random.Random(20260921)
    idx = [r for r in verdict_index() if r.get("challenge_id")
           and int(r["challenge_id"].split("-")[1]) in CHAL_RANGE and not r.get("rejection_reason")]
    print(f"{len(idx)} verdicts in range with a duel")
    cands: list[dict] = []
    for r in idx:
        chal = r["challenge_id"]
        try:
            e = eligible_turns(chal)
        except Exception as ex:  # noqa: BLE001
            print(f"  {chal}: skipped ({type(ex).__name__}: {str(ex)[:100]})")
            continue
        c = collections.Counter((t["group"], t["source"], t["dialect"]) for t in e)
        print(f"  {chal}: {len(e)} eligible; tau2 {sum(v for k, v in c.items() if k[0] == 'tau2')}, "
              f"bash {sum(v for k, v in c.items() if k[0] == 'bash')}")
        cands.extend(e)
    # one turn per rollout; telecom (affine_tau2) first, then synth; bash spread over sources
    by_roll: dict[str, list[dict]] = collections.defaultdict(list)
    for t in cands:
        by_roll[t["rollout_id"]].append(t)
    per_roll = [rng.choice(v) for v in by_roll.values()]
    rng.shuffle(per_roll)
    tau2 = [t for t in per_roll if t["group"] == "tau2"]
    tau2.sort(key=lambda t: (t["source"] != "affine_tau2", rng.random()))
    # keep both dialects represented: cap text at 2/3 of the quota when tool_call is available
    picked_tau2: list[dict] = []
    n_text_cap = int(N_TAU2 * 0.67)
    tool = [t for t in tau2 if t["dialect"] == "tool_call"]
    text = [t for t in tau2 if t["dialect"] == "text"]
    picked_tau2.extend(tool[: N_TAU2 - min(len(text), n_text_cap)])
    picked_tau2.extend(text[: N_TAU2 - len(picked_tau2)])
    bash = [t for t in per_roll if t["group"] == "bash"]
    pools = collections.defaultdict(list)
    for t in bash:
        pools[t["source"]].append(t)
    picked_bash: list[dict] = []
    while len(picked_bash) < N_BASH and any(pools.values()):
        for s in sorted(pools):
            if pools[s] and len(picked_bash) < N_BASH:
                picked_bash.append(pools[s].pop())
    picked = picked_tau2 + picked_bash
    print(f"picked tau2 {len(picked_tau2)} {dict(collections.Counter((t['source'], t['dialect']) for t in picked_tau2))}; "
          f"bash {len(picked_bash)} {dict(collections.Counter(t['source'] for t in picked_bash))}")
    print("joining task outcomes from rollouts.sqlite")
    tj = task_join(sorted({t["rollout_id"] for t in picked}))
    out = []
    for chal in sorted({t["chal"] for t in picked}):
        sub = [t for t in picked if t["chal"] == chal]
        mat = materialize(chal, [t["turn_id"] for t in sub])
        for t in sub:
            m = mat.get(t["turn_id"])
            if not m:
                print(f"  materialize miss {t['turn_id']}")
                continue
            t["prefix"] = m["prefix"]
            t["task"] = tj.get(t["rollout_id"])
            t["fault_class"] = fault_class((t["task"] or {}).get("task_uid"))
            out.append(t)
    write_jsonl(TURNS, out)
    print(f"{len(out)} turns -> {TURNS}")
    for g in ("tau2", "bash"):
        ts = [t for t in out if t["group"] == g]
        tk = [t for t in ts if t.get("task")]
        print(f"  {g}: n={len(ts)} joined={len(tk)} own_seat={dict(collections.Counter((t['task'] or {}).get('own_seat') for t in ts))} "
              f"own_outcome={dict(collections.Counter((t['task'] or {}).get('own_outcome') for t in ts))} "
              f"teacher_solved_any={sum(1 for t in tk if t['task']['teacher_solved'] > 0)}/{sum(1 for t in tk if t['task']['teacher_n'] > 0)} "
              f"prefix p50={st.median(t['n_prefix_chars'] for t in ts) if ts else 0:.0f}")
        if g == "tau2":
            print("   fault classes:", dict(collections.Counter(t["fault_class"] for t in ts).most_common(12)))


# ------------------------------------------------------------------ stage: sample
class PodClient(Engy):
    """OpenAI-compatible vLLM pod: vLLM 0.28 returns the latent thought as
    `reasoning` (Engy: `reasoning_content`); no per-call charge (pod is billed by the hour)."""

    async def chat(self, model: str, messages: list[dict], temperature: float = 0.8,
                   max_tokens: int = REF_MAX_TOKENS, **extra) -> dict:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, **extra}
        d = await self._post("/chat/completions", payload, model)
        self._tally(model, d)
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        return {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "finish": ch.get("finish_reason"), "usage": d.get("usage"),
                "cost_usd": 0.0, "worker": None, "model": d.get("model") or model}


def pod_client(endpoint: Path, concurrency: int = 12) -> Engy:
    ep = json.load(open(endpoint))
    e = PodClient(key="x", concurrency=concurrency, timeout=1800.0, retries=4)
    e.cli = httpx.AsyncClient(base_url=ep["base_url"], timeout=1800.0,
                              headers={"Authorization": f"Bearer {ep['key']}"})
    return e


def occ_client() -> Engy:
    return pod_client(OCC_ENDPOINT)


class PodEcho(TeacherEcho):
    """TeacherEcho against a rented vLLM pod serving the frozen teacher with
    the affine echo-cache plugin (ops/teacher-swarm/echo_cache_plugin): the
    request mirrors evalsrv/vllm_client.py::_echo_span (echo=True, logprobs=0,
    add_special_tokens=False, vllm_xargs.affine_echo_tail = span tokens + 9);
    cached positions come back as +1.0 and force an uncached retry when they
    reach into the span. Same output fields as TeacherEcho."""

    def __init__(self, engy: Engy, model: str):
        super().__init__(engy)
        self.model = model
        self.uncached_retries = 0

    async def _echo(self, full: str, spans: list[tuple[int, int]], tokens: bool = False) -> dict:
        n_bytes = sum(len(full[a:b].encode()) for a, b in spans) or 1
        if not spans:
            return {"sum_lp": 0.0, "n_tokens": 0, "n_bytes": 1, "lp_per_byte": 0.0,
                    **({"tokens": []} if tokens else {})}
        ids, offs = self._encode(full)
        in_span_local = [any(a <= s < b for a, b in spans) for s, _ in offs]
        first = next((i for i, v in enumerate(in_span_local) if v), None)
        if first is None:
            raise ValueError("span not found in tokenization")
        tail = len(ids) - first + 1 + 8
        payload = {"model": self.model, "prompt": full, "max_tokens": 1, "temperature": 0,
                   "echo": True, "logprobs": 0, "add_special_tokens": False,
                   "vllm_xargs": {"affine_echo_tail": tail}}
        d = await self.engy._post("/completions", payload, self.model)
        self.n_calls += 1

        def unpack(d):
            lpo = d["choices"][0]["logprobs"]
            lp = lpo["token_logprobs"]
            to = lpo.get("text_offset")
            n_prompt = len(lp) - 1                       # echo = prompt tokens + 1 generated
            if n_prompt == len(ids):
                starts = [s for s, _ in offs]             # same tokenizer, same text: local offsets
            elif to and len(to) >= n_prompt:
                self.offset_fallbacks = getattr(self, "offset_fallbacks", 0) + 1
                starts = to[:n_prompt]
            else:
                raise ValueError(f"echo token count {n_prompt} != local {len(ids)} and no text_offset")
            keep = [any(a <= s < b for a, b in spans) for s in starts]
            return lp[:n_prompt], starts, keep

        lp, starts, keep = unpack(d)
        if any(k and (x is None or x > 0) for k, x in zip(keep, lp)):
            payload.pop("vllm_xargs")
            self.uncached_retries += 1
            d = await self.engy._post("/completions", payload, self.model)
            lp, starts, keep = unpack(d)
        scored = []
        for j, (x, s, k) in enumerate(zip(lp, starts, keep)):
            if k and x is not None:
                e = starts[j + 1] if j + 1 < len(starts) else len(full)
                scored.append((s, e, x))
        ssum = sum(x for _, _, x in scored)
        out = {"sum_lp": ssum, "n_tokens": len(scored), "n_bytes": n_bytes, "lp_per_byte": ssum / n_bytes}
        if tokens:
            base = spans[0][0]
            out["tokens"] = [(a - base, b - base, x) for a, b, x in scored]
        return out


async def sample_all(turns: list[dict], engy: Engy, family: str, model: str) -> None:
    done = {(r["turn_id"], r["family"], r["i"]) for r in read_jsonl(SAMPLES) if "error" not in r}
    jobs = [(t, i) for t in turns for i in range(K_SAMPLE) if (t["turn_id"], family, i) not in done]
    print(f"sampling {len(jobs)} {family} replies from {model}")

    async def one(t, i):
        try:
            r = await engy.chat(model, t["prefix"], temperature=REF_TEMPERATURE, max_tokens=REF_MAX_TOKENS)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "family": family, "i": i, "error": repr(ex)[:300]})
            return
        p = parse_reply(r, t["dialect"])
        append_jsonl(SAMPLES, {"turn_id": t["turn_id"], "family": family, "i": i, "model": r.get("model"),
                               "reasoning": r["reasoning"], "content": r["content"], "tool_calls": r["tool_calls"],
                               "finish": r["finish"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                               **{k: p[k] for k in ("z", "y", "parsed", "kind_used", "think_closed", "repaired")}})

    step = 48
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost(f"sample_{family}", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    if args.turns:
        turns = turns[: args.turns]
    if args.family == "occ":
        engy = occ_client()
    else:
        engy = Engy(concurrency=args.concurrency)
    asyncio.run(sample_all(turns, engy, args.family, FAMILY_MODEL[args.family]))
    ss = [s for s in read_jsonl(SAMPLES) if s.get("family") == args.family]
    ok = [s for s in ss if "error" not in s]
    print(f"{args.family}: {len(ok)} replies, parsed {sum(1 for s in ok if s['parsed'])}, "
          f"cap hits {sum(1 for s in ok if s['finish'] == 'length')}, text fallback "
          f"{sum(1 for s in ok if s.get('kind_used') == 'text')}, errors {len(ss) - len(ok)}")


# ------------------------------------------------------------------ candidates
def canon_tool_call(y: str) -> str:
    """Render a JSON-style `<tool_call>{"name":..,"arguments":{..}}</tool_call>` in the
    Qwen3 XML dialect the teacher / king / Occamy emit on D (`<function=NAME>` +
    `<parameter=K>\\nV\\n</parameter>`; non-string values as JSON, like the Qwen3
    chat template). Genesis (Qwen3.6) emits the JSON form on every tool_call turn
    (150/150 samples); a routed reference must be in the miner's dialect, otherwise
    every byte-level comparison (norm-exact, Jaccard) and the action echoes compare
    formats instead of calls. Anything that is not that shape is returned unchanged."""
    s = y.strip()
    if not (s.startswith("<tool_call>") and s.endswith("</tool_call>")):
        return y
    inner = s[len("<tool_call>"):-len("</tool_call>")].strip()
    if not inner.startswith("{"):
        return y
    try:
        d = json.loads(inner)
    except ValueError:
        return y
    if not isinstance(d, dict) or not d.get("name"):
        return y
    args = d.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"arguments": args}
    parts = [f"<function={d['name']}>"]
    for k, v in (args.items() if isinstance(args, dict) else []):
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        parts.append(f"<parameter={k}>\n{val}\n</parameter>")
    parts.append("</function>")
    return "<tool_call>\n" + "\n".join(parts) + "\n</tool_call>"


def candidates(turn: dict, samples_by: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """cid -> {z, y} for one turn: q0..q2 stored, g0..g3 / o0..o3 parsed samples
    (in sample order: the first 3 parsed are the refs, the 4th the miner), king.
    Pool-member tool calls are rendered in the turn's dialect (`canon_tool_call`)."""
    out = {f"q{i}": {"z": r["z"], "y": r["y"]} for i, r in enumerate(turn["refs"])}
    out["king"] = {"z": turn["king"]["z_a"], "y": turn["king"]["y_a"]}
    for fam, tag in (("gen", "g"), ("occ", "o")):
        ss = sorted(samples_by.get(turn["turn_id"], {}).get(fam, []), key=lambda s: s["i"])
        parsed = [s for s in ss if s.get("parsed") and s.get("y") and s.get("z")]
        for j, s in enumerate(parsed[:K_SAMPLE]):
            y = canon_tool_call(s["y"]) if turn["dialect"] == "tool_call" else s["y"]
            out[f"{tag}{j}"] = {"z": s["z"], "y": y, "i": s["i"], "canon": y != s["y"]}
    # virtual hybrids = the router's target miner, "think like Qwen, do what the pool
    # member does": stored teacher thought q_j + the fresh Genesis action g3. Every echo
    # they need already exists (EchoTable resolves y_of / z_of); nothing is sampled.
    if "g3" in out:
        for j in range(K):
            out[f"h{j}"] = {"z": out[f"q{j}"]["z"], "y": out["g3"]["y"], "virtual": True,
                            "z_of": f"q{j}", "y_of": "g3"}
    return out


def samples_index() -> dict[str, dict[str, list[dict]]]:
    by: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in read_jsonl(SAMPLES):
        if "error" not in s:
            by[s["turn_id"]][s["family"]].append(s)
    return by


# ------------------------------------------------------------------ stage: echo
def read_echoes() -> list[dict]:
    out: list[dict] = []
    for p in sorted(RESULTS.glob("echoes*.jsonl")):
        out.extend(read_jsonl(p))
    return out


def echo_jobs(turn: dict, cands: dict[str, dict], done: set[str]) -> list[tuple]:
    tid = turn["turn_id"]
    jobs: list[tuple] = []

    def add(key, kind, z, y=None):
        if key not in done:
            jobs.append((key, kind, turn["prefix"], z, y))

    real = {c: r for c, r in cands.items() if not r.get("virtual")}
    for c, r in real.items():
        add(f"{tid}|th|{c}", "thought", r["z"])
        add(f"{tid}|un|{c}", "uncond", r["z"])
        add(f"{tid}|emp|{c}", "action", "", r["y"])
        for c2, r2 in real.items():
            add(f"{tid}|x|{c}|{c2}", "action", r2["z"], r["y"])      # lpC(y_c | x, z_c2)
    return jobs


async def echo_all(turns: list[dict], engy: Engy, start: int = 0, pod_model: str | None = None) -> None:
    te = PodEcho(engy, pod_model) if pod_model else TeacherEcho(engy)
    sb = samples_index()
    done = {r["key"] for r in read_echoes() if "error" not in r}
    out_path = ECHOES if start == 0 else RESULTS / f"echoes.part{start}.jsonl"
    jobs: list[tuple] = []
    for t in turns:
        jobs.extend(echo_jobs(t, candidates(t, sb), done))
    print(f"echo: {len(jobs)} echoes to run")

    async def one(key, kind, prefix, z, y):
        try:
            if kind == "action":
                r = await te.lp_action(prefix, z, y)
            elif kind == "thought":
                r = await te.lp_thought(prefix, z, tokens=True)
            else:
                r = await te.lp_thought_uncond(z, tokens=True)
            append_jsonl(out_path, {"key": key, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(out_path, {"key": key, "error": repr(ex)[:300]})

    step = 240 if pod_model else 120
    t0 = time.time()
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        done_n = min(s + step, len(jobs))
        rate = done_n / max(time.time() - t0, 1e-6) * 60
        extra = (f" | pod echoes {te.n_calls}, uncached retries {te.uncached_retries}, "
                 f"offset fallbacks {getattr(te, 'offset_fallbacks', 0)}, {rate:.0f}/min") if pod_model else ""
        log_cost("echo", engy, f"{done_n}/{len(jobs)}{extra}")


def cmd_echo(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)[args.start:]
    if args.turns:
        turns = turns[: args.turns]
    if args.pod:
        ep = json.load(open(args.pod))
        engy = pod_client(Path(args.pod), concurrency=args.concurrency)
        asyncio.run(echo_all(turns, engy, args.start, pod_model=ep["model"]))
    else:
        engy = Engy(concurrency=args.concurrency)
        asyncio.run(echo_all(turns, engy, args.start))


# ------------------------------------------------------------------ stage: report
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _paired(diffs: list) -> dict:
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    if len(d) < 3:
        return {"n": len(d), "mean": _mean(d), "se": None, "z": None}
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": (st.mean(d) / se) if se > 0 else None}


def pooled_sd(vals_by_turn: list[list[float]]) -> float | None:
    v = [st.variance(x) for x in vals_by_turn if x and all(a is not None and math.isfinite(a) for a in x) and len(x) >= 2]
    return math.sqrt(st.mean(v)) if v else None


class EchoTable:
    def __init__(self, echoes: dict[str, dict], tid: str, cands: dict[str, dict] | None = None):
        self.e = echoes
        self.tid = tid
        # virtual candidates borrow their y / z echoes from the real owners
        self.y_of = {c: r.get("y_of", c) for c, r in (cands or {}).items()}
        self.z_of = {c: r.get("z_of", c) for c, r in (cands or {}).items()}

    def x(self, yc: str, zc: str) -> float | None:
        r = self.e.get(f"{self.tid}|x|{self.y_of.get(yc, yc)}|{self.z_of.get(zc, zc)}")
        return r["lp_per_byte"] if r else None

    def emp(self, c: str) -> float | None:
        r = self.e.get(f"{self.tid}|emp|{self.y_of.get(c, c)}")
        return r["lp_per_byte"] if r else None

    def nbytes(self, c: str) -> int | None:
        r = self.e.get(f"{self.tid}|emp|{self.y_of.get(c, c)}")
        return r["n_bytes"] if r else None

    def mc(self, c: str) -> dict | None:
        c = self.z_of.get(c, c)
        a, b = self.e.get(f"{self.tid}|th|{c}"), self.e.get(f"{self.tid}|un|{c}")
        if not a or not b:
            return None
        return content_stats([tuple(x) for x in a["tokens"]], [tuple(x) for x in b["tokens"]], THETA)

    def a_terms(self, S: tuple, z: str) -> list[float] | None:
        out = []
        for i in S:
            xi, ei = self.x(i, z), self.emp(i)
            if xi is None or ei is None:
                return None
            out.append(xi - ei)
        return out

    def b_terms(self, S: tuple, y: str) -> list[float] | None:
        ey, nb = self.emp(y), self.nbytes(y)
        if ey is None or nb is None:
            return None
        out = []
        for i in S:
            xi = self.x(y, i)
            if xi is None:
                return None
            out.append((xi - ey) * nb)
        return out


def loo_values(E: EchoTable, S: tuple) -> dict | None:
    """Per ref j of S: R_j, A_j (LOO over the other refs) and m_c."""
    R, A, Mc, Nc = [], [], [], []
    for j in S:
        others = tuple(i for i in S if i != j)
        a, b = E.a_terms(others, j), E.b_terms(others, j)
        if a is None or b is None:
            return None
        R.append(clme(a, TAU))
        A.append(lme(b, TAU))
        cs = E.mc(j)
        Mc.append(cs["mc"] if cs else None)
        Nc.append(cs["n_content"] if cs else None)
    return {"R": R, "A": A, "Mc": Mc, "Nc": Nc}


def raw_legs(E: EchoTable, S: tuple, m: str, loo: dict) -> dict | None:
    """R, A of candidate m under ref set S (LOO when m in S) + the μ's it is
    measured against (mean of the other refs' LOO values)."""
    parents = {E.y_of.get(m, m), E.z_of.get(m, m)} & set(S)      # a hybrid's real owners in S
    if m in S or parents:
        held = {m} | parents
        others = tuple(i for i in S if i not in held)
        a, b = E.a_terms(others, m), E.b_terms(others, m)
        muR = _mean([v for i, v in enumerate(loo["R"]) if S[i] not in held])
        muA = _mean([v for i, v in enumerate(loo["A"]) if S[i] not in held])
    else:
        a, b = E.a_terms(S, m), E.b_terms(S, m)
        muR, muA = _mean(loo["R"]), _mean(loo["A"])
    if a is None or b is None or muR is None or muA is None:
        return None
    cs = E.mc(m)
    return {"R": clme(a, TAU), "A": lme(b, TAU), "muR": muR, "muA": muA,
            "mc": cs["mc"] if cs else None, "n_content": cs["n_content"] if cs else None,
            "a": a, "b": b}


def typ_of(mc: float | None, n_content: int | None, mu_c: float | None, sigma_c: float | None) -> float | None:
    if n_content is not None and n_content < CONTENT_MIN:
        return FLOOR
    if mc is None or mu_c is None or not sigma_c:
        return None
    return WIDTH - abs(mc - mu_c) / sigma_c


def score_of(legs: dict, sigma: dict, typ: float | None) -> dict:
    zr = (legs["R"] - legs["muR"]) / sigma["R"] if sigma.get("R") else None
    za = (legs["A"] - legs["muA"]) / sigma["A"] if sigma.get("A") else None
    live = {k: v for k, v in (("R", zr), ("Gc", typ), ("A", za)) if v is not None}
    bind = min(live, key=live.get) if live else None
    return {"z_R": zr, "typ_c": typ, "z_A": za, "score": live[bind] if bind else None, "bind": bind,
            "R": legs["R"], "A": legs["A"], "mc": legs["mc"]}


def cmd_report(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    sb = samples_index()
    all_e = read_echoes()
    echoes = {r["key"]: r for r in all_e if "error" not in r}
    n_err = sum(1 for r in all_e if "error" in r)
    rows: list[dict] = []
    cands_by_tid: dict[str, dict] = {}
    # ---------- per-turn raw values
    for t in turns:
        tid = t["turn_id"]
        cands = candidates(t, sb)
        cands_by_tid[tid] = cands
        E = EchoTable(echoes, tid, cands)
        row = {"turn_id": tid, "group": t["group"], "dialect": t["dialect"], "source": t["source"],
               "fault_class": t.get("fault_class"), "task": t.get("task"), "n_prefix_chars": t["n_prefix_chars"],
               "cands": sorted(cands), "arms": {}, "mc": {}, "agree": {}}
        for c in cands:
            cs = E.mc(c)
            row["mc"][c] = {"mc": cs["mc"], "n_content": cs["n_content"], "n_tokens": cs["n_tokens"]} if cs else None
        # anchor: Q refs' content-masked m_c (LOO for a q itself)
        row["anchor_mc"] = [(row["mc"].get(q) or {}).get("mc") for q in ARMS["Q3"]]
        for arm, S in ARMS.items():
            if not all(c in cands for c in S):
                continue
            loo = loo_values(E, S)
            if loo is None:
                continue
            legs = {}
            for m in cands:
                L = raw_legs(E, S, m, loo)
                if L is not None:
                    legs[m] = L
            row["arms"][arm] = {"loo": loo, "legs": legs}
        # stored-vs-Engy parity on Q3 (R of the king; LOO R of the q's)
        kp = t["king"]["pairs"]
        a_st = {}
        for p in kp:
            i = next((i for i in range(K) if abs(t["refs"][i]["lp_own"] - p["lpC_yc_zc"]) < 1e-12), None)
            if i is not None:
                a_st[i] = p["lpC_yc_za"] - p["lpC_yc_e"]
        row["stored"] = {
            "R_king": clme([a_st[i] for i in sorted(a_st)], TAU) if len(a_st) == K else None,
            "R_q_loo": [clme([t["refs"][i]["lp_cross"][j] - t["refs"][i]["lp_empty"] for i in range(K) if i != j], TAU)
                        for j in range(K)],
            "A_king": lme([(p["lpC_ya_zc"] - p["lpC_ya_e"]) * p["n_bytes_ya"] for p in kp], TAU),
            "mc_king": kp[0]["mc_za"], "mc_q": [r["mc_thought"] for r in t["refs"]],
        }
        # action agreement
        kind = t["dialect"]
        ys = {c: r["y"] for c, r in cands.items()}

        def fam_vs(a_ids, b_ids, loo_self=False):
            ex, jc = [], []
            for a in a_ids:
                if a not in ys:
                    continue
                tg = [ys[b] for b in b_ids if b in ys and b != a]
                if not tg:
                    continue
                ex.append(1.0 if exact(ys[a], tg, kind) else 0.0)
                jc.append(agree(ys[a], tg))
            return {"exact": _mean(ex), "jac": _mean(jc), "n": len(ex)}
        Q, G, O = ARMS["Q3"], ARMS["G3"], ARMS["O3"]
        row["agree"] = {"q_self": fam_vs(Q, Q), "g_self": fam_vs(G, G), "o_self": fam_vs(O, O),
                        "g_vs_q": fam_vs(G, Q), "o_vs_q": fam_vs(O, Q), "g3_vs_g": fam_vs(("g3",), G),
                        "g3_vs_q": fam_vs(("g3",), Q), "o3_vs_o": fam_vs(("o3",), O),
                        "king_vs_q": fam_vs(("king",), Q), "king_vs_g": fam_vs(("king",), G),
                        "king_vs_o": fam_vs(("king",), O), "o_vs_g": fam_vs(O, G)}
        rows.append(row)

    # ---------- σ per (arm, dialect) from the arm's LOO spread; σ_c per dialect from Q anchors
    sigma: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for arm in ARMS:
        for kind in ("text", "tool_call", "bash"):
            rs = [r for r in rows if r["dialect"] == kind and arm in r["arms"]]
            sigma[arm][kind] = {"R": pooled_sd([r["arms"][arm]["loo"]["R"] for r in rs]),
                                "A": pooled_sd([r["arms"][arm]["loo"]["A"] for r in rs]),
                                "n": len(rs)}
    sigma_c = {kind: pooled_sd([r["anchor_mc"] for r in rows if r["dialect"] == kind]) for kind in ("text", "tool_call", "bash")}
    # ---------- per-turn scores per arm per candidate
    for r in rows:
        kind = r["dialect"]
        anchor = r["anchor_mc"]
        r["scores"] = {}
        for arm, A in r["arms"].items():
            sg = sigma[arm][kind]
            sc = {}
            for m, L in A["legs"].items():
                zo = (cands_by_tid[r["turn_id"]].get(m) or {}).get("z_of", m)
                if zo in ARMS["Q3"]:
                    j = ARMS["Q3"].index(zo)
                    mu_c = _mean([v for i, v in enumerate(anchor) if i != j])
                else:
                    mu_c = _mean(anchor)
                typ = typ_of(L["mc"], L["n_content"], mu_c, sigma_c.get(kind))
                sc[m] = score_of(L, sg, typ)
            r["scores"][arm] = sc
        # family means per arm
        r["fam"] = {}
        for arm, sc in r["scores"].items():
            fm = {}
            for fam, members in FAMILIES.items():
                ms = [sc[m] for m in members if m in sc and sc[m]["score"] is not None]
                if not ms:
                    continue
                fm[fam] = {k: _mean([m_[k] for m_ in ms]) for k in LEGS}
                fm[fam]["bind"] = dict(collections.Counter(m_["bind"] for m_ in ms))
                fm[fam]["n_members"] = len(ms)
            r["fam"][arm] = fm
    write_jsonl(RESULTS / "turn_metrics.jsonl", rows)

    # ---------- aggregates
    groups = {"tau2/text": lambda r: r["group"] == "tau2" and r["dialect"] == "text",
              "tau2/tool_call": lambda r: r["group"] == "tau2" and r["dialect"] == "tool_call",
              "tau2/all": lambda r: r["group"] == "tau2",
              "bash": lambda r: r["group"] == "bash",
              "tau2/telecom": lambda r: r["source"] == "affine_tau2",
              "tau2/synth": lambda r: r["source"] == "affine_tau2_synth",
              "tau2/teacher_failed_task": lambda r: r["group"] == "tau2" and (r.get("task") or {}).get("teacher_n", 0) > 0
              and (r["task"]["teacher_solved"] < r["task"]["teacher_n"]),
              "tau2/teacher_solved_task": lambda r: r["group"] == "tau2" and (r.get("task") or {}).get("teacher_n", 0) > 0
              and (r["task"]["teacher_solved"] == r["task"]["teacher_n"])}
    fams = list(FAMILIES)
    rep: dict = {"n_turns": len(rows), "echo_errors": n_err, "engy_cost_usd": total_cost(),
                 "sigma": sigma, "sigma_c": sigma_c, "groups": {},
                 "mix": {"by_group_source_dialect": dict(collections.Counter(f"{r['group']}/{r['source']}/{r['dialect']}" for r in rows)),
                         "tau2_fault_classes": dict(collections.Counter(r["fault_class"] for r in rows if r["group"] == "tau2")),
                         "tau2_own_seat": dict(collections.Counter((r.get("task") or {}).get("own_seat") for r in rows if r["group"] == "tau2")),
                         "tau2_own_outcome": dict(collections.Counter((r.get("task") or {}).get("own_outcome") for r in rows if r["group"] == "tau2")),
                         "arms_available": {arm: sum(1 for r in rows if arm in r["arms"]) for arm in ARMS}}}
    for gname, pred in groups.items():
        rs = [r for r in rows if pred(r)]
        if not rs:
            continue
        g: dict = {"n": len(rs), "arms": {}}
        for arm in ARMS:
            ra = [r for r in rs if arm in r.get("fam", {})]
            if not ra:
                continue
            ga: dict = {"n": len(ra), "fam": {}, "paired": {}, "bind_A": {}, "inband": {}}
            for fam in fams:
                vals = [r["fam"][arm].get(fam) for r in ra]
                vals = [v for v in vals if v]
                if not vals:
                    continue
                ga["fam"][fam] = {k: _mean([v[k] for v in vals]) for k in LEGS}
                ga["fam"][fam]["n"] = len(vals)
                binds = collections.Counter()
                for v in vals:
                    binds.update(v["bind"])
                tot = sum(binds.values()) or 1
                ga["bind_A"][fam] = binds.get("A", 0) / tot
                ga["fam"][fam]["bind_frac"] = {k: binds.get(k, 0) / tot for k in ("R", "Gc", "A")}
                ga["inband"][fam] = _mean([1.0 if v["typ_c"] is not None and v["typ_c"] >= 0 else 0.0 for v in vals])
            for a, b in (("genesis", "teacher"), ("genesis_fresh", "teacher"), ("teacher", "king"), ("genesis", "king"),
                         ("genesis_fresh", "king"), ("occamy", "teacher"), ("occamy_fresh", "teacher"), ("occamy", "king"),
                         ("genesis_fresh", "genesis"), ("occamy", "genesis"), ("hybrid_qz_ga", "teacher"),
                         ("hybrid_qz_ga", "king"), ("hybrid_qz_ga", "genesis_fresh")):
                for leg in LEGS:
                    d = [(r["fam"][arm][a][leg] - r["fam"][arm][b][leg]) for r in ra
                         if a in r["fam"][arm] and b in r["fam"][arm]
                         and r["fam"][arm][a][leg] is not None and r["fam"][arm][b][leg] is not None]
                    ga["paired"][f"{a}-{b}|{leg}"] = _paired(d)
            g["arms"][arm] = ga
        # cross-arm: same family, G3 − Q3 (does the pool arm lift the family?)
        g["cross_arm"] = {}
        for arm in ("G3", "MIX", "O3"):
            for fam in fams:
                for leg in LEGS:
                    d = [(r["fam"][arm][fam][leg] - r["fam"]["Q3"][fam][leg]) for r in rs
                         if arm in r.get("fam", {}) and "Q3" in r["fam"] and fam in r["fam"][arm] and fam in r["fam"]["Q3"]
                         and r["fam"][arm][fam][leg] is not None and r["fam"]["Q3"][fam][leg] is not None]
                    if d:
                        g["cross_arm"][f"{arm}-Q3|{fam}|{leg}"] = _paired(d)
        # agreement
        g["agree"] = {k: {"exact": _mean([r["agree"][k]["exact"] for r in rs]), "jac": _mean([r["agree"][k]["jac"] for r in rs]),
                          "n": sum(1 for r in rs if r["agree"][k]["n"])} for k in rs[0]["agree"]}
        # typ_c of genesis / occamy thoughts against the Q anchor (per member, not family mean)
        g["typ_members"] = {}
        for tag, fam in (("g", "genesis+fresh"), ("o", "occamy+fresh"), ("king", "king"), ("q", "teacher")):
            vals, dm = [], []
            for r in rs:
                sc = r["scores"].get("Q3") or {}
                mu_all = _mean(r["anchor_mc"])
                for m, s in sc.items():
                    if m.startswith(tag) and s["typ_c"] is not None:
                        vals.append(s["typ_c"])
                        if s.get("mc") is not None and mu_all is not None and tag != "q":
                            dm.append(s["mc"] - mu_all)
            if vals:
                g["typ_members"][fam] = {"mean": st.mean(vals), "inband": st.mean(1.0 if v >= 0 else 0.0 for v in vals),
                                         "p10": sorted(vals)[int(0.1 * len(vals))], "n": len(vals),
                                         "mc_minus_anchor": _mean(dm), "below_share": (st.mean(1.0 if d < 0 else 0.0 for d in dm) if dm else None)}
        # parity Q3 stored vs Engy
        g["parity"] = {
            "R_king_engy_minus_stored": _paired([r["arms"]["Q3"]["legs"]["king"]["R"] - r["stored"]["R_king"] for r in rs
                                                 if "Q3" in r["arms"] and "king" in r["arms"]["Q3"]["legs"] and r["stored"]["R_king"] is not None]),
            "A_king_engy_minus_stored": _paired([r["arms"]["Q3"]["legs"]["king"]["A"] - r["stored"]["A_king"] for r in rs
                                                 if "Q3" in r["arms"] and "king" in r["arms"]["Q3"]["legs"]]),
            "R_qloo_engy_minus_stored": _paired([st.mean(r["arms"]["Q3"]["loo"]["R"]) - st.mean(r["stored"]["R_q_loo"]) for r in rs if "Q3" in r["arms"]]),
            "mc_king_engy_minus_stored": _paired([r["mc"]["king"]["mc"] - r["stored"]["mc_king"] for r in rs if r["mc"].get("king") and r["mc"]["king"]["mc"] is not None]),
        }
        rep["groups"][gname] = g
    # sample yields
    yields = {}
    for fam in ("gen", "occ"):
        ss = [s for s in read_jsonl(SAMPLES) if s.get("family") == fam]
        ok = [s for s in ss if "error" not in s]
        if ss:
            yields[fam] = {"n": len(ss), "parsed": sum(1 for s in ok if s["parsed"]), "cap_hits": sum(1 for s in ok if s["finish"] == "length"),
                           "text_fallback": sum(1 for s in ok if s.get("kind_used") == "text"), "errors": len(ss) - len(ok),
                           "turns_with_3refs": sum(1 for r in rows if all(f"{fam[0]}{i}" in r["cands"] for i in range(3))),
                           "turns_with_4": sum(1 for r in rows if f"{fam[0]}3" in r["cands"]),
                           "len_z_p50": st.median([len(s["z"]) for s in ok if s["parsed"]]) if any(s["parsed"] for s in ok) else None,
                           "len_y_p50": st.median([len(s["y"]) for s in ok if s["parsed"]]) if any(s["parsed"] for s in ok) else None}
    rep["yields"] = yields
    json.dump(rep, open(RESULTS / "report.json", "w"), indent=1)
    txt = render(rep)
    (RESULTS / "report.txt").write_text(txt)
    print(txt)


def fz(p: dict | None, digits: int = 2) -> str:
    if not p or p.get("mean") is None:
        return "     —      "
    z = p.get("z")
    return f"{p['mean']:+.{digits}f} (z{z:+.1f}, n{p['n']})" if z is not None else f"{p['mean']:+.{digits}f} (n{p['n']})"


def render(rep: dict) -> str:
    L: list[str] = []
    w = L.append
    w("ROUTED REFERENCES PROBE — Qwen echoes + Qwen typ_c anchor; z_R / z_A refs from a pool member")
    w(f"turns {rep['n_turns']}; echo errors {rep['echo_errors']}; Engy $ {rep['engy_cost_usd']:.2f}")
    w("mix: " + json.dumps(rep["mix"]["by_group_source_dialect"]))
    w("tau2 fault classes: " + json.dumps(rep["mix"]["tau2_fault_classes"]))
    w("tau2 own seat: " + json.dumps(rep["mix"]["tau2_own_seat"]) + "  own outcome: " + json.dumps(rep["mix"]["tau2_own_outcome"]))
    w("arms available (turns): " + json.dumps(rep["mix"]["arms_available"]))
    w("sample yields: " + json.dumps(rep.get("yields")))
    w("sigma (arm x dialect): " + json.dumps({a: {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in d.items()} for a, d in rep["sigma"].items()}))
    w("sigma_c (Q anchor): " + json.dumps({k: (round(v, 4) if v else v) for k, v in rep["sigma_c"].items()}))
    for gname in ("tau2/all", "tau2/text", "tau2/tool_call", "tau2/telecom", "tau2/synth", "tau2/teacher_failed_task",
                  "tau2/teacher_solved_task", "bash"):
        g = rep["groups"].get(gname)
        if not g:
            continue
        w("")
        w("=" * 110)
        w(f"GROUP {gname}  n={g['n']}")
        w("=" * 110)
        w("(a) HEADROOM / (b) SEPARATION / (d) INCUMBENCY — family means of turn = min(z_R, typ_c, z_A) and paired z")
        for arm, ga in g["arms"].items():
            w(f"-- arm {arm} (n={ga['n']}) --")
            w(f"   {'family':14s} {'min':>7s} {'z_R':>7s} {'typ_c':>7s} {'z_A':>7s}  bind R/Gc/A        inband")
            for fam, v in ga["fam"].items():
                bf = v["bind_frac"]
                w(f"   {fam:14s} {v['score']:+7.2f} {v['z_R']:+7.2f} {v['typ_c']:+7.2f} {v['z_A']:+7.2f}  "
                  f"{bf['R']:.2f}/{bf['Gc']:.2f}/{bf['A']:.2f}   {ga['inband'][fam]:.2f}  (n{v['n']})")
            for pair in ("genesis-teacher", "genesis_fresh-teacher", "occamy-teacher", "occamy_fresh-teacher",
                         "teacher-king", "genesis-king", "genesis_fresh-king", "occamy-king", "genesis_fresh-genesis", "occamy-genesis",
                         "hybrid_qz_ga-teacher", "hybrid_qz_ga-king", "hybrid_qz_ga-genesis_fresh"):
                ps = {leg: ga["paired"].get(f"{pair}|{leg}") for leg in LEGS}
                if all(p is None or p.get("mean") is None for p in ps.values()):
                    continue
                w(f"   paired {pair:24s} min {fz(ps['score'])}  z_R {fz(ps['z_R'])}  typ_c {fz(ps['typ_c'])}  z_A {fz(ps['z_A'])}")
        w("(e) cross-arm, same family: arm − Q3 (does the pool arm lift the family?)")
        for key, p in g["cross_arm"].items():
            arm, fam, leg = key.split("|")
            if leg == "score" or (leg in ("z_R", "z_A") and fam in ("teacher", "king", "genesis", "genesis_fresh", "occamy", "hybrid_qz_ga")):
                w(f"   {arm:7s} {fam:14s} {leg:6s} {fz(p)}")
        w("(c) TENSION: typ_c of member thoughts against the Qwen anchor (Q3 scoring)")
        for fam, v in g["typ_members"].items():
            extra = (f"  mc−anchor {v['mc_minus_anchor']:+.3f} nat/tok, below-band share {v['below_share']:.2f}"
                     if v.get("mc_minus_anchor") is not None else "")
            w(f"   {fam:14s} mean typ_c {v['mean']:+.2f}  in-band {v['inband']:.2f}  p10 {v['p10']:+.2f}  (n{v['n']}){extra}")
        w("(f) action agreement (norm-exact any / best token-Jaccard)")
        for k, v in g["agree"].items():
            if v["exact"] is not None:
                w(f"   {k:10s} exact {v['exact']:.2f}  jac {v['jac']:.2f}  (n{v['n']})")
        w("parity Q3 stored-vs-probe-echo (same tensors, different vLLM host): "
          + "; ".join(f"{k} {fz(v, 4)}" for k, v in g["parity"].items()))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("turns")
    p = sub.add_parser("sample")
    p.add_argument("--family", choices=("gen", "occ"), required=True)
    p.add_argument("--turns", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=16)
    p = sub.add_parser("echo")
    p.add_argument("--turns", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--pod", default="", help="endpoint json of a rented teacher pod (echo there instead of Engy)")
    sub.add_parser("report")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"turns": cmd_turns, "sample": cmd_sample, "echo": cmd_echo, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
