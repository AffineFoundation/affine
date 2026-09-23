"""X2 — are verified-solved teacher thoughts already inside the blind band?

The outcome probe (results/frontier_arbiter/outcome/) continued kept states
with the teacher (arm T) to a graded outcome. Its stored traces carry the
FIRST reply of each continuation with `reasoning_content`, i.e. a teacher
thought whose downstream outcome is known. Here every such thought is
echoed under the state's prefix and compared with a fresh blind band (3
teacher refs sampled now at the same state).

    python solved_band.py sample     # 3 blind refs per state (Engy qwen3.8-27b, T .8)
    python solved_band.py echo       # lp_thought(tokens) + uncond for first thoughts + blind refs
    python solved_band.py report

Terms (one line each):
  state            a kept prefix of the outcome probe (rollout, turn_idx) with its wire messages
                   (+ tool schemas for the verifiers `bash` harness).
  first thought    z of the arm-T continuation's first sampled reply (reasoning_content +
                   visible prose before the action), parsed in the state's dialect with the
                   live flags; textbased mswea fences are mapped to ```bash like the fold.
  outcome          the continuation's env grade: solved (reward 1) / failed.
  blind band       μ_c = mean m_c of 3 fresh blind teacher refs at the state; σ_c pooled per
                   dialect = sqrt(mean within-state variance of the 3 blind m_c) (live loo rule);
                   also the live verdict σ_Mc(dialect) from privileged_refs/turns.jsonl.
  m_c              content-masked mean token logprob of z under x (live content_stats, θ = 1).
  in-band          |m_c − μ_c| ≤ 2σ_c AND ≥ 10 content tokens (live typ_c ≥ 0).
  prefix rendering baked plain text via affine.toolbake.ToolBaker when the state has tool
                   traffic (what the live corpus stores), else the wire messages as-is.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    REPO, TEACHER_ENGY, Engy, TeacherEcho, append_jsonl, read_jsonl, render_tool_call,
    repair_tool_call, teacher_tokenizer, write_jsonl,
)
from privileged_refs import LIVE_SPLIT, TURNS  # noqa: E402
from affine import dialects  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402
from evalsrv import chat as live_chat  # noqa: E402
from evalsrv.sdmeter import content_stats  # noqa: E402

live_chat.set_thought_rendering("as_generated")

OUTCOME = REPO / "research" / "results" / "frontier_arbiter" / "outcome"
OUT = REPO / "research" / "results" / "frontier_arbiter" / "inband"
TRACES = Path("/tmp/fa_outcome/collect/traces")
X2_STATES = OUT / "x2_states.jsonl"
X2_SAMPLES = OUT / "x2_samples.jsonl"
X2_ECHOES = OUT / "x2_echoes.jsonl"
COST = OUT / "cost.jsonl"
K = 3
MSWEA_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
HARNESS_KIND = {"mini_swe_textbased": "bash", "bash": "tool_call", "terminus_2": "terminus_json"}


def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "stage": "x2_" + stage, "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    print(f"  [$] x2 {stage}: this run ${engy.cost_usd:.3f} {note}", flush=True)


# ------------------------------------------------------------------ states
def wire_message(m: dict) -> dict:
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m["role"] == "tool":
        for k in ("tool_call_id", "name"):
            if m.get(k):
                out[k] = m[k]
    return out


def reply_zy(reasoning: str, content: str, tool_calls: list | None, harness: str, kind: str) -> dict:
    content = content or ""
    if harness == "mini_swe_textbased":
        content = MSWEA_RE.sub("```bash\n", content)
    if tool_calls:
        content = content.rstrip() + "\n" + "\n".join(render_tool_call(c if "function" in c else
                                                                       {"function": {"name": c.get("name"), "arguments": c.get("arguments")}})
                                                      for c in tool_calls)
    if kind == "tool_call":
        content, _ = repair_tool_call(content)
    text = (reasoning or "") + "\n" + live_chat.THINK_CLOSE + "\n" + content
    z, y = live_chat.split_rollout(text, kind, **LIVE_SPLIT)
    return {"z": z, "y": y, "parsed": bool(y), "visible": live_chat.THINK_CLOSE in z,
            "reasoning_chars": len(reasoning or ""), "content_chars": len(content)}


def build_states() -> list[dict]:
    rows = read_jsonl(OUTCOME / "continuations.jsonl")
    kept = {k["state_id"]: k for k in read_jsonl(OUTCOME / "kept.jsonl")}
    baker = ToolBaker(teacher_tokenizer())
    out = []
    skipped = collections.Counter()
    for r in rows:
        if r.get("arm") != "T" or r.get("outcome") not in ("solved", "failed"):
            continue
        rid, ti, arm = r["state_id"].split(":")
        p = TRACES / f"{rid}_{ti}_{arm}.json"
        if not p.exists():
            skipped["no_trace"] += 1
            continue
        tr = json.loads(p.read_text())
        nodes = tr["nodes"]
        fi = next((i for i, n in enumerate(nodes) if n.get("sampled") and n["message"]["role"] == "assistant"), None)
        if fi is None:
            skipped["no_sampled_reply"] += 1
            continue
        m = nodes[fi]["message"]
        if not m.get("reasoning_content"):
            skipped["no_reasoning"] += 1
            continue
        stt = json.loads(Path(kept[r["kept_state_id"]]["path"]).read_text())
        harness = stt["harness"]
        kind = HARNESS_KIND[harness]
        msgs = [wire_message(x) for x in stt["messages"]]
        tools = stt.get("tools") or []
        has_tools = bool(tools) or any(x["role"] == "tool" or x.get("tool_calls") for x in msgs)
        if has_tools:
            baked = baker.bake(msgs, tools)
            parity = baker.parity_ok(msgs, tools, baked)
        else:
            baked = [{"role": x["role"], "content": x["content"]} for x in msgs if x["role"] != "tool"]
            parity = True
        first = reply_zy(m.get("reasoning_content") or "", m.get("content") or "", m.get("tool_calls"), harness, kind)
        out.append({"state_id": r["state_id"], "kept_state_id": r["kept_state_id"], "harness": harness, "kind": kind,
                    "outcome": r["outcome"], "orig_outcome": stt.get("orig_outcome"), "source": stt.get("source"),
                    "depth": stt.get("depth"), "n_turns_cont": r.get("n_turns"), "stop": r.get("stop_condition"),
                    "prefix": baked, "parity": parity, "wire": msgs, "tools": tools, "first": first,
                    "first_finish": next((c.get("finish_reason") for c in tr.get("calls", []) if c.get("node") == fi), None)})
    print(f"x2 states: {len(out)} ({collections.Counter(s['outcome'] for s in out)}; "
          f"{collections.Counter(s['harness'] for s in out)}); skipped {dict(skipped)}; "
          f"parity fails {sum(1 for s in out if not s['parity'])}; first parsed {sum(1 for s in out if s['first']['parsed'])}")
    write_jsonl(X2_STATES, out)
    return out


def load_states() -> list[dict]:
    return read_jsonl(X2_STATES) if X2_STATES.exists() else build_states()


# ------------------------------------------------------------------ sample
async def sample_blind(states: list[dict], engy: Engy, max_tokens: int = 4096, retry_cap: bool = False) -> None:
    prev = [r for r in read_jsonl(X2_SAMPLES) if "error" not in r]
    # --retry-cap: refs that hit the token cap unparsed are re-sampled with a larger cap
    # (the outcome probe sampled at 16384); the capped row is superseded (last row wins).
    done = {(r["state_id"], r["i"]) for r in prev
            if not (retry_cap and not r.get("parsed") and r.get("finish") == "length")}
    jobs = [(s, i) for s in states for i in range(K) if (s["state_id"], i) not in done]
    print(f"x2 blind refs: {len(jobs)} to sample (max_tokens {max_tokens})")

    async def one(s, i):
        extra = {"tools": s["tools"]} if s["harness"] == "bash" and s["tools"] else {}
        try:
            r = await engy.chat(TEACHER_ENGY, s["wire"], temperature=0.8, max_tokens=max_tokens, **extra)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(X2_SAMPLES, {"state_id": s["state_id"], "i": i, "error": repr(ex)[:300]})
            return
        zy = reply_zy(r["reasoning"], r["content"], r["tool_calls"], s["harness"], s["kind"])
        append_jsonl(X2_SAMPLES, {"state_id": s["state_id"], "i": i, "finish": r["finish"], "usage": r["usage"],
                                  "cost_usd": r["cost_usd"], **zy})

    for k in range(0, len(jobs), 48):
        await asyncio.gather(*[one(*j) for j in jobs[k:k + 48]])
        log_cost("sample", engy, f"{min(k + 48, len(jobs))}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    states = build_states()
    engy = Engy(concurrency=24)
    asyncio.run(sample_blind(states, engy, args.max_tokens, args.retry_cap))


# ------------------------------------------------------------------ echo
async def echo_all(states: list[dict], engy: Engy) -> None:
    te = TeacherEcho(engy)
    samples = [r for r in read_jsonl(X2_SAMPLES) if "error" not in r and r.get("parsed")]
    done = {r["key"] for r in read_jsonl(X2_ECHOES) if "error" not in r}
    jobs = []
    for s in states:
        sid = s["state_id"]
        if s["first"]["parsed"]:
            for kind, tag in (("thought", "th"), ("uncond", "un")):
                key = f"{sid}|{tag}|first|0"
                if key not in done:
                    jobs.append((key, kind, s["prefix"], s["first"]["z"]))
        for r in samples:
            if r["state_id"] != sid:
                continue
            for kind, tag in (("thought", "th"), ("uncond", "un")):
                key = f"{sid}|{tag}|blind|{r['i']}"
                if key not in done:
                    jobs.append((key, kind, s["prefix"], r["z"]))
    print(f"x2 echo: {len(jobs)}")

    async def one(key, kind, prefix, z):
        try:
            r = await te.lp_thought(prefix, z, tokens=True) if kind == "thought" else await te.lp_thought_uncond(z, tokens=True)
            append_jsonl(X2_ECHOES, {"key": key, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(X2_ECHOES, {"key": key, "error": repr(ex)[:300]})

    n_done = 0
    tasks = [asyncio.ensure_future(one(*j)) for j in jobs]
    for fut in asyncio.as_completed(tasks):
        await fut
        n_done += 1
        if n_done % 48 == 0 or n_done == len(jobs):
            log_cost("echo", engy, f"{n_done}/{len(jobs)}")


def cmd_echo(args: argparse.Namespace) -> None:
    engy = Engy(concurrency=24)
    asyncio.run(echo_all(load_states(), engy))


# ------------------------------------------------------------------ report
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _p50(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def _rate(f):
    f = [x for x in f if x is not None]
    return (sum(1.0 for x in f if x) / len(f)) if f else None


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{p}f}"


def mc_of(ex, eu):
    if not ex or not eu:
        return None, None
    cs = content_stats([tuple(x) for x in ex["tokens"]], [tuple(x) for x in eu["tokens"]], 1.0)
    return cs["mc"], cs["n_content"]


def cmd_report(args: argparse.Namespace) -> None:
    states = load_states()
    samples = {(r["state_id"], r["i"]): r for r in read_jsonl(X2_SAMPLES) if "error" not in r}
    echoes = {r["key"]: r for r in read_jsonl(X2_ECHOES) if "error" not in r}
    live_sigma = {}
    for t in read_jsonl(TURNS):
        if t.get("sigma_mc"):
            live_sigma.setdefault(t["dialect"], t["sigma_mc"])
    rows = []
    for s in states:
        sid = s["state_id"]
        mcb, ncb = [], []
        for i in range(K):
            m, n = mc_of(echoes.get(f"{sid}|th|blind|{i}"), echoes.get(f"{sid}|un|blind|{i}"))
            if m is not None:
                mcb.append(m)
                ncb.append(n)
        mf, nf = mc_of(echoes.get(f"{sid}|th|first|0"), echoes.get(f"{sid}|un|first|0"))
        rows.append({"state_id": sid, "harness": s["harness"], "kind": s["kind"], "outcome": s["outcome"],
                     "orig_outcome": s["orig_outcome"], "depth": s["depth"], "mc_blind": mcb, "nc_blind": ncb,
                     "mc_first": mf, "nc_first": nf, "len_first": len(s["first"]["z"]),
                     "visible_first": s["first"]["visible"], "parsed_first": s["first"]["parsed"],
                     "len_blind": [len(samples[(sid, i)]["z"]) for i in range(K) if (sid, i) in samples],
                     "visible_blind": [samples[(sid, i)]["visible"] for i in range(K) if (sid, i) in samples],
                     "blind_parsed": sum(1 for i in range(K) if (sid, i) in samples and samples[(sid, i)]["parsed"]),
                     "first_finish": s.get("first_finish")})
    # pooled sigma per dialect from the blind triples
    sig_pooled = {}
    for kd in ("bash", "tool_call", "terminus_json"):
        vs = [st.variance(r["mc_blind"]) for r in rows if r["kind"] == kd and len(r["mc_blind"]) == K]
        sig_pooled[kd] = math.sqrt(st.mean(vs)) if vs else None
    for r in rows:
        ok = len(r["mc_blind"]) >= 2 and r["mc_first"] is not None
        mu = st.mean(r["mc_blind"]) if ok else None
        r["mu_blind"] = mu
        for tag, sig in (("pooled", sig_pooled.get(r["kind"])), ("live", live_sigma.get(r["kind"]))):
            z = ((r["mc_first"] - mu) / sig) if (ok and sig) else None
            r[f"z_{tag}"] = z
            r[f"inband_{tag}"] = (z is not None and abs(z) <= 2 and (r["nc_first"] or 0) >= 10)
        # blind LOO z (each blind ref vs the other two), live sigma
        loo = []
        if len(r["mc_blind"]) == K and live_sigma.get(r["kind"]):
            for j in range(K):
                others = [r["mc_blind"][i] for i in range(K) if i != j]
                loo.append((r["mc_blind"][j] - st.mean(others)) / live_sigma[r["kind"]])
        r["blind_loo_z_live"] = loo
        r["blind_loo_inband_live"] = _rate([abs(z) <= 2 and n >= 10 for z, n in zip(loo, r["nc_blind"])]) if loo else None
    write_jsonl(OUT / "x2_rows.jsonl", rows)

    def agg(rs: list[dict]) -> dict:
        return {"n": len(rs),
                "inband_live": _rate([r["inband_live"] for r in rs if r["z_live"] is not None]),
                "inband_pooled": _rate([r["inband_pooled"] for r in rs if r["z_pooled"] is not None]),
                "z_live_mean": _mean([r["z_live"] for r in rs]), "z_live_p50": _p50([r["z_live"] for r in rs]),
                "z_live_below": _rate([r["z_live"] < -2 for r in rs if r["z_live"] is not None]),
                "z_live_above": _rate([r["z_live"] > 2 for r in rs if r["z_live"] is not None]),
                "absz_live_p50": _p50([abs(r["z_live"]) for r in rs if r["z_live"] is not None]),
                "z_pooled_mean": _mean([r["z_pooled"] for r in rs]),
                "content_floor": _rate([(r["nc_first"] or 0) < 10 for r in rs if r["mc_first"] is not None]),
                "len_first_p50": _p50([r["len_first"] for r in rs]), "len_first_mean": _mean([r["len_first"] for r in rs]),
                "len_blind_p50": _p50([x for r in rs for x in r["len_blind"]]),
                "visible_first": _rate([r["visible_first"] for r in rs]),
                "visible_blind": _rate([x for r in rs for x in r["visible_blind"]]),
                "blind_loo_inband_live": _mean([r["blind_loo_inband_live"] for r in rs]),
                "mc_first_mean": _mean([r["mc_first"] for r in rs]), "mu_blind_mean": _mean([r["mu_blind"] for r in rs]),
                "n_scored": sum(1 for r in rs if r["z_live"] is not None)}

    rep = {"n_states": len(rows), "sigma_pooled": sig_pooled, "sigma_live": live_sigma,
           "first_parsed": sum(1 for r in rows if r["parsed_first"]),
           "blind_parsed": sum(r["blind_parsed"] for r in rows), "blind_sampled": len(samples),
           "echo_errors": sum(1 for r in read_jsonl(X2_ECHOES) if "error" in r),
           "by_outcome": {o: agg([r for r in rows if r["outcome"] == o]) for o in ("solved", "failed")},
           "all": agg(rows),
           "by_kind_outcome": {kd: {o: agg([r for r in rows if r["outcome"] == o and r["kind"] == kd]) for o in ("solved", "failed")}
                               for kd in ("bash", "tool_call", "terminus_json")},
           "by_orig_outcome": {o: agg([r for r in rows if r["orig_outcome"] == o]) for o in ("solved", "failed")}}
    # two-proportion z for in-band(solved) − in-band(failed), and a Welch z on z_live means
    so = [r for r in rows if r["outcome"] == "solved" and r["z_live"] is not None]
    fa = [r for r in rows if r["outcome"] == "failed" and r["z_live"] is not None]
    if so and fa:
        p1, p2 = _rate([r["inband_live"] for r in so]), _rate([r["inband_live"] for r in fa])
        pp = (sum(r["inband_live"] for r in so) + sum(r["inband_live"] for r in fa)) / (len(so) + len(fa))
        se = math.sqrt(pp * (1 - pp) * (1 / len(so) + 1 / len(fa))) if 0 < pp < 1 else None
        rep["inband_diff_z"] = ((p1 - p2) / se) if se else None
        zs, zf = [r["z_live"] for r in so], [r["z_live"] for r in fa]
        if len(zs) > 1 and len(zf) > 1:
            sew = math.sqrt(st.variance(zs) / len(zs) + st.variance(zf) / len(zf))
            rep["z_live_mean_diff_welch_z"] = ((st.mean(zs) - st.mean(zf)) / sew) if sew else None
    cost = 0.0
    run_max = 0.0
    last_stage = None
    for c in read_jsonl(COST):
        if not c["stage"].startswith("x2_"):
            continue
        if c["stage"] != last_stage or c["cost_usd"] < run_max:
            cost += run_max
            run_max = 0.0
        run_max = max(run_max, c["cost_usd"])
        last_stage = c["stage"]
    cost += run_max
    rep["cost_usd"] = cost
    (OUT / "x2_report.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = []
    P = lines.append
    P("X2 — verified-outcome teacher first thoughts vs the blind band")
    P(f"states {rep['n_states']} (arm T continuations with a stored trace carrying reasoning_content; outcome solved/failed); "
      f"first thoughts parsed {rep['first_parsed']}; blind refs sampled {rep['blind_sampled']} parsed {rep['blind_parsed']}; echo errors {rep['echo_errors']}; $ {cost:.2f}")
    P(f"σ_c pooled here (sqrt mean within-state var of 3 blind m_c): {({k: round(v, 3) if v else None for k, v in sig_pooled.items()})}; live verdict σ_Mc: {({k: round(v, 3) for k, v in live_sigma.items()})}")
    P("Terms: in-band(live) = |m_c(first) − mean_3 m_c(blind)| ≤ 2·σ_Mc(dialect, live verdict) and ≥10 content tokens; in-band(pooled) uses σ pooled on these states;")
    P("       z = (m_c(first) − μ_blind)/σ; below/above = z < −2 / z > 2; blindLOO = each fresh blind ref vs the other two (the band's own hit rate);")
    P("       len = chars of z (latent + visible); visible = z carries prose after </think>.")
    P(f"{'group':<26} {'n':>3} {'nsc':>3} {'inband':>6} {'inbP':>6} {'zmean':>6} {'zp50':>6} {'|z|p50':>6} {'below':>6} {'above':>6} {'<10ct':>6} {'lenp50':>6} {'lenmn':>6} {'lenBl':>6} {'visF':>6} {'visB':>6} {'blLOO':>6} {'mcF':>7} {'muB':>7}")

    def line(name, g):
        P(f"{name:<26} {g['n']:>3} {g['n_scored']:>3} {_f(g['inband_live'])} {_f(g['inband_pooled'])} {_f(g['z_live_mean'],6,2)} {_f(g['z_live_p50'],6,2)} {_f(g['absz_live_p50'],6,2)} "
          f"{_f(g['z_live_below'])} {_f(g['z_live_above'])} {_f(g['content_floor'])} {_f(g['len_first_p50'],6,0)} {_f(g['len_first_mean'],6,0)} {_f(g['len_blind_p50'],6,0)} "
          f"{_f(g['visible_first'])} {_f(g['visible_blind'])} {_f(g['blind_loo_inband_live'])} {_f(g['mc_first_mean'],7,3)} {_f(g['mu_blind_mean'],7,3)}")
    line("ALL", rep["all"])
    for o in ("solved", "failed"):
        line(f"outcome {o}", rep["by_outcome"][o])
    for kd, d in rep["by_kind_outcome"].items():
        for o in ("solved", "failed"):
            if d[o]["n"]:
                line(f"{kd} {o}", d[o])
    for o in ("solved", "failed"):
        line(f"orig-traj {o}", rep["by_orig_outcome"][o])
    P(f"in-band(solved) − in-band(failed): two-proportion z {_f(rep.get('inband_diff_z'),6,2)}; z_live mean difference Welch z {_f(rep.get('z_live_mean_diff_welch_z'),6,2)}")
    txt = "\n".join(lines) + "\n"
    (OUT / "x2_report.txt").write_text(txt)
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sample", cmd_sample), ("echo", cmd_echo), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name == "sample":
            p.add_argument("--max-tokens", type=int, default=4096)
            p.add_argument("--retry-cap", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
