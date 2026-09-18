"""Pad-after-</think> attack arm under the as_generated rendering, run with the
NEW evalsrv code (chat.thought_body two-span echoes + content mask) against
the live teacher swarm router. Arms on each king thought of one stored duel:
honest (latent-only), repeat (visible = latent), tail (visible = last
paragraph), generic (fixed sentence). Refs rendered latent+visible via the
worker's heuristic split. Reports band position / in-band and typ_c per arm.

  python pad_arm.py --stage ~/wvk22_stage/affine --duel chal-00578 --per-kind bash=30,tool_call=20,text=15
"""
from __future__ import annotations

import argparse, asyncio, json, math, statistics as st, sys, time
from pathlib import Path

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True)
ap.add_argument("--duel", default="chal-00578")
ap.add_argument("--per-kind", default="bash=30,tool_call=20,text=15,boxed=5,terminus_json=5")
ap.add_argument("--teacher-url", default="http://127.0.0.1:9100/v1")
ap.add_argument("--turn-conc", type=int, default=8)
args = ap.parse_args()

sys.path.insert(0, args.stage)
sys.path.insert(0, "/home/const/scoring-align/p2")
sys.path.insert(0, "/home/const/g-rendering")
from evalsrv import chat, sdmeter  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402
from echo_render import heuristic_split, infer_kind, pick_turns  # noqa: E402
from aleg_turns import TurnStore  # noqa: E402
from panel_driver import CACHE, SOURCES, load_slice, materialize  # noqa: E402

TEACHER = "Qwen/Qwen3.8-27B"
chat.set_thought_rendering("as_generated")
GENERIC = "Let me run this and check the result."
per_kind = {k: int(v) for k, v in (kv.split("=") for kv in args.per_kind.split(","))}


def compose(latent: str, visible: str) -> str:
    lat, vis = latent.strip(), visible.strip()
    if vis:
        return (lat + chat.Z_SPLIT + vis) if lat else (chat.THINK_CLOSE + "\n" + vis)
    return lat


def last_par(z: str) -> str:
    pars = [p for p in z.strip().split("\n\n") if p.strip()]
    return pars[-1] if pars else z


async def main():
    http = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=15.0), limits=httpx.Limits(max_connections=64))
    teacher = VllmModel(Served(name="teacher", repo=TEACHER, revision=None, port=0, base_url=args.teacher_url), http, asyncio.Semaphore(48))
    store = TurnStore(CACHE, SOURCES)
    rec, _, _ = load_slice(args.duel)
    picked = pick_turns(rec, per_kind)
    prefixes, _ = materialize(store, rec, [p[0] for p in picked])
    sem = asyncio.Semaphore(args.turn_conc)
    out = []
    t0 = time.time()

    async def echo_pair(pre, z):
        x, e = await asyncio.gather(teacher.score_thought(pre, z, tokens=True), teacher.score_thought_uncond(z))
        cs = sdmeter.content_stats(x.get("tokens") or [], e.get("tokens") or [], 1.0)
        return {"m": x["lp_per_byte"], "n_tok": x["n_tokens"], "n_bytes": x["n_bytes"], "mc": cs["mc"], "n_content": cs["n_content"]}

    async def one(tid, kind, rl, kp, cp):
        pre = prefixes.get(tid)
        if pre is None or isinstance(pre, dict):
            return
        async with sem:
            try:
                refs = []
                for r in rl:
                    lat, vis = heuristic_split(r["z"], kind)
                    refs.append(await echo_pair(pre, compose(lat, vis)))
                kz = kp["z_a"]
                arms = {"honest": compose(kz, ""), "repeat": compose(kz, kz), "tail": compose(kz, last_par(kz)),
                        "generic": compose(kz, GENERIC), "chal_honest": compose(cp["z_a"], "")}
                # mirror arm: ref0 without its visible part
                lat0, vis0 = heuristic_split(rl[0]["z"], kind)
                arms["ref0_latent_only"] = compose(lat0, "")
                arms["ref0_as_is"] = compose(lat0, vis0)
                res = dict(zip(arms, await asyncio.gather(*[echo_pair(pre, z) for z in arms.values()])))
                out.append({"tid": tid, "kind": kind, "refs": refs, "arms": res})
            except Exception as ex:  # noqa: BLE001
                out.append({"tid": tid, "kind": kind, "error": f"{type(ex).__name__}:{str(ex)[:80]}"})
            print(f"{time.time()-t0:6.0f}s {len(out)}/{len(picked)}", flush=True)

    await asyncio.gather(*[one(*p) for p in picked])
    Path("/home/const/g-rendering/data/pad_arm.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")
    good = [o for o in out if "refs" in o]
    print(f"turns {len(good)} errors {len(out)-len(good)}")
    # sigma_c per kind (pooled within-turn over refs' mc), band per turn (band_c 4, floor 0.002)
    var = {}
    for o in good:
        mcs = [r["mc"] for r in o["refs"] if r["mc"] is not None]
        if len(mcs) == 3:
            var.setdefault(o["kind"], []).append(st.variance(mcs))
    sig = {k: math.sqrt(st.mean(v)) for k, v in var.items()}
    print("sigma_c (b, new code):", {k: round(v, 3) for k, v in sig.items()})
    names = ["honest", "repeat", "tail", "generic", "chal_honest", "ref0_latent_only", "ref0_as_is"]
    agg = {n: {"m": [], "pos": [], "inband": [], "typ": [], "typ_neg": []} for n in names}
    for o in good:
        ts = [r["m"] for r in o["refs"]]; mu = st.mean(ts); w = max(4 * st.stdev(ts), 0.002)
        mcs = [r["mc"] for r in o["refs"] if r["mc"] is not None]
        if len(mcs) < 3 or o["kind"] not in sig:
            continue
        muc = st.mean(mcs); s = sig[o["kind"]]
        for n in names:
            a = o["arms"][n]
            agg[n]["m"].append(a["m"]); agg[n]["pos"].append((a["m"] - mu) / w); agg[n]["inband"].append(abs(a["m"] - mu) <= w)
            if a["mc"] is not None and a["n_content"] >= 10:
                t = 2 - abs(a["mc"] - muc) / s; agg[n]["typ"].append(t); agg[n]["typ_neg"].append(t < 0)
    print(f"{'arm':18s} {'m/byte':>8s} {'band pos':>9s} {'in-band':>8s} {'typ_c':>7s} {'typ<0':>6s} {'above':>6s}")
    for n in names:
        g = agg[n]
        above = sum(1 for p in g["pos"] if p > 1) / len(g["pos"])
        print(f"{n:18s} {st.mean(g['m']):+8.4f} {st.mean(g['pos']):+9.2f} {sum(g['inband'])/len(g['inband']):8.2f} {st.mean(g['typ']):+7.2f} {sum(g['typ_neg'])/len(g['typ_neg']):6.2f} {above:6.2f}")
    # paired: arm − honest typ_c
    for n in ("repeat", "tail", "generic"):
        d = [a - b for a, b in zip(agg[n]["typ"], agg["honest"]["typ"])]
        print(f"paired typ_c {n} − honest: {st.mean(d):+.2f} ± {st.stdev(d)/math.sqrt(len(d)):.2f}")
    print("echo stats", json.dumps(teacher.echo_stats()))

asyncio.run(main())
