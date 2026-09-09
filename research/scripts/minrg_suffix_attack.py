"""Model-free suffix attack on a GLM teacher (self-contained).

Reproduces the live reign41 exploit by hand and measures it under both rules.
For each held-out turn (precomputed refs + grounding band in the refs file):

  honest      = teacher ref thought z_C            (task-specific, in-band)
  h_suffix    = honest z_C + the filler suffix     (the exploit, layered)
  stub_suffix = short task stub + filler suffix     (mimics the king)
  suffix_only = the filler suffix alone
  boiler      = generic task-agnostic paragraph

Each thought z is scored on the teacher:
  R_i    = lpC(y_i | z) - lp_empty_i    per ref i   (y_i, lp_empty_i precomputed)
  lme    = tau*log mean_i exp(R_i/tau)              (v4 Reason -- the live rule)
  r_c    = lme - mean_i R_i                          (centered R, new-rule leg 1)
  m      = lpC(z | x) per byte
  G      = min(m-(mu-w), (mu+w)-m), w = max(2*sd, 0.002)   (banded G, leg 2)
  min_rg = min(r_c, G)

Mechanism claim: the suffix variants raise lme (>= honest) but crater G (leave
the band), so v4 crowns the exploit while min(R,G) rejects it.

Usage: minrg_suffix_attack.py --ports 8001,8002 --refs ... --prefixes ...
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import random
import statistics as st
import sys
import zlib
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MODEL = "zai-org/GLM-4.5-Air-FP8"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TAU = 0.03
BAND_C = 2.0
BAND_W_MIN = 0.002

FILLER_SUFFIX = (
    "\nI could either verify the current state or proceed with the change; "
    "the usual choice is to check first.\n"
    "The analysis is complete. Next command:"
)
STUB = "I need to understand the repository first. Let me check the files."
BOILER = (
    "Looking at the current state of the task, I need to carefully consider "
    "the best next step. The previous actions have provided useful context, "
    "and it is important to proceed methodically. I will review the relevant "
    "files and outputs and then run the appropriate command."
)

tok = AutoTokenizer.from_pretrained(MODEL)


def gen_prompt(prefix: list[dict]) -> str:
    p = tok.apply_chat_template(prefix, tokenize=False,
                                add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


async def _echo_span(client, port, full, span_start, span_bytes):
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    n_prompt = sum(1 for s, _ in enc["offset_mapping"] if s < span_start)
    payload = {"model": MODEL, "prompt": full, "max_tokens": 1,
               "temperature": 0, "echo": True, "logprobs": 0,
               "add_special_tokens": False}
    for attempt in range(3):
        try:
            r = await client.post(f"http://127.0.0.1:{port}/v1/completions",
                                  json=payload,
                                  timeout=httpx.Timeout(480, connect=10))
            if r.status_code == 400:
                return None
            r.raise_for_status()
            lp = r.json()["choices"][0]["logprobs"]["token_logprobs"]
            span = [x for x in lp[n_prompt:-1] if x is not None]
            return sum(span) / max(span_bytes, 1) if span else 0.0
        except (httpx.HTTPError, KeyError):
            if attempt == 2:
                return None
            await asyncio.sleep(2 * (attempt + 1))
    return None


async def score_action(client, port, gen, thought, action):
    full = gen + THINK_CLOSE + "\nTHOUGHT: " + thought + "\n\n" + action
    return await _echo_span(client, port, full, len(full) - len(action),
                            len(action.encode()))


async def score_thought(client, port, gen, thought):
    full = gen + THINK_CLOSE + "\nTHOUGHT: " + thought
    return await _echo_span(client, port, full, len(full) - len(thought),
                            len(thought.encode()))


def variants(honest_z: str) -> dict[str, str]:
    return {
        "honest": honest_z,
        "h_suffix": honest_z + FILLER_SUFFIX,
        "stub_suffix": STUB + FILLER_SUFFIX,
        "suffix_only": FILLER_SUFFIX.strip(),
        "boiler": BOILER,
    }


async def score_variant(client, port, gen, row, z) -> dict:
    lps = await asyncio.gather(*[score_action(client, port, gen, z, ref["y"])
                                 for ref in row["refs"]])
    m = await score_thought(client, port, gen, z)
    rs = [lp - ref["lp_empty"] for lp, ref in zip(lps, row["refs"])
          if lp is not None]
    if not rs or m is None:
        return {"lme": None, "r_c": None, "g": None, "min_rg": None, "m": m}
    lme = TAU * math.log(sum(math.exp(r / TAU) for r in rs) / len(rs))
    r_c = lme - sum(rs) / len(rs)
    w = max(BAND_C * row["sd"], BAND_W_MIN)
    g = min(m - (row["mu"] - w), (row["mu"] + w) - m)
    return {"lme": lme, "r_c": r_c, "g": g, "min_rg": min(r_c, g), "m": m}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8001")
    ap.add_argument("--refs", type=Path, default=Path("/root/minrg/train_refs.jsonl"))
    ap.add_argument("--prefixes", type=Path,
                    default=Path("/root/minrg/train_prefixes.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=Path("/root/minrg/suffix_attack.jsonl"))
    ap.add_argument("--n-turns", type=int, default=120)
    ap.add_argument("--conc", type=int, default=24)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]

    prefixes = {}
    with gzip.open(args.prefixes, "rt") as f:
        for line in f:
            r = json.loads(line)
            prefixes[r["turn_id"]] = r["prefix"]
    rows = []
    for line in open(args.refs):
        r = json.loads(line)
        if (r.get("mu") is not None and len(r.get("refs") or []) >= 2
                and r["turn_id"] in prefixes
                and r["refs"][0].get("z", "").strip()):
            rows.append(r)
    rng = random.Random(424242)
    picked = rng.sample(rows, min(args.n_turns, len(rows)))
    print(f"{len(picked)} turns, ports={ports}", file=sys.stderr, flush=True)

    sems = {p: asyncio.Semaphore(args.conc) for p in ports}
    lock = asyncio.Lock()
    out_f = open(args.out, "w")
    agg: dict[str, dict[str, list]] = {}
    n = 0

    async with httpx.AsyncClient() as client:
        async def do(row):
            nonlocal n
            port = ports[zlib.adler32(row["turn_id"].encode()) % len(ports)]
            gen = gen_prompt(prefixes[row["turn_id"]])
            vs = variants(row["refs"][0]["z"])
            async with sems[port]:
                for kind, z in vs.items():
                    sc = await score_variant(client, port, gen, row, z)
                    async with lock:
                        out_f.write(json.dumps(
                            {"turn_id": row["turn_id"], "kind": kind, **sc})
                            + "\n")
                        agg.setdefault(kind, {k: [] for k in
                                             ("lme", "r_c", "g", "min_rg", "m")})
                        for k in agg[kind]:
                            if sc[k] is not None:
                                agg[kind][k].append(sc[k])
            n += 1
            if n % 20 == 0:
                out_f.flush()
                print(f"{n}/{len(picked)}", file=sys.stderr, flush=True)
        await asyncio.gather(*[do(r) for r in picked])
    out_f.close()

    print("\nkind         n   lme      r_c      G        min_rg   m")
    for kind in ("honest", "h_suffix", "stub_suffix", "suffix_only", "boiler"):
        a = agg.get(kind)
        if not a:
            continue
        m = {k: (st.mean(a[k]) if a[k] else float("nan")) for k in a}
        print(f"{kind:12s} {len(a['min_rg']):3d} "
              f"{m['lme']:+.4f}  {m['r_c']:+.4f}  {m['g']:+.4f}  "
              f"{m['min_rg']:+.4f}  {m['m']:+.4f}")

    by_turn: dict = {}
    for line in open(args.out):
        r = json.loads(line)
        by_turn.setdefault(r["turn_id"], {})[r["kind"]] = r
    v4_wins = nrg_wins = npair = 0
    for d in by_turn.values():
        if ("honest" not in d or "h_suffix" not in d
                or d["honest"]["lme"] is None or d["h_suffix"]["lme"] is None):
            continue
        npair += 1
        v4_wins += d["h_suffix"]["lme"] > d["honest"]["lme"]
        nrg_wins += d["h_suffix"]["min_rg"] > d["honest"]["min_rg"]
    if npair:
        print(f"\nsuffix raises v4 Reason on {v4_wins}/{npair} "
              f"({100*v4_wins/npair:.0f}%)")
        print(f"suffix raises min(R,G)  on {nrg_wins}/{npair} "
              f"({100*nrg_wins/npair:.0f}%)")
    print("SUFFIX_ATTACK_DONE")


if __name__ == "__main__":
    asyncio.run(main())
