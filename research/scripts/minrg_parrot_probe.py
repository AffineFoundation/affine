"""Upper-band (parrot) probe for min(R, G) (self-contained, echo-only).

The suffix attack pushes the thought likelihood BELOW the band (low-prob
filler). The band is two-sided, so this probes the OTHER side: thoughts whose
likelihood sits ABOVE the band (mu + w) should also be rejected.

Per held-out turn, constructs high-likelihood thoughts and scores them the
same way as the suffix attack:
  parrot     = tail of the last user message (restate the task)
  ref_x3     = honest ref thought repeated 3x (inflate m by repetition)
  generic    = a maximally generic high-frequency opener

Reports mean lme / r_c / G / min(R,G) / m per variant and how often honest
beats each. Uses the shared refs + prefixes on the train box.
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

GENERIC = ("Let me carefully look at the task and the current state, then "
           "decide the best next step and run the appropriate command.")

tok = AutoTokenizer.from_pretrained(MODEL)


def gen_prompt(prefix: list[dict]) -> str:
    p = tok.apply_chat_template(prefix, tokenize=False,
                                add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


async def _echo(client, port, full, span_start, span_bytes):
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


async def action_lp(client, port, gen, thought, action):
    full = gen + THINK_CLOSE + "\nTHOUGHT: " + thought + "\n\n" + action
    return await _echo(client, port, full, len(full) - len(action),
                       len(action.encode()))


async def thought_lp(client, port, gen, thought):
    full = gen + THINK_CLOSE + "\nTHOUGHT: " + thought
    return await _echo(client, port, full, len(full) - len(thought),
                       len(thought.encode()))


def variants(honest_z: str, prefix: list[dict]) -> dict[str, str]:
    last_user = next((m["content"] for m in reversed(prefix)
                      if m.get("role") == "user"), "")
    parrot = last_user.strip()[-600:]
    return {
        "honest": honest_z,
        "parrot": parrot or honest_z,
        "ref_x3": (honest_z + " ") * 3,
        "generic": GENERIC,
    }


async def score(client, port, gen, row, z) -> dict:
    lps = await asyncio.gather(*[action_lp(client, port, gen, z, ref["y"])
                                 for ref in row["refs"]])
    m = await thought_lp(client, port, gen, z)
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
    ap.add_argument("--ports", default="8001,8002")
    ap.add_argument("--refs", type=Path, default=Path("/root/minrg/train_refs.jsonl"))
    ap.add_argument("--prefixes", type=Path,
                    default=Path("/root/minrg/train_prefixes.jsonl.gz"))
    ap.add_argument("--out", type=Path, default=Path("/root/minrg/parrot_probe.jsonl"))
    ap.add_argument("--n-turns", type=int, default=105)
    ap.add_argument("--conc", type=int, default=12)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]

    prefixes = {}
    with gzip.open(args.prefixes, "rt") as f:
        for line in f:
            r = json.loads(line)
            prefixes[r["turn_id"]] = r["prefix"]
    rows = [json.loads(l) for l in open(args.refs)]
    rows = [r for r in rows if r.get("mu") is not None
            and len(r.get("refs") or []) >= 2 and r["turn_id"] in prefixes
            and r["refs"][0].get("z", "").strip()]
    rng = random.Random(424242)  # same seed as suffix attack -> same turns
    picked = rng.sample(rows, min(args.n_turns, len(rows)))
    print(f"{len(picked)} turns, ports={ports}", file=sys.stderr, flush=True)

    kinds = ["honest", "parrot", "ref_x3", "generic"]
    sems = {p: asyncio.Semaphore(args.conc) for p in ports}
    lock = asyncio.Lock()
    out_f = open(args.out, "w")
    agg = {k: {j: [] for j in ("lme", "r_c", "g", "min_rg", "m")} for k in kinds}
    n = 0
    async with httpx.AsyncClient() as client:
        async def do(row):
            nonlocal n
            port = ports[zlib.adler32(row["turn_id"].encode()) % len(ports)]
            gen = gen_prompt(prefixes[row["turn_id"]])
            vs = variants(row["refs"][0]["z"], prefixes[row["turn_id"]])
            async with sems[port]:
                for k, z in vs.items():
                    sc = await score(client, port, gen, row, z)
                    async with lock:
                        out_f.write(json.dumps(
                            {"turn_id": row["turn_id"], "kind": k, **sc}) + "\n")
                        for j in agg[k]:
                            if sc[j] is not None:
                                agg[k][j].append(sc[j])
            n += 1
            if n % 20 == 0:
                out_f.flush()
                print(f"{n}/{len(picked)}", file=sys.stderr, flush=True)
        await asyncio.gather(*[do(r) for r in picked])
    out_f.close()

    print("\nkind      n   lme      r_c      G        min_rg   m")
    for k in kinds:
        a = agg[k]
        mn = {j: (st.mean(a[j]) if a[j] else float("nan")) for j in a}
        print(f"{k:9s} {len(a['min_rg']):3d} {mn['lme']:+.4f}  {mn['r_c']:+.4f}  "
              f"{mn['g']:+.4f}  {mn['min_rg']:+.4f}  {mn['m']:+.4f}")

    by = {}
    for line in open(args.out):
        r = json.loads(line)
        by.setdefault(r["turn_id"], {})[r["kind"]] = r
    for k in kinds:
        if k == "honest":
            continue
        w = t = 0
        for d in by.values():
            if ("honest" in d and k in d and d["honest"]["min_rg"] is not None
                    and d[k]["min_rg"] is not None):
                t += 1
                w += d["honest"]["min_rg"] > d[k]["min_rg"]
        if t:
            print(f"honest beats {k}: {w}/{t} ({100*w/t:.0f}%)")
    print("PARROT_PROBE_DONE")


if __name__ == "__main__":
    asyncio.run(main())
