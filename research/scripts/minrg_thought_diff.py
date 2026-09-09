"""Compare what the two GRPO objectives wrote (step-150 held-out evals).

Loads base / min(R,G)-trained / v4-trained eval JSONs (each has per-turn z,
lme, r_c, g, m). Reports distribution stats and, since the three evals sample
the SAME held-out turns, a per-turn paired comparison of the trained arms vs
base: did min(R,G) training raise G? did v4 training drift toward lower G
(more filler-like) while chasing Reason?
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "minrg_round2"
FILES = {"base": "eval_base.json",
         "minrg150": "eval_minrg150.json",
         "v4_150": "eval_v4_150.json"}


def load(fn: str) -> dict:
    d = json.loads((RES / fn).read_text())
    turns = {t["turn_id"]: t for t in d["turns"]}
    return {"summary": d["summary"], "turns": turns}


def fmt(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return (f"mean={st.mean(xs):+.4f} med={st.median(xs):+.4f} "
            f"min={min(xs):+.4f} max={max(xs):+.4f}")


def main() -> None:
    data = {k: load(v) for k, v in FILES.items()}
    print("=== per-arm distributions (step-150 held-out) ===")
    for k, d in data.items():
        ts = list(d["turns"].values())
        zb = [t.get("z_bytes") for t in ts]
        g = [t.get("g") for t in ts]
        m = [t.get("m") for t in ts]
        lme = [t.get("lme") for t in ts]
        rc = [t.get("r_c") for t in ts]
        mr = [t.get("min_rg") for t in ts]
        print(f"\n[{k}] n={len(ts)}")
        print(f"  z_bytes  {fmt(zb)}")
        print(f"  lme      {fmt(lme)}")
        print(f"  r_c      {fmt(rc)}")
        print(f"  G        {fmt(g)}")
        print(f"  m        {fmt(m)}")
        print(f"  min_rg   {fmt(mr)}")

    # paired vs base on shared turns
    base = data["base"]["turns"]
    for arm in ("minrg150", "v4_150"):
        t = data[arm]["turns"]
        shared = [tid for tid in t if tid in base
                  and t[tid].get("g") is not None
                  and base[tid].get("g") is not None]
        dG = [t[tid]["g"] - base[tid]["g"] for tid in shared]
        dLme = [t[tid]["lme"] - base[tid]["lme"] for tid in shared]
        dM = [t[tid]["m"] - base[tid]["m"] for tid in shared]
        up_g = sum(1 for x in dG if x > 0)
        print(f"\n=== {arm} vs base (paired, n={len(shared)}) ===")
        print(f"  dG:   mean={st.mean(dG):+.4f}  raised_G={up_g}/{len(shared)}"
              f" ({100*up_g/len(shared):.0f}%)")
        print(f"  dLme: mean={st.mean(dLme):+.4f}")
        print(f"  dM:   mean={st.mean(dM):+.4f}")


if __name__ == "__main__":
    main()
