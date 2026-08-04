"""Compare ungrounded vs prefix-grounded softΛ2; offline pad-attack check."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runner import load_turns, turn_id  # noqa: E402

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
ALPHA, TARGET = 0.05, 20
TRUTH = {
    "king-genesis": 58.2,
    "king-I": 38.4,
    "king-XCIX": 39.8,
    "king-VIII": 36.2,
    "king-XI": 33.6,
    "king-V": 32.0,
    "king-XLV": 26.0,
    "king-XLVI": 13.2,
    "king-LI": 11.6,
    "king-CI": 12.4,
}


def idents(s: str) -> set[str]:
    return set(IDENT_RE.findall(s or ""))


def prefix_text(prefix: list[dict]) -> str:
    parts = []
    for m in prefix:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif c is not None:
            parts.append(json.dumps(c))
    return " ".join(parts)


def soft(l2: float, z: str, pref_ids: set[str] | None, grounded: bool) -> float:
    zids = idents(z)
    ni = len(zids & pref_ids) if grounded and pref_ids is not None else len(zids)
    return l2 - ALPHA * max(0, TARGET - ni) / TARGET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ekings", default="/root/results/ekings_v2_all.jsonl")
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    args = ap.parse_args()

    pref: dict[str, set[str]] = {}
    for rec in load_turns(args.turns, 100_000):
        pref[turn_id(rec)] = idents(prefix_text(rec["prefix"]))
    print(f"prefix turns={len(pref)}")

    rows: dict[str, list] = defaultdict(list)
    for line in open(args.ekings):
        r = json.loads(line)
        if r.get("valid") and "pairs" in r and r["miner"] in TRUTH:
            rows[r["miner"]].append(r)

    for label, grounded in [("ungrounded", False), ("grounded", True)]:
        means = {}
        for m, rs in rows.items():
            vals = []
            for r in rs:
                pids = pref.get(r["turn_id"], set())
                for p in r["pairs"]:
                    l2 = p["lpC_yc_za"] - p["lpC_yc_e"]
                    vals.append(soft(l2, p.get("z_a", ""), pids, grounded))
            means[m] = st.mean(vals)
        miners = sorted(TRUTH, key=lambda m: -TRUTH[m])
        rho, p = stats.spearmanr(
            [means[m] for m in miners], [TRUTH[m] for m in miners]
        )
        print(
            f"{label}: Spearman={rho:+.3f} p={p:.4g} "
            f"genesis={means['king-genesis']:+.4f} I={means['king-I']:+.4f}"
        )
        order = sorted(means.items(), key=lambda x: -x[1])
        print(
            "  rank:",
            " > ".join(f"{m.removeprefix('king-')}:{v:+.4f}" for m, v in order),
        )

    pad_lex: set[str] = set()
    for r in rows["king-genesis"]:
        for p in r["pairs"]:
            pad_lex |= idents(p.get("z_a", ""))
    pad_lex_l = sorted(pad_lex)
    random.seed(0)

    def pad_z(z: str, pref_ids: set[str], target_n: int = 30,
              use_grounded_tokens: bool = False) -> str:
        cur = idents(z)
        if use_grounded_tokens:
            base = len(cur & pref_ids)
            pool = [t for t in pad_lex_l if t in pref_ids and t not in cur]
        else:
            base = len(cur)
            pool = [t for t in pad_lex_l if t not in cur and t not in pref_ids]
        need = max(0, target_n - base)
        extra = random.sample(pool, min(need, len(pool))) if pool else []
        return z + ("\n# " + " ".join(extra) if extra else "")

    by: dict[str, dict] = defaultdict(dict)
    for m in ("king-genesis", "king-I"):
        for r in rows[m]:
            by[r["turn_id"]][m] = r

    for mode, grounded_score, grounded_pad in [
        ("ungrounded_score+ungrounded_pad", False, False),
        ("grounded_score+ungrounded_pad", True, False),
        ("grounded_score+grounded_pad", True, True),
    ]:
        d = []
        for tid, sides in by.items():
            if len(sides) < 2:
                continue
            gp = sides["king-genesis"]["pairs"][0]
            ip = sides["king-I"]["pairs"][0]
            pids = pref.get(tid, set())
            gl2 = gp["lpC_yc_za"] - gp["lpC_yc_e"]
            il2 = ip["lpC_yc_za"] - ip["lpC_yc_e"]
            iz2 = pad_z(ip.get("z_a", ""), pids, use_grounded_tokens=grounded_pad)
            d.append(
                soft(il2, iz2, pids, grounded_score)
                - soft(gl2, gp.get("z_a", ""), pids, grounded_score)
            )
        print(
            f"{mode}: mean ΔI-G={st.mean(d):+.4f} "
            f"win_I={sum(1 for x in d if x > 0) / len(d):.0%} n={len(d)}"
        )


if __name__ == "__main__":
    main()
