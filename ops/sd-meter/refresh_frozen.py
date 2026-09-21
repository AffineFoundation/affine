#!/usr/bin/env python3
"""Refresh the [duel.sd_meter.frozen.<dialect>] constants from published verdicts.

Every verdict that carries ``shadow.sd_meter`` (shadow or rule) stamps the
duel's leave-one-out anchors: ``mu_mean_by_dialect`` (mean per-turn μ of R,
A and Mc) and ``sigma_by_dialect`` (pooled within-turn sd), with
``n_loo_turns_by_dialect``. This pools the last N such verdicts, turn-
weighted, and prints the toml block (or rewrites it in place with --apply).

  python ops/sd-meter/refresh_frozen.py --last 20            # print
  python ops/sd-meter/refresh_frozen.py --last 20 --apply    # rewrite affine/affine.toml + website mirror
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HIST = REPO / "affine" / "state" / "history.jsonl"
TOML = REPO / "affine" / "affine.toml"
MIRROR = REPO / "affine" / "website" / "code" / "affine.toml"
LEGS = ("R", "A", "Mc")


def pooled(verdicts: list[dict]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list]] = {}
    for v in verdicts:
        sd = (v.get("shadow") or {}).get("sd_meter") or {}
        n_by = sd.get("n_loo_turns_by_dialect") or {}
        for kind, n in n_by.items():
            mu = (sd.get("mu_mean_by_dialect") or {}).get(kind) or {}
            sig = (sd.get("sigma_by_dialect") or {}).get(kind) or {}
            a = acc.setdefault(kind, {leg: [] for leg in LEGS})
            for leg in LEGS:
                if mu.get(leg) is not None and sig.get(leg) is not None:
                    a[leg].append((n, mu[leg], sig[leg]))
    out = {}
    for kind, legs in acc.items():
        row = {}
        for leg, xs in legs.items():
            n = sum(x[0] for x in xs)
            if not n:
                continue
            row[f"{leg}_mu"] = sum(x[0] * x[1] for x in xs) / n
            row[f"{leg}_sigma"] = math.sqrt(sum(x[0] * x[2] ** 2 for x in xs) / n)
            row[f"{leg}_n"] = n
        if row:
            out[kind] = row
    return out


def render(table: dict[str, dict[str, float]], n_verdicts: int, span: str) -> str:
    lines = [f"# Frozen per-dialect anchors refreshed from {n_verdicts} published verdicts ({span}),",
             "# turn-weighted pooled LOO μ/σ (ops/sd-meter/refresh_frozen.py)."]
    for kind in sorted(table):
        lines.append(f"[duel.sd_meter.frozen.{kind}]")
        for leg in LEGS:
            if f"{leg}_mu" in table[kind]:
                lines.append(f"{leg}_mu = {table[kind][f'{leg}_mu']:.6g}")
                lines.append(f"{leg}_sigma = {table[kind][f'{leg}_sigma']:.6g}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in HIST.open() if l.strip()]
    vs = [r["verdict"] | {"_at": r.get("at")} for r in rows
          if r.get("event") == "verdict" and isinstance(r.get("verdict"), dict)
          and (r["verdict"].get("shadow") or {}).get("sd_meter")]
    vs = vs[-a.last:]
    if not vs:
        print("no verdicts with shadow.sd_meter yet")
        return 1
    table = pooled(vs)
    span = f"{vs[0]['_at'][:16]} .. {vs[-1]['_at'][:16]}"
    block = render(table, len(vs), span)
    print(block)
    if not a.apply:
        return 0
    for path in (TOML, MIRROR):
        s = path.read_text()
        # Replace everything from the first frozen table to the next top-level table.
        m = re.search(r"(?ms)^# Frozen per-dialect anchors.*?(?=^\[(?!duel\.sd_meter\.frozen)[^\]]+\]$)", s)
        if not m:
            raise SystemExit(f"{path}: frozen block not found")
        s = s[:m.start()] + block + "\n" + s[m.end():]
        path.write_text(s)
        print("rewrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
