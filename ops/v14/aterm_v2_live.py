#!/usr/bin/env python
"""A-term v2 live read on stored verdicts (offline, box CPU).

v1 (docs/a-term-proposal.md): S_w = min(R,G) + w·(A_match − pair) on valid
turns where defined. v2 (coordinator, 2026-09-16 12:50 UTC): the same term
EXCEPT (i) terminus_json turns earn no A credit, (ii) an action whose
normalised form is "generic" — appears in more than f of ALL reference
actions of its dialect in the corpus-wide reference pool (every teacher
reference of every stored duel) — earns no A credit. This script builds the
pool, publishes the generic list for several f, and recomputes the live
read (separation on wvk-18 duels, generic credit share, decision flips,
max |Δz|) under v1 and v2.

    python ops/v14/aterm_v2_live.py --evals affine/state/evals --first chal-00470 --out ops/v14/out
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))
from affine import score as S  # noqa: E402
from evalsrv import amatch  # noqa: E402

WEIGHTS = (0.01, 0.02)
FORFEIT = -0.1
GENERIC_RE = re.compile(r"^(ls( -[a-z]+)?( \S+)?|pwd|cat \S+|git (status|diff|log)( [^|;&]*)?|echo .*|head [^|;&]*|tail [^|;&]*)$")
F_GRID = (0.0005, 0.001, 0.002, 0.005)


def kind_of(y: str) -> str:
    y = (y or "").lstrip()
    if y.startswith("```"):
        return "bash"
    if "<tool_call>" in y or y.startswith("[{") or (y.startswith("{") and '"name"' in y[:200]):
        return "tool_call"
    if "\\boxed" in y:
        return "boxed"
    if y.startswith("{") and '"commands"' in y:
        return "terminus_json"
    return "text"


def load(evals: Path, first: str, last: str) -> list[tuple[str, dict]]:
    out = []
    for f in sorted(glob.glob(str(evals / "chal-*.json.gz"))):
        cid = Path(f).name.split(".")[0]
        if "-confirm" in cid or cid < first or cid > last:
            continue
        d = json.load(gzip.open(f))
        if not d.get("king_rows") or not d.get("challenger_rows"):
            continue
        out.append((cid, d))
    return out


def ref_norms(refs: list[dict]) -> tuple[str, list[str]]:
    """(kind, normalised ref actions) for one turn's references."""
    kinds = collections.Counter(kind_of(r.get("y", "")) for r in refs)
    kind = kinds.most_common(1)[0][0] if kinds else "text"
    norms = [n for n in (amatch.norm_action(r.get("y", ""), kind) for r in refs) if n]
    return kind, norms


def build_pool(duels) -> dict[str, collections.Counter]:
    pool: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _, d in duels:
        for tid, refs in (d.get("teacher_refs") or {}).items():
            kind, norms = ref_norms(refs)
            for n in norms:
                pool[kind][n] += 1
    return pool


FILTER_KINDS = ("bash", "tool_call")


def generic_sets(pool, f: float) -> dict[str, set[str]]:
    """Exact normalised actions whose share of the dialect's reference pool
    exceeds f — bash / tool_call only (boxed answers and text are not
    'generic commands'; terminus is excluded from the term outright)."""
    out = {}
    for kind, c in pool.items():
        if kind not in FILTER_KINDS:
            continue
        total = sum(c.values())
        out[kind] = {a for a, n in c.items() if n / total > f}
    return out


def head_of(ynorm: str | None, kind: str) -> str | None:
    """Command head for the v2b (head-frequency) filter: the first word of a
    bash command, two words for git/docker/npm/pip/python -m; for a bash-like
    tool call the same on its command argument; other tool calls: the tool
    name. None where there is no head."""
    if not ynorm:
        return None
    cmd = _cmd_of(ynorm, kind) if kind == "tool_call" else (ynorm if kind == "bash" else None)
    if cmd is None:
        if kind == "tool_call":
            try:
                calls = json.loads(ynorm)
                return "tool:" + str(calls[0][0]) if calls else None
            except Exception:
                return None
        return None
    toks = cmd.strip().split()
    if not toks:
        return None
    if toks[0] in ("cd",) and "&&" in toks:
        toks = toks[toks.index("&&") + 1:]
        if not toks:
            return None
    head = toks[0]
    if head in ("git", "docker", "npm", "pip", "pip3", "cargo", "go", "make") and len(toks) > 1:
        head += " " + toks[1]
    elif head in ("python", "python3") and len(toks) > 2 and toks[1] == "-m":
        head += " -m " + toks[2]
    return head


def build_head_pool(duels) -> dict[str, collections.Counter]:
    pool: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _, d in duels:
        for tid, refs in (d.get("teacher_refs") or {}).items():
            kind, norms = ref_norms(refs)
            if kind not in FILTER_KINDS:
                continue
            for n in norms:
                h = head_of(n, kind)
                if h:
                    pool[kind][h] += 1
    return pool


def head_sets(hpool, f: float) -> dict[str, set[str]]:
    return {k: {h for h, n in c.items() if n / sum(c.values()) > f} for k, c in hpool.items()}


def side_rows(rows, refs_by_tid, dp):
    tau, bc, bf = dp.get("tau", 0.03), dp.get("band_c", 2.0), dp.get("band_floor", 0.002)
    out = {}
    for r in rows:
        tid = r["turn_id"]
        if S.is_forfeit(r):
            out[tid] = {"forfeit": True, "S0": FORFEIT}
            continue
        pairs = r["pairs"]
        S0 = S.turn_min_rg(pairs, tau, bc, bf)
        y = pairs[0].get("y_a") or ""
        refs = refs_by_tid.get(tid) or []
        kind, rnorms = ref_norms(refs) if refs else (kind_of(y), [])
        ynorm = amatch.norm_action(y, kind)
        A, pair = r.get("a_match"), r.get("ref_pair")
        if A is None and rnorms and ynorm:
            A = sum(1 for a in rnorms if a == ynorm) / len(rnorms)
        if pair is None and rnorms:
            pair = amatch.pairwise(rnorms)
        out[tid] = {"forfeit": False, "S0": S0, "A": A, "pair": pair, "kind": kind,
                    "ynorm": ynorm, "generic_re": bool(kind in ("bash", "tool_call") and ynorm
                                                        and GENERIC_RE.match(_cmd_of(ynorm, kind) or ""))}
    return out


def _cmd_of(ynorm: str, kind: str) -> str | None:
    if kind == "bash":
        return ynorm
    if kind == "tool_call":
        try:
            calls = json.loads(ynorm)
            if len(calls) == 1 and calls[0][0] in ("bash", "execute_bash", "run_command", "shell"):
                args = calls[0][1]
                return args.get("command") or args.get("cmd") or next(iter(args.values()), None)
        except Exception:
            return None
    return None


def term(t: dict, w: float, v2, generic: dict[str, set[str]], heads: dict[str, set[str]] | None = None) -> float:
    """v2 = False: v1; True: exact-list filter + terminus excluded; "b": also
    the head-frequency filter."""
    if t["forfeit"] or t.get("A") is None or t.get("pair") is None:
        return 0.0
    if v2:
        if t["kind"] == "terminus_json":
            return 0.0
        if t.get("ynorm") and t["ynorm"] in generic.get(t["kind"], set()):
            return 0.0
        if v2 == "b" and heads is not None and t["kind"] in FILTER_KINDS:
            h = head_of(t.get("ynorm"), t["kind"])
            if h and h in heads.get(t["kind"], set()):
                return 0.0
    return w * (t["A"] - t["pair"])


def decide(diffs, dp):
    n = len(diffs)
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n) if n > 1 else float("inf")
    z = m / se if se > 0 else 0.0
    wins = m > max(float(dp.get("k_sigma", 2.0)) * se, float(dp.get("min_margin", 0.002)))
    return m, se, z, wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, default=REPO / "affine/state/evals")
    ap.add_argument("--first", default="chal-00470")
    ap.add_argument("--last", default="chal-99999")
    ap.add_argument("--f", type=float, default=0.001, help="generic threshold: share of the dialect's reference pool")
    ap.add_argument("--f-head", type=float, default=0.01, help="v2b: command-head share threshold")
    ap.add_argument("--out", type=Path, default=REPO / "ops/v14/out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    duels = load(a.evals, a.first, a.last)
    print(f"duels: {len(duels)} ({duels[0][0]} … {duels[-1][0]})")
    pool = build_pool(duels)
    lines = ["# A-term v2 live read", "",
             f"Duels: {len(duels)} ({duels[0][0]} … {duels[-1][0]}). Reference pool = every teacher reference action of these duels, normalised per dialect (evalsrv/amatch.py).", ""]
    lines += ["## Reference pool and generic lists", "", "| dialect | refs | distinct normalised actions | " + " | ".join(f"generic @ f={f:g} (n actions / share of refs)" for f in F_GRID) + " |", "|---|---|---|" + "---|" * len(F_GRID)]
    for kind, c in sorted(pool.items()):
        total = sum(c.values())
        cells = []
        for f in F_GRID:
            g = {x: n for x, n in c.items() if n / total > f}
            cells.append(f"{len(g)} / {sum(g.values()) / total:.1%}")
        lines.append(f"| {kind} | {total} | {len(c)} | " + " | ".join(cells) + " |")
    generic = generic_sets(pool, a.f)
    hpool = build_head_pool(duels)
    heads = head_sets(hpool, a.f_head)
    lines += ["", f"### Command-head pool (v2b filter, f_head = {a.f_head:g})", "", "| dialect | heads (refs) | distinct heads | generic heads | share of refs |", "|---|---|---|---|---|"]
    for kind, c in sorted(hpool.items()):
        tot = sum(c.values()); g = heads.get(kind, set())
        lines.append(f"| {kind} | {tot} | {len(c)} | {len(g)} | {sum(c[h] for h in g) / tot:.1%} |")
        lines.append(f"  - {kind} generic heads: " + ", ".join(f"`{h}` ({c[h] / tot:.2%})" for h in sorted(g, key=lambda x: -c[x])))
    (a.out / f"generic_heads_f{a.f_head:g}.json").write_text(json.dumps(
        {"f_head": a.f_head, "rule": "command head share of the dialect's reference pool > f_head -> no A credit",
         "heads": {k: [{"head": h, "n": hpool[k][h], "share": hpool[k][h] / sum(hpool[k].values())} for h in sorted(v, key=lambda x: -hpool[k][x])] for k, v in heads.items()}}, indent=1))
    pub = {kind: sorted(((x, c[x], c[x] / sum(c.values())) for x in g), key=lambda t: -t[1]) for kind, g in generic.items() for c in [pool[kind]]}
    (a.out / f"generic_actions_f{a.f:g}.json").write_text(json.dumps(
        {"f": a.f, "rule": "normalised action share of the dialect's reference pool > f -> no A credit",
         "pool_refs": {k: sum(c.values()) for k, c in pool.items()},
         "generic": {k: [{"action": x, "n": n, "share": s} for x, n, s in v] for k, v in pub.items()}}, indent=1, ensure_ascii=False))
    lines += ["", f"Staged list: f = {a.f:g} → `generic_actions_f{a.f:g}.json`. Top entries:", ""]
    for kind, v in sorted(pub.items()):
        lines.append(f"- **{kind}** ({len(v)}): " + ", ".join(f"`{x[:40]}` ({s:.2%})" for x, n, s in v[:12]))

    # per-duel recompute
    per = []
    generic_credit = {"v1": collections.Counter(), "v2": collections.Counter(), "v2b": collections.Counter()}
    pos_credit = {"v1": collections.Counter(), "v2": collections.Counter(), "v2b": collections.Counter()}
    term_credit = {"v1": collections.Counter(), "v2": collections.Counter(), "v2b": collections.Counter()}
    for cid, d in duels:
        v = d["verdict"]
        dp = v.get("duel_params") or {}
        refs = d.get("teacher_refs") or {}
        wvk = "18+" if "a_match" in d["king_rows"][0] else ("17" if float(dp.get("band_c", 2.0)) >= 4 else "16")
        k = side_rows(d["king_rows"], refs, dp)
        c = side_rows(d["challenger_rows"], refs, dp)
        common = sorted(set(k) & set(c))
        res = {}
        for label, v2 in (("S0", None), ("v1", False), ("v2", True), ("v2b", "b")):
            for w in ([0.0] if v2 is None else WEIGHTS):
                diffs = []
                for tid in common:
                    tc = c[tid]["S0"] + (term(c[tid], w, v2, generic, heads) if v2 is not None else 0.0)
                    tk = k[tid]["S0"] + (term(k[tid], w, v2, generic, heads) if v2 is not None else 0.0)
                    diffs.append(tc - tk)
                res[(label, w)] = decide(diffs, dp)
        # credit accounting (w=0.01), both sides
        for label, v2 in (("v1", False), ("v2", True), ("v2b", "b")):
            for side in (k, c):
                for t in side.values():
                    if t["forfeit"]:
                        continue
                    x = term(t, 0.01, v2, generic, heads)
                    kind = t.get("kind")
                    term_credit[label][kind] += 1
                    if x > 0:
                        pos_credit[label][kind] += x
                        if t.get("generic_re"):
                            generic_credit[label][kind] += x
        stored = bool(v.get("challenger_wins") or v.get("duel_rule_wins"))
        per.append((cid, wvk, len(common), stored, res))
    lines += ["", "## Re-scoring under v1 / v2 (w = 0.01 and 0.02)", "",
              "| duel | wvk | n | stored | z S0 | z v1 .01 | z v2 .01 | z v2b .01 | z v2b .02 | margin S0 → v2b .01 | decision S0 / v1 .01 / v2 .01 / v2b .01 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    flips = collections.Counter()
    dz = collections.defaultdict(list)
    for cid, wvk, n, stored, res in per:
        z0 = res[("S0", 0.0)][2]
        cells = [f"{res[(l, w)][2]:.2f}" for l, w in (("v1", 0.01), ("v2", 0.01), ("v2b", 0.01), ("v2b", 0.02))]
        dec = "win" if res[("S0", 0.0)][3] else "lose"
        decs = [("win" if res[(l, w)][3] else "lose") for l, w in (("v1", 0.01), ("v2", 0.01), ("v2b", 0.01))]
        for l, w in (("v1", 0.01), ("v2", 0.01), ("v2b", 0.01), ("v1", 0.02), ("v2", 0.02), ("v2b", 0.02)):
            if res[(l, w)][3] != res[("S0", 0.0)][3]:
                flips[(l, w)] += 1
            dz[(l, w)].append(res[(l, w)][2] - z0)
        lines.append(f"| {cid} | {wvk} | {n} | {'win' if stored else 'lose'} | {z0:.2f} | " + " | ".join(cells)
                     + f" | {res[('S0', 0.0)][0]:+.4f} → {res[('v2b', 0.01)][0]:+.4f} | {dec} / " + " / ".join(decs) + " |")
    lines += ["", "### Flips and Δz vs S0", ""]
    for l, w in (("v1", 0.01), ("v2", 0.01), ("v2b", 0.01), ("v1", 0.02), ("v2", 0.02), ("v2b", 0.02)):
        d_ = dz[(l, w)]
        lines.append(f"- {l} w={w:g}: decision flips {flips[(l, w)]} / {len(per)}; mean Δz {st.mean(d_):+.3f}; max |Δz| {max(abs(x) for x in d_):.3f}")
    # separation on wvk-18 flat duels
    flat = [(cid, res) for cid, wvk, n, stored, res in per if wvk == "18+" and abs(res[("S0", 0.0)][2]) < 1.0]
    lines += ["", f"### Separation on wvk-18 flat duels (|z S0| < 1): {len(flat)}", "", "| duel | z S0 | z v1 .01 | z v2 .01 | z v2b .01 | Δ|z| v1 | Δ|z| v2 | Δ|z| v2b |", "|---|---|---|---|---|---|---|---|"]
    sep = {"v1": 0, "v2": 0, "v2b": 0}
    for cid, res in flat:
        z0, z1, z2, z3 = res[("S0", 0.0)][2], res[("v1", 0.01)][2], res[("v2", 0.01)][2], res[("v2b", 0.01)][2]
        sep["v1"] += int(abs(z1) - abs(z0) > 0.25)
        sep["v2"] += int(abs(z2) - abs(z0) > 0.25)
        sep["v2b"] += int(abs(z3) - abs(z0) > 0.25)
        lines.append(f"| {cid} | {z0:.2f} | {z1:.2f} | {z2:.2f} | {z3:.2f} | {abs(z1) - abs(z0):+.2f} | {abs(z2) - abs(z0):+.2f} | {abs(z3) - abs(z0):+.2f} |")
    lines.append(f"\nseparates (Δ|z| > 0.25): v1 {sep['v1']} / {len(flat)}, v2 {sep['v2']} / {len(flat)}, v2b {sep['v2b']} / {len(flat)}")
    lines += ["", "## Generic credit share (w = 0.01; generic = E5 regex on bash / bash-tool actions)", "",
              "| dialect | valid turns (both sides) | v1: positive credit to generic / all positive | v2 (exact list + no terminus) | v2b (+ head list) |", "|---|---|---|---|---|"]
    for kind in sorted(set(term_credit["v1"]) | set(term_credit["v2"])):
        cells = []
        for lab in ("v1", "v2", "v2b"):
            p, g = pos_credit[lab][kind], generic_credit[lab][kind]
            cells.append(f"{g:.3f} / {p:.3f} = {g / p if p else 0:.1%}")
        lines.append(f"| {kind} | {term_credit['v1'][kind]} | " + " | ".join(cells) + " |")
    (a.out / "aterm_v2_live.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
