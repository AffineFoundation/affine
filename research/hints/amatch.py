#!/usr/bin/env python
"""Offline probe of a DISCRETE action-agreement term on the E5 echoes.

  A_match(miner)  = fraction of the k valid reference actions equal to the
                    miner's action after dialect-aware normalization
  pair            = mean pairwise equality among the refs (how often the
                    teacher agrees with itself; the "free credit" a
                    deterministic-teacher turn would hand out)
Candidate turn scores (per miner, refs = the condition's valid refs):
  S0   = min(R, G_0)                          the live rule
  S_w  = S0 + w·(A_match − pair)               w ∈ {0.005, 0.01, 0.02}
  Smin = min(R, G_0, c·A_match)                c = 0.02 (G scale; a miner
                                               matching no ref scores ≤ 0)
Readouts: paired z (unhinted teacher held-out − live king; stored
challenger − king; coached held-out − king) per group / dialect under every
rule; mode-guessing exposure (A_match of synthetic miners: generic `ls -la`,
repeat-last prefix action, shrink = first line of repeat-last; plus `pair`
itself as the ceiling for a mode guesser); length bias (Spearman of action
length with A_match, R, S_w − S0).

  python amatch.py --run-dir RUN --turns turns.jsonl --out RUN/analysis
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))
sys.path.insert(0, str(HERE))

from affine import dialects  # noqa: E402

import analyze as A  # noqa: E402
from e4_analyze import spearman  # noqa: E402

WS = re.compile(r"\s+")
FENCE = re.compile(r"```(?:bash|mswea_bash_command)[ \t]*\n(.*?)\n```", re.S)
XML_CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
WEIGHTS = (0.005, 0.01, 0.02)
C_MIN = 0.02
FORFEIT = -0.1
GROUPS = ["all", "king_loop_onset", "king_pivot", "king_fail", "king_recoverable", "king_done",
          "completion_pre", "completion", "coached_decisive"]


def norm_ws(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def norm_bash(y: str) -> str | None:
    m = FENCE.search(y or "")
    cmd = m.group(1) if m else (y or "")
    cmd = norm_ws(cmd).replace('"', "'").rstrip(";").strip()
    return cmd or None


def norm_tool_call(y: str) -> str | None:
    calls = []
    for m in XML_CALL.finditer(y or ""):
        name = m.group(1).strip()
        args = {k.strip(): norm_ws(v) for k, v in XML_PARAM.findall(m.group(2))}
        calls.append((name, args))
    if not calls:
        # JSON tool calls: {"name": ..., "arguments": {...}} or a list of them
        body = re.sub(r"</?tool_call>", "", y or "").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None
        for c in obj if isinstance(obj, list) else [obj]:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or (c.get("function") or {}).get("name")
            args = c.get("arguments") or (c.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": norm_ws(args)}
            calls.append((name, {str(k): norm_ws(str(v)) for k, v in (args or {}).items()}))
    if not calls:
        return None
    return json.dumps(calls, sort_keys=True, ensure_ascii=False)


def norm_boxed(y: str) -> str | None:
    m = BOXED.search(y or "")
    if not m:
        return None
    s = m.group(1)
    s = re.sub(r"\\(left|right|,|;|!|text|mathrm|displaystyle)", "", s)
    s = s.replace("$", "").replace(" ", "").rstrip(".").lower()
    return s or None


def norm_terminus(y: str) -> str | None:
    try:
        obj = json.loads(y)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", y or "", re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    cmds = obj.get("commands") if isinstance(obj, dict) else None
    if cmds is None:
        return None
    keys = []
    for c in cmds:
        if isinstance(c, dict):
            keys.append(norm_ws(str(c.get("keystrokes") or c.get("command") or "")))
        else:
            keys.append(norm_ws(str(c)))
    return json.dumps({"commands": keys, "done": bool(obj.get("task_complete"))}, ensure_ascii=False)


NORM = {"bash": norm_bash, "tool_call": norm_tool_call, "boxed": norm_boxed,
        "terminus_json": norm_terminus}


def norm_action(y: str, kind: str) -> str | None:
    fn = NORM.get(kind)
    return fn(y) if fn else None


def pairwise(acts: list[str]) -> float | None:
    if len(acts) < 2:
        return None
    n = len(acts)
    eq = sum(1 for i in range(n) for j in range(i + 1, n) if acts[i] == acts[j])
    return eq / (n * (n - 1) / 2)


def synthetic_actions(turn: dict, kind: str) -> dict[str, str | None]:
    """Generic / repeat-last / shrink miners, in the turn's dialect."""
    out: dict[str, str | None] = {}
    if kind == "bash":
        out["generic"] = "ls -la"
    elif kind == "tool_call":
        out["generic"] = json.dumps([("bash", {"command": "ls -la"})], sort_keys=True)
    elif kind == "terminus_json":
        out["generic"] = json.dumps({"commands": ["ls -la"], "done": False})
    else:
        out["generic"] = None
    last = None
    for m in reversed(turn.get("prefix") or []):
        if m["role"] == "assistant":
            try:
                y = dialects.last_action(m["content"], kind)
            except Exception:  # noqa: BLE001
                y = ""
            if y:
                last = norm_action(y, kind)
            break
    out["repeat_last"] = last
    if last and kind == "bash":
        out["shrink"] = last.split(" && ")[0].split(" | ")[0].strip() or None
    elif last and kind == "terminus_json":
        try:
            o = json.loads(last)
            out["shrink"] = json.dumps({"commands": o["commands"][:1], "done": o["done"]})
        except (json.JSONDecodeError, KeyError):
            out["shrink"] = None
    else:
        out["shrink"] = None
    return out


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+")
    ap.add_argument("--turns", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="H0,fact_4096,fact_1792,mix3,coached_1792")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    turns = {}
    for p in args.turns:
        for line in open(p):
            t = json.loads(line)
            turns[t["turn_id"]] = {"prefix": t["prefix"], "group": t["group"]}
    rows = []
    for rd in args.run_dir:
        for line in open(Path(rd) / "results.jsonl"):
            r = json.loads(line)
            if not r.get("failed") and "H0" in r.get("conditions", {}):
                rows.append(r)
    conds = args.conds.split(",")

    # per (turn, cond, miner): scores under every rule
    per: list[dict] = []
    coverage = collections.Counter()
    synth = collections.defaultdict(list)     # (kind, arm) -> A_match values
    pair_by_kind = collections.defaultdict(list)
    for r in rows:
        kind = r["action_kind"]
        tprefix = turns.get(r["turn_id"], {})
        syn = synthetic_actions(tprefix, kind) if tprefix else {}
        for cond in conds:
            if cond not in r["conditions"]:
                continue
            tc = A.turn_condition(r, cond)
            if tc is None:
                continue
            refs = [norm_action(x["y"], kind) for x in r["conditions"][cond]["refs"] if x.get("valid")]
            refs_ok = [a for a in refs if a]
            coverage[(kind, "ref_valid")] += len(refs)
            coverage[(kind, "ref_normalized")] += len(refs_ok)
            pair = pairwise(refs_ok)
            if cond == "H0":
                if pair is not None:
                    pair_by_kind[kind].append(pair)
                if refs_ok:
                    for arm, a in syn.items():
                        if a is not None:
                            synth[(kind, arm)].append(sum(1 for x in refs_ok if x == a) / len(refs_ok))
            for mname, m in tc["miners"].items():
                rec = {"turn_id": r["turn_id"], "group": r["group"], "kind": kind, "cond": cond, "miner": mname}
                if m["forfeit"]:
                    rec.update(forfeit=True, S0=FORFEIT, Smin=FORFEIT, **{f"S_{w}": FORFEIT for w in WEIGHTS},
                               A=None, pair=pair, R=None, len_y=None)
                    per.append(rec)
                    continue
                raw = r["miners"][mname]
                ya = norm_action(raw.get("y") or "", kind)
                S0 = m.get("score")
                if S0 is None or not refs_ok or ya is None or pair is None:
                    rec.update(forfeit=False, S0=S0, Smin=None, A=None, pair=pair, R=m.get("R"),
                               len_y=len(raw.get("y") or ""), **{f"S_{w}": None for w in WEIGHTS})
                    per.append(rec)
                    continue
                a_match = sum(1 for x in refs_ok if x == ya) / len(refs_ok)
                rec.update(forfeit=False, S0=S0, A=a_match, pair=pair, R=m.get("R"), G0=m.get("G0"),
                           len_y=len(raw.get("y") or ""),
                           Smin=min(m["R"], m["G0"], C_MIN * a_match),
                           **{f"S_{w}": S0 + w * (a_match - pair) for w in WEIGHTS})
                per.append(rec)
    with open(out / "amatch_turns.jsonl", "w") as f:
        for p in per:
            f.write(json.dumps(p) + "\n")

    by = collections.defaultdict(dict)
    for p in per:
        by[(p["turn_id"], p["cond"])][p["miner"]] = p
    rules = ["S0"] + [f"S_{w}" for w in WEIGHTS] + ["Smin"]

    def paired_z(cond: str, a: str, b: str, group: str | None = None, kind: str | None = None):
        outz = {}
        for rule in rules:
            d = []
            for (tid, c), ms in by.items():
                if c != cond or a not in ms or b not in ms:
                    continue
                if group and group != "all" and ms[a]["group"] != group:
                    continue
                if kind and ms[a]["kind"] != kind:
                    continue
                sa, sb = ms[a].get(rule), ms[b].get(rule)
                if sa is None or sb is None:
                    continue
                d.append(sa - sb)
            m_, se, z, n = A.zstat(d)
            outz[rule] = (m_, z, n)
        return outz

    lines = [f"# Discrete action-agreement term A_match — offline on the E5 echoes ({len(rows)} turns)\n",
             "Rules: S0 = min(R, G_0) (live); S_w = S0 + w·(A_match − pair); Smin = min(R, G_0, 0.02·A_match). "
             "Paired z over turns; a miner with no valid action scores −0.1 under every rule.\n"]
    # coverage
    lines.append("## Normalization coverage (refs)\n")
    lines.append("| dialect | valid refs | normalized | share | mean ref pairwise agreement (H0) |")
    lines.append("|---|---|---|---|---|")
    for kind in sorted({k for k, _ in coverage}):
        v, nn = coverage[(kind, "ref_valid")], coverage[(kind, "ref_normalized")]
        pk = pair_by_kind.get(kind, [])
        lines.append(f"| {kind} | {v} | {nn} | {fmt(nn / v if v else float('nan'))} | {fmt(st.mean(pk)) if pk else '–'} |")
    lines.append("\n`text` turns have no action normalization (the whole visible reply is the action): A_match is undefined there and those turns keep S0 under every rule.\n")

    # positive control per group / dialect
    for label, a, b, cond in (("Unhinted teacher held-out − live king (H0 refs)", "teacher_heldout", "king_live", "H0"),
                              ("Stored challenger − stored king (H0 refs)", "stored_chal", "stored_king", "H0"),
                              ("Coached held-out (fact_4096) − live king (fact_4096 refs)", "coached_fact_4096", "king_live", "fact_4096"),
                              ("Coached held-out (fact_1792) − live king (fact_1792 refs)", "coached_fact_1792", "king_live", "fact_1792"),
                              ("Coached held-out (coach's note) − live king (coached states)", "coached_coached_1792", "king_live", "coached_1792")):
        lines.append(f"\n## {label}\n")
        lines.append("| slice | n | " + " | ".join(f"{r} z (d)" for r in rules) + " |")
        lines.append("|---|---|" + "---|" * len(rules))
        for g in GROUPS:
            z = paired_z(cond, a, b, group=g)
            if z["S0"][2] < 5:
                continue
            lines.append(f"| {g} | {z['S0'][2]} | " + " | ".join(f"{fmt(z[r][1])} ({fmt(z[r][0], 4)})" for r in rules) + " |")
        for k in ("bash", "tool_call", "terminus_json", "text", "boxed"):
            z = paired_z(cond, a, b, kind=k)
            if z["S0"][2] < 5:
                continue
            lines.append(f"| dialect {k} | {z['S0'][2]} | " + " | ".join(f"{fmt(z[r][1])} ({fmt(z[r][0], 4)})" for r in rules) + " |")

    # A_match levels
    lines.append("\n## A_match by miner (H0 refs)\n")
    lines.append("| miner | n | mean A_match | share A_match = 1 | share A_match = 0 | mean pair |")
    lines.append("|---|---|---|---|---|---|")
    for mname in ("teacher_heldout", "king_live", "recorded", "stored_king", "stored_chal"):
        vals = [p for p in per if p["cond"] == "H0" and p["miner"] == mname and p.get("A") is not None]
        if not vals:
            continue
        lines.append(f"| {mname} | {len(vals)} | {fmt(st.mean(p['A'] for p in vals))} | {fmt(st.mean(1.0 if p['A'] == 1 else 0.0 for p in vals))} | "
                     f"{fmt(st.mean(1.0 if p['A'] == 0 else 0.0 for p in vals))} | {fmt(st.mean(p['pair'] for p in vals))} |")
    for cond in conds[1:]:
        vals = [p for p in per if p["cond"] == cond and p["miner"] == f"coached_{cond}" and p.get("A") is not None]
        if vals:
            lines.append(f"| coached_{cond} (refs {cond}) | {len(vals)} | {fmt(st.mean(p['A'] for p in vals))} | {fmt(st.mean(1.0 if p['A'] == 1 else 0.0 for p in vals))} | "
                         f"{fmt(st.mean(1.0 if p['A'] == 0 else 0.0 for p in vals))} | {fmt(st.mean(p['pair'] for p in vals))} |")

    # mode-guessing exposure
    lines.append("\n## Mode-guessing exposure — A_match of synthetic miners (H0 refs)\n")
    lines.append("`generic` = `ls -la` in the dialect; `repeat_last` = the last assistant action in the prefix; `shrink` = repeat_last cut to its first command. "
                 "`pair` = how often the refs agree with each other = the ceiling for a miner who guesses the teacher's mode.\n")
    lines.append("| dialect | arm | n turns | mean A_match | share A_match ≥ 2/3 | mean centred (A − pair) |")
    lines.append("|---|---|---|---|---|---|")
    for (kind, arm), vals in sorted(synth.items()):
        pk = pair_by_kind.get(kind, [])
        lines.append(f"| {kind} | {arm} | {len(vals)} | {fmt(st.mean(vals), 3)} | {fmt(st.mean(1.0 if v >= 2 / 3 else 0.0 for v in vals), 3)} | "
                     f"{fmt(st.mean(vals) - (st.mean(pk) if pk else 0), 3)} |")
    for kind, pk in sorted(pair_by_kind.items()):
        lines.append(f"| {kind} | (refs pairwise) | {len(pk)} | {fmt(st.mean(pk), 3)} | {fmt(st.mean(1.0 if v >= 2 / 3 else 0.0 for v in pk), 3)} | 0 |")

    # length bias
    lines.append("\n## Length bias (H0 refs) — Spearman with the miner's action length in chars\n")
    lines.append("| miner | n | ρ(len, A_match) | ρ(len, R) | ρ(len, S_0.01 − S0) | ρ(len, pair) |")
    lines.append("|---|---|---|---|---|---|")
    for mname in ("teacher_heldout", "king_live"):
        vals = [p for p in per if p["cond"] == "H0" and p["miner"] == mname and p.get("A") is not None and p.get("R") is not None]
        if len(vals) < 10:
            continue
        L = [p["len_y"] for p in vals]
        lines.append(f"| {mname} | {len(vals)} | {fmt(spearman(L, [p['A'] for p in vals]))} | {fmt(spearman(L, [p['R'] for p in vals]))} | "
                     f"{fmt(spearman(L, [p['S_0.01'] - p['S0'] for p in vals]))} | {fmt(spearman(L, [p['pair'] for p in vals]))} |")
    (out / "amatch_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
