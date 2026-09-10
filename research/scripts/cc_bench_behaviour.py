"""Behaviour tables for Claude Code benchmark traces (verifiers traces.jsonl):
king(s) vs teacher on the same SWE-bench Pro tasks. (2026-09-09)

Reads every `<root>/*/<label>*/traces.jsonl`, keeps the best record per task
(a scored run beats an errored one, a solve beats a miss), and prints:

  1. per outcome class (both / teacher-only / king-only / neither): steps,
     reasoning chars per step and per run, max identical-call repeats, test
     runs, edits, reasoning-only replies, length cuts, max prompt tokens
  2. per prompt-depth bin: repeat rate, re-Read rate, reasoning-only rate,
     median reasoning — the king's degradation with context depth
  3. reasoning-length distribution per step vs the duel's thought cap
     (1024 tokens ~ 3.7 chars/token ~ 3.8k chars)

    python research/scripts/cc_bench_behaviour.py --root /tmp/swepro/cc \
        --king king --teacher teacher
    python research/scripts/cc_bench_behaviour.py --root /tmp/r9/cc --king king9 \
        --teacher-root /tmp/swepro/cc --teacher teacher
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import Counter, defaultdict

import numpy as np

TEST_RE = re.compile(r"pytest|go test|npm test|yarn test|jest|vitest|make test|tox\b|cargo test|"
                     r"python -m unittest|ansible-test|npx (jest|vitest)")
DEPTH_BINS = ((0, 32_000), (32_000, 64_000), (64_000, 100_000), (100_000, 10**9))
CHARS_PER_TOKEN = 3.7   # measured on 3,000 teacher thoughts with the Qwen3.6 tokenizer


def content_text(c) -> str:
    if isinstance(c, list):
        return " ".join(str(x.get("text", "") or "") for x in c if isinstance(x, dict))
    return c or ""


def tool_calls(m: dict) -> list[tuple[str, str]]:
    out = []
    for tc in m.get("tool_calls") or []:
        if isinstance(tc, dict):
            f = tc.get("function") or tc
            a = f.get("arguments") or ""
            if not isinstance(a, str):
                a = json.dumps(a, sort_keys=True)
            out.append((f.get("name"), a[:600]))
    return out


def load(root: str, label: str) -> dict:
    res = {}
    for p in sorted(glob.glob(f"{root}/*/{label}*/traces.jsonl")):
        for line in open(p):
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            for t in ep.get("traces", []):
                name = t["task"]["data"]["name"].rsplit("/", 1)[-1]
                s = ((t.get("rewards") or {}).get("solved") or {}).get("score")
                nodes = [(n.get("message") or n) for n in t.get("nodes") or []]
                calls = {c.get("node"): c for c in (t.get("calls") or [])}
                steps = []
                seen: Counter = Counter()
                read_seen: Counter = Counter()
                for i, m in enumerate(nodes):
                    if m.get("role") != "assistant":
                        continue
                    tcs = tool_calls(m)
                    c = calls.get(i) or {}
                    pt = (c.get("usage") or {}).get("prompt_tokens")
                    rep = any(seen[tc] >= 1 for tc in tcs)
                    reread = any(read_seen[tc[1]] >= 1 for tc in tcs if tc[0] == "Read")
                    for tc in tcs:
                        seen[tc] += 1
                        if tc[0] == "Read":
                            read_seen[tc[1]] += 1
                    steps.append({
                        "reason": len(m.get("reasoning_content") or ""),
                        "prompt_tokens": pt,
                        "tcs": tcs,
                        "repeat": rep, "reread": reread,
                        "has_read": any(tc[0] == "Read" for tc in tcs),
                        "ronly": bool(m.get("reasoning_content")) and not tcs
                                 and not content_text(m.get("content")).strip(),
                    })
                all_tcs = [tc for s_ in steps for tc in s_["tcs"]]
                rec = {
                    "solved": s == 1.0, "scored": s is not None, "steps": steps,
                    "ncalls": len(steps),
                    "rmed": st.median([s_["reason"] for s_ in steps]) if steps else 0,
                    "rsum": sum(s_["reason"] for s_ in steps),
                    "rep": Counter(all_tcs).most_common(1)[0][1] if all_tcs else 0,
                    "tests": sum(1 for n, a in all_tcs if n == "Bash" and TEST_RE.search(a)),
                    "edits": sum(1 for n, _ in all_tcs if n in ("Edit", "Write", "MultiEdit")),
                    "ronly": sum(1 for s_ in steps if s_["ronly"]),
                    "lencut": sum(1 for c in calls.values() if c.get("finish_reason") == "length"),
                    "maxin": max([(s_["prompt_tokens"] or 0) for s_ in steps] or [0]),
                }
                prev = res.get(name)
                if prev is None or (rec["scored"] and not prev["scored"]) or (rec["solved"] and not prev["solved"]):
                    res[name] = rec
    return res


def med(rs, key) -> str:
    return f"{st.median(r[key] for r in rs):.0f}" if rs else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--king", required=True, help="label glob prefix of the king traces")
    ap.add_argument("--teacher", default="teacher")
    ap.add_argument("--teacher-root", default=None, help="root of the teacher traces if different")
    args = ap.parse_args()

    K = load(args.root, args.king)
    T = load(args.teacher_root or args.root, args.teacher)
    common = [n for n in K if n in T and K[n]["scored"] and T[n]["scored"]]
    classes = {
        "both": [n for n in common if K[n]["solved"] and T[n]["solved"]],
        "teacher-only": [n for n in common if T[n]["solved"] and not K[n]["solved"]],
        "king-only": [n for n in common if K[n]["solved"] and not T[n]["solved"]],
        "neither": [n for n in common if not K[n]["solved"] and not T[n]["solved"]],
    }
    print(f"paired scored tasks: {len(common)}  " + "  ".join(f"{k} {len(v)}" for k, v in classes.items()))
    print(f"\n== 1. per outcome class (median per run) ==")
    print(f"{'class':<13} {'model':<8} {'n':>4} | {'steps':>5} {'reason/step':>11} {'reason/run':>10} "
          f"{'repeat-max':>10} {'tests':>5} {'edits':>5} {'reason-only':>11} {'len-cut':>7} {'max-prompt':>10}")
    for lab, names in classes.items():
        for who, D in ((args.king, K), (args.teacher, T)):
            rs = [D[n] for n in names]
            if not rs:
                continue
            print(f"{lab:<13} {who:<8} {len(rs):>4} | {med(rs, 'ncalls'):>5} {med(rs, 'rmed'):>11} "
                  f"{med(rs, 'rsum'):>10} {med(rs, 'rep'):>10} {med(rs, 'tests'):>5} {med(rs, 'edits'):>5} "
                  f"{med(rs, 'ronly'):>11} {med(rs, 'lencut'):>7} {med(rs, 'maxin'):>10}")
    for who, D in ((args.king, K), (args.teacher, T)):
        rs = [D[n] for n in common]
        print(f"{who:<8} identical call >=3x in {sum(1 for r in rs if r['rep'] >= 3) / len(rs):.0%} of runs, "
              f">=5x in {sum(1 for r in rs if r['rep'] >= 5) / len(rs):.0%}; >=3 reasoning-only replies "
              f"{sum(1 for r in rs if r['ronly'] >= 3) / len(rs):.0%}; zero test runs "
              f"{sum(1 for r in rs if r['tests'] == 0) / len(rs):.0%}")
    rs = [K[n] for n in common]
    for thr in (3, 5):
        a = [r for r in rs if r["rep"] >= thr]
        b = [r for r in rs if r["rep"] < thr]
        print(f"{args.king}: solve rate when an identical call repeats >={thr}x: "
              f"{sum(r['solved'] for r in a) / max(1, len(a)):.2f} (n={len(a)}) vs "
              f"{sum(r['solved'] for r in b) / max(1, len(b)):.2f} (n={len(b)})")

    print(f"\n== 2. per prompt-depth bin (all scored runs) ==")
    for who, D in ((args.king, K), (args.teacher, T)):
        tot = Counter()
        rep = Counter()
        rer = Counter()
        rd = Counter()
        ro = Counter()
        rl = defaultdict(list)
        for n in common:
            for s_ in D[n]["steps"]:
                pt = s_["prompt_tokens"]
                if pt is None:
                    continue
                b = next(i for i, (lo, hi) in enumerate(DEPTH_BINS) if lo <= pt < hi)
                tot[b] += 1
                rep[b] += s_["repeat"]
                rd[b] += s_["has_read"]
                rer[b] += s_["reread"]
                ro[b] += s_["ronly"]
                rl[b].append(s_["reason"])
        print(who)
        for b, (lo, hi) in enumerate(DEPTH_BINS):
            n = tot[b] or 1
            print(f"  prompt {lo // 1000:>3}k-{(hi // 1000 if hi < 10**9 else 'inf'):>3}k tok: steps {tot[b]:>6} | "
                  f"repeats an earlier identical call {rep[b] / n:5.1%} | re-Reads a file already read "
                  f"{rer[b] / max(1, rd[b]):5.1%} of Read steps | reasoning-only {ro[b] / n:5.1%} | "
                  f"median reasoning chars {np.median(rl[b]) if rl[b] else 0:5.0f}")
        allpt = [s_["prompt_tokens"] for n in common for s_ in D[n]["steps"] if s_["prompt_tokens"]]
        print(f"  share of steps beyond D's 120k-char prefix cap (~32k tok): {np.mean(np.array(allpt) > 32_000):.1%}; "
              f"beyond 64k tok: {np.mean(np.array(allpt) > 64_000):.1%}")

    print(f"\n== 3. reasoning length per step vs the duel's thought cap ==")
    for who, D in ((args.king, K), (args.teacher, T)):
        r = np.array([s_["reason"] for n in common for s_ in D[n]["steps"]])
        cap = 1024 * CHARS_PER_TOKEN
        print(f"{who:<8} steps {len(r)}  p50 {np.median(r):.0f}  p90 {np.percentile(r, 90):.0f}  p99 {np.percentile(r, 99):.0f} chars | "
              f"> 1024-token cap: {np.mean(r > cap):.1%} | > 1792 tokens: {np.mean(r > 1792 * CHARS_PER_TOKEN):.1%} | "
              f"share of all reasoning chars in steps over the cap: {r[r > cap].sum() / max(1, r.sum()):.0%}")


if __name__ == "__main__":
    main()
