"""Stage 4 — READ the probe: classification tables, per-arm solve rates on
contested vs agreed states with paired statistics, agreement-with-delay,
disagreement taxonomy; writes report.txt + report.json.

    python analyze.py --samples .../samples.jsonl --kept .../kept.jsonl \
        --results .../continuations.jsonl --out .../report

Terms (one line each)
  state          a prefix of a teacher trajectory (depth = number of teacher replies before it)
  contested      frontier (glm-5.3 greedy) action differs from all 3 teacher samples (norm-exact) and max token-Jaccard < 0.5
  agreed         not contested
  judge          glm-5.3-flash T=0 verdict on whether frontier action and closest teacher action are the same decision
  kept           states that went to the pod: contested AND judge=different, or agreed AND judge=same
  arm T          teacher (qwen3.8-27b @T0.8) finishes the episode from the state
  arm F          frontier (glm-5.3 @T0.8) finishes the episode from the state
  arm F1         frontier's greedy action is executed as the next step, then the teacher finishes
  solved         the environment's own grader gives the continuation its primary reward >= 1.0
  F1-T           per-state difference of solved indicators (1 = F1 solved, T did not; -1 the reverse)
  sign test      exact binomial test on the discordant pairs (P(+)=0.5 under H0)
  bootstrap CI   95 % percentile interval of the mean paired difference over 10,000 state resamples
  delay-agree    in arm T, the teacher's first k replies contain an action norm-exact / Jaccard>=0.5 equal to the frontier's contested action
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import exact, jaccard, read_jsonl, reply_to_rollout  # noqa: E402

ARMS = ("T", "F1", "F")


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(p_le, p_ge))


def boot_ci(diffs: list[float], n: int = 10_000, seed: int = 1) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n) -> str:
    return f"{k}/{n} = {100 * k / n:5.1f}%" if n else "  —  "


def paired_block(name: str, pairs: list[tuple[int, int]]) -> tuple[list[str], dict]:
    """pairs = [(solved_arm, solved_T)]."""
    n = len(pairs)
    a = sum(x for x, _ in pairs)
    t = sum(y for _, y in pairs)
    diffs = [x - y for x, y in pairs]
    plus = sum(1 for d in diffs if d > 0)
    minus = sum(1 for d in diffs if d < 0)
    p = binom_two_sided(plus, plus + minus)
    lo, hi = boot_ci([float(d) for d in diffs])
    mean = sum(diffs) / n if n else float("nan")
    lines = [f"  {name}: n={n}  solved arm {pct(a, n)}  solved T {pct(t, n)}  "
             f"mean diff {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  discordant +{plus}/-{minus}  sign p={p:.3f}"]
    return lines, {"n": n, "solved_arm": a, "solved_T": t, "mean_diff": mean, "ci95": [lo, hi],
                   "plus": plus, "minus": minus, "sign_p": p}


def arm_actions(trace: dict, kind: str, k: int) -> list[str]:
    """The first k actions of a stored continuation trace, in the dialect."""
    out = []
    for nd in trace.get("nodes") or []:
        m = nd.get("message") or {}
        if not (nd.get("sampled") and m.get("role") == "assistant"):
            continue
        reply = {"content": (m.get("content") or "").replace("```mswea_bash_command\n", "```bash\n"),
                 "reasoning": "", "tool_calls": [
                     {"function": {"name": tc.get("name") or (tc.get("function") or {}).get("name"),
                                   "arguments": tc.get("arguments") if tc.get("arguments") is not None
                                   else (tc.get("function") or {}).get("arguments")}}
                     for tc in (m.get("tool_calls") or [])]}
        y = reply_to_rollout(reply, kind)["y"]
        if y:
            out.append(y)
        if len(out) >= k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True, help="continuations.jsonl (all arms)")
    ap.add_argument("--traces", type=Path, default=None, help="dir of <arm>/<state_id>.json traces (delay-agree)")
    ap.add_argument("--out", type=Path, required=True, help="report path stem")
    args = ap.parse_args()
    samples = read_jsonl(args.samples)
    kept = {r["state_id"]: r for r in read_jsonl(args.kept)}
    results = read_jsonl(args.results)
    L: list[str] = []
    J: dict = {}

    # ---------------------------------------------------------- classification
    L.append("=" * 78)
    L.append("1. CLASSIFICATION (all candidate states; frontier = glm-5.3 greedy vs 3 teacher samples @T0.8)")
    L.append("=" * 78)
    cls = collections.Counter(r["class"] for r in samples)
    L.append(f"candidates {len(samples)}: " + ", ".join(f"{k} {v}" for k, v in sorted(cls.items())))
    usable = [r for r in samples if r["class"] in ("contested", "agreed")]
    J["classification"] = {"n": len(samples), "classes": dict(cls)}

    def rate_table(keyf, title):
        L.append(f"-- contested rate by {title} --")
        groups = collections.defaultdict(list)
        for r in usable:
            groups[keyf(r)].append(r)
        tab = {}
        for k in sorted(groups, key=str):
            g = groups[k]
            c = sum(1 for r in g if r["class"] == "contested")
            jd = [r for r in g if "same_decision" in (r.get("judge") or {})]
            # judge agreement with the surface rule
            agree_c = sum(1 for r in jd if r["class"] == "contested" and r["judge"]["same_decision"] is False)
            agree_a = sum(1 for r in jd if r["class"] == "agreed" and r["judge"]["same_decision"] is True)
            lo, hi = wilson(c, len(g))
            L.append(f"  {str(k):34s} contested {pct(c, len(g))} [{100*lo:4.0f}–{100*hi:3.0f}]  "
                     f"judge-differs|contested {pct(agree_c, sum(1 for r in jd if r['class']=='contested'))}  "
                     f"judge-same|agreed {pct(agree_a, sum(1 for r in jd if r['class']=='agreed'))}")
            tab[str(k)] = {"n": len(g), "contested": c, "judge_differs_given_contested":
                           [agree_c, sum(1 for r in jd if r["class"] == "contested")],
                           "judge_same_given_agreed": [agree_a, sum(1 for r in jd if r["class"] == "agreed")]}
        J[f"contested_by_{title}"] = tab

    rate_table(lambda r: "ALL", "all")
    rate_table(lambda r: r["harness"], "harness")
    rate_table(lambda r: r["group"] + "/" + r["source"], "source")
    rate_table(lambda r: f"depth {3 if r['depth'] <= 5 else 6 if r['depth'] <= 9 else 10}–"
               f"{5 if r['depth'] <= 5 else 9 if r['depth'] <= 9 else 15}", "depth")
    rate_table(lambda r: "orig " + str(r["orig_outcome"]), "orig_outcome")
    # teacher self-agreement and frontier self-agreement
    tsa = [r for r in usable if r.get("t_pairwise_n")]
    L.append("-- reference spreads --")
    L.append(f"  teacher 3-sample pairwise norm-exact agreement: "
             f"{sum(r['t_pairwise_exact'] for r in tsa)}/{sum(r['t_pairwise_n'] for r in tsa)}"
             f" = {100*sum(r['t_pairwise_exact'] for r in tsa)/max(1,sum(r['t_pairwise_n'] for r in tsa)):.1f}%;"
             f" mean pairwise Jaccard {sum(r['t_jaccard_mean'] for r in tsa)/len(tsa):.2f}")
    fsa = [r["f_t08_vs_greedy_jaccard"] for r in usable if r.get("f_t08_vs_greedy_jaccard") is not None]
    L.append(f"  frontier greedy vs frontier @T0.8 Jaccard: mean {sum(fsa)/len(fsa):.2f} "
             f"(share >=0.5: {100*sum(1 for x in fsa if x >= 0.5)/len(fsa):.0f}%)")
    fo = [r for r in usable if r.get("f_jaccard_orig") is not None]
    L.append(f"  frontier greedy vs the teacher's ORIGINAL action at the state: norm-exact "
             f"{pct(sum(1 for r in fo if r['f_exact_orig']), len(fo))}, Jaccard>=0.5 "
             f"{pct(sum(1 for r in fo if r['f_jaccard_orig'] >= 0.5), len(fo))}")
    hv = [r for r in usable if r["frontier_greedy"]]
    L.append(f"  frontier greedy reply harness-valid as emitted: "
             f"{pct(sum(1 for r in hv if r['frontier_greedy']['harness_valid']), len(hv))}"
             f" (fence fixed ```bash->mswea: {sum(1 for r in hv if r['frontier_greedy'].get('fence_fixed'))})")
    # judge taxonomy on all contested + agreed
    L.append("-- judge taxonomy (frontier action relative to closest teacher action) --")
    for c in ("contested", "agreed"):
        rel = collections.Counter((r.get("judge") or {}).get("relation", "?") for r in usable if r["class"] == c)
        typ = collections.Counter((r.get("judge") or {}).get("a_type", "?") for r in usable if r["class"] == c)
        L.append(f"  {c:9s} relation: " + ", ".join(f"{k} {v}" for k, v in rel.most_common()))
        L.append(f"  {c:9s} frontier a_type: " + ", ".join(f"{k} {v}" for k, v in typ.most_common()))
        J[f"taxonomy_{c}"] = {"relation": dict(rel), "a_type": dict(typ)}

    # ---------------------------------------------------------- continuations
    L.append("")
    L.append("=" * 78)
    L.append("2. CONTINUATIONS (kept states; solved = env grader reward >= 1.0; errored/unrun excluded per pair)")
    L.append("=" * 78)
    by_state: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        sid, arm = r["state_id"].rsplit(":", 1)
        by_state[sid + ":frontier"][arm].append(r)
    status = collections.Counter((r["state_id"].rsplit(":", 1)[1], r.get("status")) for r in results)
    L.append("runs per arm/status: " + ", ".join(f"{a}/{s} {n}" for (a, s), n in sorted(status.items())))
    J["runs"] = {f"{a}/{s}": n for (a, s), n in status.items()}

    def solved_of(runs: list[dict]) -> int | None:
        ok = [r for r in runs if r.get("status") == "ok"]
        if not ok:
            return None
        return 1 if any(r.get("outcome") == "solved" for r in ok) else 0   # 1 continuation per arm in practice

    J["arms"] = {}
    for label, sel in (("CONTESTED", lambda k: k["kept_class"] == "contested"),
                       ("AGREED", lambda k: k["kept_class"] == "agreed")):
        ks = [k for k in kept.values() if sel(k)]
        L.append(f"-- {label} kept states: {len(ks)} --")
        J["arms"][label] = {}
        for arm in ARMS:
            rows = []
            for k in ks:
                s = solved_of(by_state[k["state_id"]].get(arm, []))
                if s is not None:
                    rows.append(s)
            if rows:
                lo, hi = wilson(sum(rows), len(rows))
                L.append(f"  arm {arm:2s} solved {pct(sum(rows), len(rows))} [{100*lo:4.0f}–{100*hi:3.0f}]")
                J["arms"][label][arm] = {"solved": sum(rows), "n": len(rows)}
        for arm in ("F1", "F"):
            pairs = []
            for k in ks:
                a = solved_of(by_state[k["state_id"]].get(arm, []))
                t = solved_of(by_state[k["state_id"]].get("T", []))
                if a is not None and t is not None:
                    pairs.append((a, t))
            if pairs:
                lines, d = paired_block(f"{arm}−T", pairs)
                L += lines
                J["arms"][label][f"{arm}-T"] = d
        # splits
        for split_name, keyf in (("harness", lambda k: k["harness"]), ("orig outcome", lambda k: k["orig_outcome"]),
                                 ("group", lambda k: k["group"]),
                                 ("judge relation", lambda k: (k.get("judge") or {}).get("relation", "?"))):
            L.append(f"  by {split_name}:")
            for val in sorted({keyf(k) for k in ks}, key=str):
                sub = [k for k in ks if keyf(k) == val]
                cells = []
                for arm in ARMS:
                    rows = [solved_of(by_state[k["state_id"]].get(arm, [])) for k in sub]
                    rows = [x for x in rows if x is not None]
                    cells.append(f"{arm} {pct(sum(rows), len(rows))}" if rows else f"{arm}   —  ")
                pairs = [(solved_of(by_state[k["state_id"]].get("F1", [])), solved_of(by_state[k["state_id"]].get("T", [])))
                         for k in sub]
                pairs = [(a, t) for a, t in pairs if a is not None and t is not None]
                plus = sum(1 for a, t in pairs if a > t)
                minus = sum(1 for a, t in pairs if a < t)
                L.append(f"    {str(val):22s} n={len(sub):3d}  " + "  ".join(cells) +
                         f"  F1−T +{plus}/−{minus} p={binom_two_sided(plus, plus+minus):.2f}")

    # ---------------------------------------------------------- delay agreement
    if args.traces:
        L.append("")
        L.append("=" * 78)
        L.append("3. AGREEMENT WITH DELAY (contested states, arm T: does the teacher take the frontier's action within k steps?)")
        L.append("=" * 78)
        hits = {1: 0, 3: 0, 5: 0}
        hits_j = {1: 0, 3: 0, 5: 0}
        n = 0
        first_action_same = 0
        for k in kept.values():
            if k["kept_class"] != "contested":
                continue
            p = args.traces / (k["state_id"].replace(":frontier", "_T").replace(":", "_") + ".json")
            if not p.exists():
                continue
            trace = json.loads(p.read_text())
            acts = arm_actions(trace, k["action_kind"], 5)
            if not acts:
                continue
            n += 1
            fy = k["frontier_greedy"]["y"]
            for kk in hits:
                if any(exact(fy, [a], k["action_kind"]) for a in acts[:kk]):
                    hits[kk] += 1
                if any(jaccard(fy, a) >= 0.5 for a in acts[:kk]):
                    hits_j[kk] += 1
            first_action_same += exact(acts[0], [k["closest_teacher_action"]], k["action_kind"])
        L.append(f"  contested states with a T trace: {n}")
        for kk in hits:
            L.append(f"  within {kk} step(s): norm-exact {pct(hits[kk], n)}   Jaccard>=0.5 {pct(hits_j[kk], n)}")
        L.append(f"  (T's first action norm-exact = the closest sampled teacher action: {pct(first_action_same, n)})")
        J["delay_agree"] = {"n": n, "exact": hits, "jaccard": hits_j}

    # ---------------------------------------------------------- cost
    cost = sum(float(r.get("cost_usd") or 0) for r in results)
    cost_s = sum(float(r.get("cost_usd") or 0) for r in samples)
    L.append("")
    L.append(f"COST  classification ${cost_s:.2f} (Engy charged_micro)  continuations ${cost:.2f} "
             f"(Engy list price from trace usage)  total ${cost + cost_s:.2f}")
    J["cost_usd"] = {"classification": cost_s, "continuations": cost, "total": cost + cost_s}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".txt").write_text("\n".join(L) + "\n")
    Path(str(args.out) + ".json").write_text(json.dumps(J, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
