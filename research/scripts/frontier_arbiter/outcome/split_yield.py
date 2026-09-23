"""Split-states probe (2026-09-21) — READ: Y1 (can split decision states be
harvested at a high rate from cheap candidates?) and Y2 (at states the
teacher never solves, do frontier proposals continued by the teacher solve?).
Writes report.txt + report.json.

    python split_yield.py --selected .../split_states/selected.jsonl \
        --results .../split_states/continuations.jsonl --samples .../outcome/samples.jsonl \
        --kept .../outcome/kept.jsonl --prev .../outcome/continuations.jsonl \
        --proposals .../split_states/proposals.jsonl --out .../split_states/report

Terms (one line each)
  state              a prefix of a public teacher trajectory (depth = teacher replies before it), rebuilt in a fresh container
  origin             outcome of the trajectory the state was cut from: solved / failed (the env grader on the ORIGINAL run)
  continuation       the teacher (qwen3.8-27b @T0.8) finishes the episode from the state under the same harness; solved = primary reward >= 1.0
  N, s               OK (non-errored) continuations of a state, and how many of them solved
  split state        0 < s < N: the teacher's own samples sometimes solve and sometimes fail from the same state
  ceiling state      s = 0 with N >= 3: the teacher never solves from here
  split rate         share of states with N >= 3 that are split, by origin and harness (Wilson 95 % CI)
  first action       the harness action of a continuation's first reply (norm_action; token-Jaccard for the loose form)
  decision class     a group of first actions that are norm-exact equal (strict) or linked by Jaccard >= 0.5 (loose, single linkage)
  first-action divergence  in a split state, the solved and the failed continuations share NO decision class (the choice at the state decides)
  diverges later     a split state where some solved and some failed continuation start with the same class
  yield              kept split states per 100 candidates = 100 x P(split) under the candidate origin mix; continuations per kept = N / P(split)
  sequential design  run continuations one at a time, stop at the first split (max N); expected continuations from all orderings of each state's outcome vector
  proposal           a frontier (glm-5.3) action sampled at the state (greedy, T0.8, extra T0.8)
  arm F1             the proposal is executed as the next step, then the teacher finishes; F1 solved = that continuation solved
  above the ceiling  a ceiling state (or a yesterday state whose only T continuation failed) where some F1 continuation solved
  novel proposal     norm-exact different from every teacher first action seen at the state (continuations + the 3 cheap API samples); novel_loose = also Jaccard < 0.5 to all of them (=teacher / near-teacher / novel in the per-state list)
  judge taxonomy     glm-5.3-flash label of the proposal vs the closest teacher action: relation (same/different_target/act_or_finish/explore_more/different_fix), a_type (explore/act/verify/finish)
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import jaccard, norm_action, read_jsonl, reply_to_rollout  # noqa: E402

HARNESSES = ("mini_swe_textbased", "bash", "terminus_2")
LOOSE = 0.5


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
    if not n:
        return "   —   "
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:5.1f}% [{100 * lo:3.0f}–{100 * hi:3.0f}]"


def first_action(row: dict, kind: str) -> str:
    fr = row.get("first_reply") or {}
    content = (fr.get("content") or "").replace("```mswea_bash_command\n", "```bash\n")
    tcs = [{"function": {"name": tc.get("name") or (tc.get("function") or {}).get("name"),
                         "arguments": tc.get("arguments") if tc.get("arguments") is not None
                         else (tc.get("function") or {}).get("arguments")}}
           for tc in (fr.get("tool_calls") or [])]
    return reply_to_rollout({"content": content, "reasoning": "", "tool_calls": tcs}, kind)["y"] or ""


def classes(actions: list[str], kind: str, loose: bool) -> list[int]:
    """Class index per action: strict = norm-exact; loose = single-linkage Jaccard >= 0.5."""
    if not loose:
        seen: dict[str, int] = {}
        return [seen.setdefault(norm_action(a, kind), len(seen)) for a in actions]
    n = len(actions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if norm_action(actions[i], kind) == norm_action(actions[j], kind) or jaccard(actions[i], actions[j]) >= LOOSE:
                parent[find(i)] = find(j)
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def seq_expected(outcomes: list[int], cap: int) -> tuple[float, float]:
    """Sequential design over all orderings of this state's outcome vector:
    (expected continuations used, P(split found within cap))."""
    n = len(outcomes)
    tot = found = 0.0
    perms = set(itertools.permutations(outcomes))
    for perm in perms:
        used = n
        hit = 0.0
        for k in range(2, min(cap, n) + 1):
            s = sum(perm[:k])
            if 0 < s < k:
                used, hit = k, 1.0
                break
        else:
            used = min(cap, n)
        tot += used
        found += hit
    return tot / len(perms), found / len(perms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True, help="split_states/continuations.jsonl (arms T, F1*)")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--prev", type=Path, required=True, help="yesterday's continuations.jsonl")
    ap.add_argument("--proposals", type=Path, default=None, help="proposals.jsonl from select_f1.py")
    ap.add_argument("--n-cap", type=int, default=4)
    ap.add_argument("--notes", type=Path, default=None, help="run notes prepended to report.txt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    selected = {m["state_id"].replace(":T", ":frontier"): m for m in read_jsonl(args.selected)}
    samples = {r["state_id"]: r for r in read_jsonl(args.samples)}
    kept = {r["state_id"]: r for r in read_jsonl(args.kept)}
    results = read_jsonl(args.results)
    prev = read_jsonl(args.prev)
    proposals = {p["state_id"]: p for p in read_jsonl(args.proposals)} if args.proposals and args.proposals.exists() else {}
    L: list[str] = []
    J: dict = {}
    if args.notes and args.notes.exists():
        L += ["RUN NOTES", args.notes.read_text().rstrip(), ""]

    by_state: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        sid, arm = r["state_id"].rsplit(":", 1)
        by_state[sid + ":frontier"][arm].append(r)

    # ------------------------------------------------------------------ Y1
    L.append("=" * 78)
    n_f = sum(1 for m in selected.values() if m["orig_outcome"] == "failed")
    L.append(f"Y1. SPLIT STATES — {len(selected)} states ({n_f} failed-origin x {args.n_cap} + {len(selected) - n_f} solved-origin x 3 "
             f"teacher continuations @T0.8; one of them = yesterday's arm T where it existed; solved = env grader primary reward >= 1.0; errored excluded)")
    L.append("=" * 78)
    per_state = []
    for fid, m in selected.items():
        runs = by_state[fid].get("T", [])
        ok = [r for r in runs if r.get("status") == "ok"]
        outcomes = [1 if r.get("outcome") == "solved" else 0 for r in ok]
        kind = m["action_kind"]
        firsts = [first_action(r, kind) for r in ok]
        api = [p["y"] for p in (samples.get(fid) or {}).get("teacher") or [] if p and p.get("y")]
        st = {"state_id": fid, "harness": m["harness"], "source": m["source"], "origin": m["orig_outcome"],
              "depth": m["depth"], "N": len(ok), "s": sum(outcomes), "errored": len(runs) - len(ok),
              "outcomes": outcomes, "first_actions": firsts, "api_first_actions": api,
              "kept_class": (kept.get(fid) or {}).get("kept_class")}
        st["split"] = 0 < st["s"] < st["N"]
        st["ceiling"] = st["N"] >= 3 and st["s"] == 0
        if firsts:
            st["classes_strict"] = len(set(classes(firsts, kind, False)))
            st["classes_loose"] = len(set(classes(firsts, kind, True)))
            allf = firsts + api
            st["classes_strict_7"] = len(set(classes(allf, kind, False)))
            st["classes_loose_7"] = len(set(classes(allf, kind, True)))
        if st["split"]:
            for loose, key in ((False, "strict"), (True, "loose")):
                cl = classes(firsts, kind, loose)
                sol = {c for c, o in zip(cl, outcomes) if o}
                fail = {c for c, o in zip(cl, outcomes) if not o}
                st[f"diverge_first_{key}"] = not (sol & fail)
        if st["N"] >= 2:
            st["seq_exp_n"], st["seq_p_found"] = seq_expected(outcomes, args.n_cap)
        per_state.append(st)
    J["states"] = per_state
    full = [s for s in per_state if s["N"] >= 3]
    L.append(f"states with >= 3 OK continuations: {len(full)} / {len(per_state)}; "
             f"errored continuations: {sum(s['errored'] for s in per_state)}; "
             f"s/N histogram: " + ", ".join(f"{k} x{v}" for k, v in sorted(collections.Counter(f"{s['s']}/{s['N']}" for s in full).items())))
    L.append("-- split rate (0 < s < N) by origin x harness --")
    tab = {}
    for origin in ("failed", "solved", "ALL"):
        for h in HARNESSES + ("ALL",):
            g = [s for s in full if (origin == "ALL" or s["origin"] == origin) and (h == "ALL" or s["harness"] == h)]
            if not g:
                continue
            sp = sum(1 for s in g if s["split"])
            ce = sum(1 for s in g if s["ceiling"])
            al = sum(1 for s in g if s["s"] == s["N"])
            L.append(f"  origin {origin:6s} {h:20s} split {pct(sp, len(g)):28s} ceiling(0/N) {ce:2d}  always(N/N) {al:2d}"
                     f"  mean s/N {sum(s['s'] / s['N'] for s in g) / len(g):.2f}")
            tab[f"{origin}/{h}"] = {"n": len(g), "split": sp, "ceiling": ce, "always": al}
    J["split_rate"] = tab
    L.append("-- split rate by depth --")
    for lo, hi in ((3, 5), (6, 9), (10, 15)):
        g = [s for s in full if lo <= s["depth"] <= hi]
        if g:
            L.append(f"  depth {lo:2d}–{hi:2d}: split {pct(sum(1 for s in g if s['split']), len(g))}  "
                     f"(failed-origin {pct(sum(1 for s in g if s['split'] and s['origin']=='failed'), sum(1 for s in g if s['origin']=='failed'))})")
    L.append("-- versus yesterday's single continuation (prev_T) and the frontier classification --")
    for key, name in (("prev_T", "prev_T"), ("kept_class", "kept_class")):
        c = collections.Counter()
        for s in full:
            v = selected[s["state_id"]].get("prev_T") if key == "prev_T" else s["kept_class"]
            c[(str(v), "split" if s["split"] else "ceiling" if s["ceiling"] else "always" if s["s"] == s["N"] else "other")] += 1
        L.append(f"  by {name}: " + ", ".join(f"{k[0]}→{k[1]} {v}" for k, v in sorted(c.items())))
    # decision classes
    L.append("-- decision classes among the first actions (N continuations; +3 = with the 3 cheap API samples of the same state) --")
    with_f = [s for s in full if s.get("classes_strict")]
    for key in ("classes_strict", "classes_loose", "classes_strict_7", "classes_loose_7"):
        vals = [s[key] for s in with_f]
        if not vals:
            break
        L.append(f"  {key:18s} mean {sum(vals) / len(vals):.2f}  hist " +
                 ", ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(vals).items())) +
                 f"   (split states: mean {sum(s[key] for s in with_f if s['split']) / max(1, sum(1 for s in with_f if s['split'])):.2f}; "
                 f"ceiling: {sum(s[key] for s in with_f if s['ceiling']) / max(1, sum(1 for s in with_f if s['ceiling'])):.2f})")
    J["classes"] = {k: [s[k] for s in with_f] for k in ("classes_strict", "classes_loose", "classes_strict_7", "classes_loose_7")}
    splits = [s for s in full if s["split"]]
    L.append("-- first-action divergence in split states (solved vs failed continuations share no decision class) --")
    for key in ("strict", "loose"):
        d = sum(1 for s in splits if s.get(f"diverge_first_{key}"))
        L.append(f"  {key:6s}: diverge at the state {pct(d, len(splits))}; diverge later {len(splits) - d}")
        for h in HARNESSES:
            g = [s for s in splits if s["harness"] == h]
            if g:
                L.append(f"      {h:20s} {pct(sum(1 for s in g if s.get(f'diverge_first_{key}')), len(g))}")
    J["divergence"] = {k: [s.get(f"diverge_first_{k}") for s in splits] for k in ("strict", "loose")}
    # yield
    L.append("-- implied yield (candidate origin mix from the 245 cheap candidates; N-cap = %d) --" % args.n_cap)
    cand_mix = collections.Counter(r["orig_outcome"] for r in samples.values() if r.get("orig_outcome") in ("failed", "solved"))
    tot = sum(cand_mix.values())
    p_split = {}
    for origin in ("failed", "solved"):
        g = [s for s in full if s["origin"] == origin]
        p_split[origin] = sum(1 for s in g if s["split"]) / len(g) if g else float("nan")
    p_mix = sum(p_split[o] * cand_mix[o] / tot for o in p_split)
    seq = [s for s in full if "seq_exp_n" in s]
    seq_n = {o: sum(s["seq_exp_n"] for s in seq if s["origin"] == o) / max(1, sum(1 for s in seq if s["origin"] == o)) for o in ("failed", "solved")}
    seq_p = {o: sum(s["seq_p_found"] for s in seq if s["origin"] == o) / max(1, sum(1 for s in seq if s["origin"] == o)) for o in ("failed", "solved")}
    for o in ("failed", "solved"):
        if math.isnan(p_split[o]):
            continue
        n_o = args.n_cap if o == "failed" else 3
        L.append(f"  {o:6s}-origin: P(split|N={n_o}) {p_split[o]:.2f} → kept per 100 candidates {100 * p_split[o]:.0f}, "
                 f"continuations per kept {n_o / p_split[o] if p_split[o] else float('inf'):.1f} (fixed N); "
                 f"sequential: {seq_n[o]:.2f} cont/state, P(found) {seq_p[o]:.2f} → {seq_n[o] / seq_p[o] if seq_p[o] else float('inf'):.1f} per kept")
    L.append(f"  candidate mix failed {cand_mix['failed']} / solved {cand_mix['solved']}: P(split) {p_mix:.2f} → "
             f"{100 * p_mix:.0f} kept per 100 candidates, {args.n_cap / p_mix if p_mix else float('inf'):.1f} continuations per kept (fixed N); "
             f"failed-only candidates: {100 * p_split['failed']:.0f} per 100, {args.n_cap / p_split['failed'] if p_split['failed'] else float('inf'):.1f} per kept")
    J["yield"] = {"p_split": p_split, "p_split_mix": p_mix, "cand_mix": dict(cand_mix), "seq_n": seq_n, "seq_p": seq_p}
    L.append("-- per state (s/N, strict/loose decision classes among the N first actions, divergence for splits) --")
    for st in sorted(per_state, key=lambda s: (s["origin"], s["harness"], -s["N"], s["s"])):
        tag = "SPLIT" if st["split"] else "CEIL" if st["ceiling"] else "ALL" if st["N"] and st["s"] == st["N"] else "incomplete"
        L.append(f"    {st['origin']:6s} {st['harness']:18s} {st['source']:16s} d={st['depth']:2d} {st['state_id'][:8]}  "
                 f"s/N {st['s']}/{st['N']} err {st['errored']}  classes {st.get('classes_strict', '-')}/{st.get('classes_loose', '-')}  "
                 f"{tag:10s}" + (f" diverge strict={st.get('diverge_first_strict')} loose={st.get('diverge_first_loose')}" if st["split"] else ""))

    # ------------------------------------------------------------------ Y2
    L.append("")
    L.append("=" * 78)
    L.append("Y2. FRONTIER PROPOSALS AT TEACHER-CEILING STATES (arm F1: proposal executed, teacher finishes)")
    L.append("=" * 78)
    prev_T = {r["kept_state_id"]: r for r in prev if r.get("arm") == "T"}
    prev_F1 = collections.defaultdict(list)
    for r in prev:
        if r.get("arm") == "F1":
            prev_F1[r["kept_state_id"]].append(r)
    y2_rows = []
    f1_arms = ("F1", "F1b", "F1c")
    for fid, p in proposals.items():
        st = next((s for s in per_state if s["state_id"] == fid), None)
        t_runs = [r for r in by_state[fid].get("T", []) if r.get("status") == "ok"]
        if st is None:
            pr = prev_T.get(fid)
            t_desc = f"0/1 (yesterday)" if pr and pr.get("status") == "ok" and pr.get("outcome") != "solved" else "?"
            t_s, t_n = (0, 1) if pr else (0, 0)
        else:
            t_s, t_n = st["s"], st["N"]
            t_desc = f"{t_s}/{t_n}"
        teacher_firsts = [first_action(r, p["action_kind"]) for r in t_runs] + \
            [q["y"] for q in (samples.get(fid) or {}).get("teacher") or [] if q and q.get("y")]
        props = []
        for pr in p["proposals"]:
            runs = [r for r in by_state[fid].get(pr["arm"], []) if r.get("status") == "ok"]
            if not runs and pr["arm"] == "F1":
                runs = [r for r in prev_F1.get(fid, []) if r.get("status") == "ok"]
            solved = None if not runs else int(any(r.get("outcome") == "solved" for r in runs))
            novel = all(norm_action(pr["y"], p["action_kind"]) != norm_action(t, p["action_kind"]) for t in teacher_firsts)
            novel_loose = all(jaccard(pr["y"], t) < LOOSE for t in teacher_firsts)
            props.append({"arm": pr["arm"], "source": pr["source"], "solved": solved, "novel": novel, "novel_loose": novel_loose,
                          "relation": (pr.get("judge") or {}).get("relation"), "a_type": (pr.get("judge") or {}).get("a_type"),
                          "y_head": pr["y"][:120]})
        y2_rows.append({"state_id": fid, "harness": p["harness"], "source": p["source"], "origin": p["orig_outcome"],
                        "teacher": t_desc, "teacher_s": t_s, "teacher_n": t_n, "proposals": props})
    J["y2"] = y2_rows
    ran = [r for r in y2_rows if any(q["solved"] is not None for q in r["proposals"])]
    L.append(f"ceiling states with an F1 result: {len(ran)} / {len(y2_rows)} proposed "
             f"(teacher 0/N with N>=3: {sum(1 for r in ran if r['teacher_n'] >= 3)}; N<3: {sum(1 for r in ran if r['teacher_n'] < 3)})")
    above = [r for r in ran if any(q["solved"] for q in r["proposals"])]
    L.append(f"  states where SOME frontier proposal solved (above the ceiling): {pct(len(above), len(ran))}")
    for n_min, name in ((3, "teacher 0/N, N>=3"), (1, "teacher 0/N, N<3 (yesterday's 0/1 + errored-heavy)")):
        g = [r for r in ran if (r["teacher_n"] >= 3) == (n_min == 3)]
        if g:
            L.append(f"    {name:26s}: {pct(sum(1 for r in g if any(q['solved'] for q in r['proposals'])), len(g))}")
    allp = [q for r in ran for q in r["proposals"] if q["solved"] is not None]
    L.append(f"  per proposal: solved {pct(sum(q['solved'] for q in allp), len(allp))}")
    if not allp:
        L.append("  (no F1 results yet)")
    for key in ("source", "relation", "a_type", "novel", "novel_loose"):
        c = collections.defaultdict(lambda: [0, 0])
        for q in allp:
            c[str(q[key])][0] += q["solved"]
            c[str(q[key])][1] += 1
        L.append(f"    by {key:8s}: " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(c.items())))
    for h in HARNESSES:
        g = [r for r in ran if r["harness"] == h]
        if g:
            L.append(f"    {h:20s}: states above ceiling {pct(sum(1 for r in g if any(q['solved'] for q in r['proposals'])), len(g))}")
    L.append("  per state:")
    for r in sorted(ran, key=lambda r: (-r["teacher_n"], r["harness"])):
        L.append(f"    {r['state_id'][:12]}…:{r['state_id'].split(':')[1]:>2s} {r['harness']:18s} {r['source']:16s} T {r['teacher']:4s}  " +
                 "  ".join(f"{q['arm']}[{q['source']},{q['relation']}/{q['a_type']}{',novel' if q['novel_loose'] else ',near-teacher' if q['novel'] else ',=teacher'}]="
                           f"{'?' if q['solved'] is None else 'SOLVED' if q['solved'] else 'failed'}" for q in r["proposals"]))

    # ------------------------------------------------------------------ cost / wall
    new = [r for r in results if r.get("started_at", "") >= "2026-09-21"]
    cost = sum(float(r.get("cost_usd") or 0) for r in new)
    wall = sum(float(r.get("wall_s") or 0) for r in new)
    prop_cost = sum(float(p.get("cost_usd") or 0) for p in proposals.values())
    starts = sorted(r["started_at"] for r in new if r.get("started_at"))
    L.append("")
    L.append(f"COST  today's continuations {len(new)}: ${cost:.2f} (Engy list price from trace usage), "
             f"proposal sampling + judge ${prop_cost:.2f}, total ${cost + prop_cost:.2f}; "
             f"container-time {wall / 3600:.1f} h (mean {wall / max(1, len(new)):.0f} s/continuation); "
             f"first start {starts[0] if starts else '-'}, last start {starts[-1] if starts else '-'}")
    J["cost"] = {"n_new": len(new), "continuations_usd": cost, "proposals_usd": prop_cost, "container_hours": wall / 3600}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".txt").write_text("\n".join(L) + "\n")
    Path(str(args.out) + ".json").write_text(json.dumps(J, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
