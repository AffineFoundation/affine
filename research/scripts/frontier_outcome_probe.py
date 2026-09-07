"""Tier-0 probes A and B (2026-09-07): does frontier disagreement mark the
teacher's mistakes?

Premise of the frontier-action regulariser: the frontier points further than
the teacher. Two checks that need no GPU:

  corpus   A/B on D. Every post-T0 duel turn already frontier-sampled
           (research/results/frontier_rule_probe, frontier_agree_bench/chal-00283)
           is joined to its rollout's outcome from the trace envelope
           (trace.rewards: solved / correct / passed_fraction). D is
           teacher-generated, so the trajectory's own next action ("gold") is
           a teacher action, and the k=3 duel refs are teacher resamples of
           the same turn. Question: on FAILED rollouts, does the frontier
           disagree with gold more than the teacher's own resamples do?
           d = agree(gold, frontier) − agree(gold, teacher refs), turn and
           rollout level, AUC for solved.

  bench    B on the benchmark itself. affine/state/benches/*.json.gz hold the
           9 reign kings' on-policy swe-rebench transcripts (25 tasks each, x2
           suites) with per-task `resolved`. Frontier sampled on 4 turns per
           trajectory. Questions: does frontier agreement with the king's
           action predict task resolution (within model), and does a model's
           mean agreement track its bench score on the SAME 25 tasks.

  python research/scripts/frontier_outcome_probe.py corpus
  python research/scripts/frontier_outcome_probe.py bench-select
  python research/scripts/frontier_agree_bench.py --out research/results/frontier_outcome_probe sample --record <run>
  python research/scripts/frontier_outcome_probe.py bench-analyze
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "scripts"))
sys.path.insert(0, str(REPO / "affine"))

import frontier_agree_bench as fab  # noqa: E402
import frontier_rule_probe as frp  # noqa: E402
from affine.corpus.materialize import materialize_turn  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402

OUT = REPO / "research/results/frontier_outcome_probe"
REWARDS = Path("/tmp/traces/rewards.json")
VIEW_CACHES = Path("/tmp/frontier_probe_corpus")
BENCH_DIR = REPO / "affine/state/benches"
TURNS_PER_TRAJ = 4
MAX_PREFIX_CHARS = 160_000

GROUP = {"swesmith": "coding", "multiswe": "coding", "scaleswe": "coding", "swelego": "coding",
         "swerebench_v2": "coding", "r2e_gym": "coding", "terminal_lego": "terminal",
         "terminal_bench_2": "terminal", "affine_math": "math", "affine_wiki": "tool_use",
         "nl2repobench": "nl2repo"}


def solved_of(rewards: dict) -> float | None:
    for k in ("solved", "correct", "passed_fraction"):
        if rewards.get(k) is not None:
            return float(rewards[k])
    return None


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score_pos > score_neg) with ties at 1/2 (Mann–Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def auc_ci(pos: list[float], neg: list[float], n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        p = [rng.choice(pos) for _ in pos]
        n = [rng.choice(neg) for _ in neg]
        vals.append(auc(p, n))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


# ------------------------------------------------------------------ corpus
def view_records() -> dict[str, tuple[dict, dict]]:
    """turn_id -> (traj record, turn meta) over every cached schema-3 view chunk."""
    out: dict[str, tuple[dict, dict]] = {}
    for f in glob.glob(str(VIEW_CACHES / "*" / "chunks" / "view_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                for meta in r["turns"]:
                    out[f"{r['traj_id']}:{meta['turn_idx']}"] = (r, meta)
    return out


def cmd_corpus(args) -> int:
    rewards = json.loads(REWARDS.read_text())
    views = view_records()
    sources = [(frp.OUT_DEFAULT, rec) for rec in ("chal-00286", "chal-00287", "chal-00288", "chal-00289")]
    sources.append((fab.OUT_DEFAULT, "chal-00283"))
    rows: list[dict] = []
    miss = {"view": 0, "reward": 0, "gold": 0, "frontier": 0}
    for base, rec in sources:
        fr = {r["turn_id"]: r for r in frp.read_jsonl(base / "frontier" / f"{rec}.jsonl") if not r.get("error")}
        for t in frp.read_jsonl(base / "turns" / f"{rec}.jsonl"):
            tid = t["turn_id"]
            if tid not in views:
                miss["view"] += 1
                continue
            traj, meta = views[tid]
            rw = rewards.get(traj["rollout_id"])
            score = solved_of(rw["rewards"]) if rw else None
            if score is None:
                miss["reward"] += 1
                continue
            kind = t["kind"]
            _, gold = split_rollout(materialize_turn(traj, meta)["reference_turn"], kind)
            if not gold:
                miss["gold"] += 1
                continue
            f = fr.get(tid)
            if not f:
                miss["frontier"] += 1
                continue
            yF = [s["y"] for s in f["samples"] if s["parsed"]]
            if not yF:
                miss["frontier"] += 1
                continue
            yG = f["greedy"]["y"] if f.get("greedy") and f["greedy"]["parsed"] else None
            refs_y = [r["y"] for r in t["refs"]] if "refs" in t else t["refs_y"]
            king_y = [p["y_a"] for p in t["king"]] if "king" in t else t["king_y"]
            chal_y = [p["y_a"] for p in t["challenger"]] if "challenger" in t else t["challenger_y"]
            aT_gold = st.mean(frp.jaccard(gold, y) for y in refs_y)
            aF_gold = st.mean(frp.jaccard(gold, y) for y in yF)
            rows.append({
                "record": rec, "turn_id": tid, "rollout_id": traj["rollout_id"], "kind": kind,
                "source": traj.get("source") or t.get("source"),
                "group": GROUP.get(traj.get("source") or t.get("source") or "", "other"),
                "phase": meta.get("phase"), "score": score, "solved": score >= 0.5,
                "aT_gold": aT_gold, "aF_gold": aF_gold, "d": aF_gold - aT_gold,
                "aG_gold": frp.jaccard(gold, yG) if yG else None,
                "aT_gold_max": max(frp.jaccard(gold, y) for y in refs_y),
                "aF_gold_max": max(frp.jaccard(gold, y) for y in yF),
                "aFT": st.mean(frp.agree(y, yF) for y in refs_y),       # teacher refs vs frontier
                "king_aF": st.mean(frp.agree(y, yF) for y in king_y if y) if any(king_y) else None,
                "chal_aF": st.mean(frp.agree(y, yF) for y in chal_y if y) if any(chal_y) else None,
                "king_aT": st.mean(frp.agree(y, refs_y) for y in king_y if y) if any(king_y) else None,
            })
    L: list[str] = []
    P = L.append
    P(f"A/B on D: {len(rows)} frontier-sampled turns joined to rollout outcomes (missing: {miss})")
    P("gold = the rollout's own next action (teacher-generated); refs = k teacher resamples of the turn; F = 2-3 frontier samples")
    P("aX_gold = mean Jaccard(gold, X); d = aF_gold − aT_gold (frontier vs teacher-resample closeness to gold)")
    report: dict = {"n": len(rows), "missing": miss, "groups": {}}

    def block(sub: list[dict], title: str) -> None:
        n = len(sub)
        P("")
        P(f"===== {title} (n={n} turns, {len({r['rollout_id'] for r in sub})} rollouts) =====")
        if n < 20:
            P("  too few"); return
        pos = [r for r in sub if r["solved"]]
        neg = [r for r in sub if not r["solved"]]
        P(f"solved {len(pos)} turns / failed {len(neg)} turns  (mean score {st.mean(r['score'] for r in sub):.3f})")
        P(f"{'':18} {'solved':>8} {'failed':>8} {'diff':>8}")
        g: dict = {"n": n, "n_solved": len(pos), "n_failed": len(neg), "means": {}}
        for met in ("aT_gold", "aF_gold", "aG_gold", "d", "aFT", "aT_gold_max", "aF_gold_max", "king_aF", "chal_aF", "king_aT"):
            ps = [r[met] for r in pos if r[met] is not None]
            ns = [r[met] for r in neg if r[met] is not None]
            if not ps or not ns:
                continue
            g["means"][met] = {"solved": st.mean(ps), "failed": st.mean(ns)}
            P(f"{met:18} {st.mean(ps):8.3f} {st.mean(ns):8.3f} {st.mean(ps) - st.mean(ns):+8.3f}")
        P("-- AUC(solved | feature), turn level; >0.5 = higher feature on solved turns --")
        g["auc_turn"] = {}
        for met in ("aT_gold", "aF_gold", "aG_gold", "d", "aFT"):
            ps = [r[met] for r in pos if r[met] is not None]
            ns = [r[met] for r in neg if r[met] is not None]
            if len(ps) < 5 or len(ns) < 5:
                continue
            a = auc(ps, ns)
            lo, hi = auc_ci(ps, ns)
            g["auc_turn"][met] = {"auc": a, "ci": [lo, hi]}
            P(f"  {met:12} AUC {a:.3f}  [{lo:.3f}, {hi:.3f}]")
        # rollout level: mean feature over the rollout's sampled turns
        by_roll: dict[str, list[dict]] = {}
        for r in sub:
            by_roll.setdefault(r["rollout_id"], []).append(r)
        P(f"-- AUC at rollout level (mean over the rollout's turns; {len(by_roll)} rollouts) --")
        g["auc_rollout"] = {}
        for met in ("aT_gold", "aF_gold", "d", "aFT"):
            ps = [st.mean(x[met] for x in rs) for rs in by_roll.values() if rs[0]["solved"]]
            ns = [st.mean(x[met] for x in rs) for rs in by_roll.values() if not rs[0]["solved"]]
            if len(ps) < 5 or len(ns) < 5:
                continue
            a = auc(ps, ns)
            lo, hi = auc_ci(ps, ns)
            g["auc_rollout"][met] = {"auc": a, "ci": [lo, hi], "n_pos": len(ps), "n_neg": len(ns)}
            P(f"  {met:12} AUC {a:.3f}  [{lo:.3f}, {hi:.3f}]  (solved {len(ps)} / failed {len(ns)})")
        # where the frontier breaks from gold but the teacher does not
        brk = [r for r in sub if r["aF_gold"] < 0.3 <= r["aT_gold"]]
        agr = [r for r in sub if r["aF_gold"] >= 0.5 and r["aT_gold"] >= 0.5]
        if brk and agr:
            P(f"-- frontier breaks from gold while teacher resamples keep it (aF<0.3<=aT): {len(brk)} turns, "
              f"solved rate {st.mean(r['solved'] for r in brk):.2f} vs {st.mean(r['solved'] for r in agr):.2f} where both agree (n={len(agr)})")
            g["break"] = {"n": len(brk), "solved_rate": st.mean(r["solved"] for r in brk),
                          "n_agree": len(agr), "solved_rate_agree": st.mean(r["solved"] for r in agr)}
        report["groups"][title] = g

    block(rows, "ALL")
    for grp in sorted({r["group"] for r in rows}):
        block([r for r in rows if r["group"] == grp], f"group {grp}")
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        if len(sub) >= 80:
            block(sub, f"source {src}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "report_corpus.txt").write_text("\n".join(L) + "\n")
    (OUT / "report_corpus.json").write_text(json.dumps(report, indent=1))
    print("\n".join(L))
    return 0


# ------------------------------------------------------------------ bench
def bench_runs() -> list[dict]:
    panel = fab.reigns_panel()
    runs = []
    for f in sorted(BENCH_DIR.glob("bench-*.json.gz")):
        d = json.load(gzip.open(f))
        rev = d["request"]["revision"]
        if rev not in panel or not (d.get("result") or {}).get("ok"):
            continue
        runs.append({"run": f.name.split(".")[0], "revision": rev, "label": panel[rev]["label"],
                     "swe": panel[rev]["swe"], "suite": d["request"]["suite"], "instances": d["instances"]})
    return runs


def cmd_bench_select(args) -> int:
    (OUT / "turns").mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}
    total_chars = 0
    n_turns = 0
    for run in bench_runs():
        dst = OUT / "turns" / f"{run['run']}.jsonl"
        recs = []
        for inst, v in run["instances"].items():
            msgs = v["messages"]
            cand = []
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                _, y = split_rollout(m["content"], "bash")
                if y and sum(len(x["content"]) for x in msgs[:i]) <= MAX_PREFIX_CHARS:
                    cand.append((i, y))
            if not cand:
                continue
            qs = [(j + 0.5) / TURNS_PER_TRAJ for j in range(TURNS_PER_TRAJ)]
            picks = sorted({cand[min(len(cand) - 1, int(q * len(cand)))] for q in qs})
            for i, y in picks:
                prefix = [{"role": m["role"], "content": m["content"]} for m in msgs[:i]]
                total_chars += sum(len(m["content"]) for m in prefix)
                recs.append({"record": run["run"], "turn_id": f"{inst}:{i}", "kind": "bash",
                             "instance": inst, "msg_idx": i, "n_assistant": len(cand),
                             "pos": [c[0] for c in cand].index(i) / max(len(cand) - 1, 1),
                             "resolved": bool(v["resolved"]), "exit_status": v.get("exit_status"),
                             "prefix": prefix, "king_y": [y]})
        dst.write_text("".join(json.dumps(r) + "\n" for r in recs))
        n_turns += len(recs)
        meta[run["run"]] = {"label": run["label"], "revision": run["revision"], "swe": run["swe"],
                            "suite": run["suite"], "n_turns": len(recs),
                            "n_resolved": sum(v["resolved"] for v in run["instances"].values()),
                            "n_instances": len(run["instances"])}
        print(f"{run['run']}: {run['label']} {run['suite']} {len(recs)} turns, "
              f"resolved {meta[run['run']]['n_resolved']}/{meta[run['run']]['n_instances']}")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"total {n_turns} turns, ~{total_chars / 4e6:.0f}M prompt tokens per call")
    return 0


def cmd_bench_analyze(args) -> int:
    meta = json.loads((OUT / "meta.json").read_text())
    rows: list[dict] = []
    for run, m in meta.items():
        fr = {r["turn_id"]: r for r in frp.read_jsonl(OUT / "frontier" / f"{run}.jsonl") if not r.get("error")}
        for t in frp.read_jsonl(OUT / "turns" / f"{run}.jsonl"):
            f = fr.get(t["turn_id"])
            if not f:
                continue
            yF = [s["y"] for s in f["samples"] if s["parsed"]]
            if not yF:
                continue
            yG = f["greedy"]["y"] if f.get("greedy") and f["greedy"]["parsed"] else None
            y = t["king_y"][0]
            rows.append({"run": run, "label": m["label"], "swe": m["swe"], "instance": t["instance"],
                         "resolved": t["resolved"], "pos": t["pos"], "aF": frp.agree(y, yF),
                         "aG": frp.jaccard(y, yG) if yG else None,
                         "exact": any(y.strip() == x.strip() for x in yF),
                         "fself": frp.agree(yF[0], yF[1:]) if len(yF) > 1 else None})
    L: list[str] = []
    P = L.append
    P(f"B on the bench: {len(rows)} king turns from {len({(r['run'], r['instance']) for r in rows})} on-policy "
      f"swe-rebench trajectories, frontier glm-5.2 sampled per turn")
    pos = [r for r in rows if r["resolved"]]
    neg = [r for r in rows if not r["resolved"]]
    P(f"resolved turns {len(pos)} / unresolved {len(neg)}")
    P(f"{'':8} {'resolved':>9} {'unresolved':>11}")
    for met in ("aF", "aG", "exact"):
        ps = [float(r[met]) for r in pos if r[met] is not None]
        ns = [float(r[met]) for r in neg if r[met] is not None]
        P(f"{met:8} {st.mean(ps):9.3f} {st.mean(ns):11.3f}   turn AUC {auc(ps, ns):.3f}")
    P("-- by position in trajectory (quartile) --")
    for q in range(4):
        sub = [r for r in rows if q / 4 <= r["pos"] < (q + 1) / 4 or (q == 3 and r["pos"] == 1.0)]
        ps = [r["aF"] for r in sub if r["resolved"]]
        ns = [r["aF"] for r in sub if not r["resolved"]]
        if ps and ns:
            P(f"  Q{q + 1}: resolved {st.mean(ps):.3f} unresolved {st.mean(ns):.3f}  AUC {auc(ps, ns):.3f}  (n={len(sub)})")
    # trajectory level, within model (rank within run to remove the model effect)
    by_traj: dict[tuple, list[dict]] = {}
    for r in rows:
        by_traj.setdefault((r["run"], r["instance"]), []).append(r)
    traj = [{"run": k[0], "label": rs[0]["label"], "resolved": rs[0]["resolved"],
             "aF": st.mean(r["aF"] for r in rs)} for k, rs in by_traj.items()]
    ps = [t["aF"] for t in traj if t["resolved"]]
    ns = [t["aF"] for t in traj if not t["resolved"]]
    lo, hi = auc_ci(ps, ns)
    P(f"-- trajectory level: AUC(resolved | mean aF) pooled {auc(ps, ns):.3f} [{lo:.3f}, {hi:.3f}] "
      f"(resolved {len(ps)} / unresolved {len(ns)})")
    within = []
    for run in meta:
        ts = [t for t in traj if t["run"] == run]
        p = [t["aF"] for t in ts if t["resolved"]]
        n = [t["aF"] for t in ts if not t["resolved"]]
        if p and n:
            within.append(auc(p, n))
    if within:
        P(f"   within-run AUCs: mean {st.mean(within):.3f}  ({', '.join(f'{a:.2f}' for a in within)})")
    # model level: mean aF over both suites vs pooled swe, same 25 tasks for everyone
    P("-- model level (same 25 tasks for every model) --")
    by_model: dict[str, dict] = {}
    for r in rows:
        e = by_model.setdefault(r["label"], {"swe": r["swe"], "aF": [], "aG": []})
        e["aF"].append(r["aF"])
        if r["aG"] is not None:
            e["aG"].append(r["aG"])
    P(f"{'model':10} {'swe':>6} {'turns':>6} {'agree_F':>8} {'agree_G':>8}")
    ms = sorted(by_model.items(), key=lambda kv: -kv[1]["swe"])
    for lab, e in ms:
        P(f"{lab:10} {e['swe']:6.2f} {len(e['aF']):6d} {st.mean(e['aF']):8.3f} {st.mean(e['aG']):8.3f}")
    swe = [e["swe"] for _, e in ms]
    for met in ("aF", "aG"):
        xs = [st.mean(e[met]) for _, e in ms]
        rho = frp.spearman(xs, swe)
        P(f"Spearman(mean {met}, swe) = {rho:+.3f}  perm p {fab.permutation_p(xs, swe, rho):.3f}")
    text = "\n".join(L) + "\n"
    (OUT / "report_bench.txt").write_text(text)
    (OUT / "bench_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus")
    sub.add_parser("bench-select")
    sub.add_parser("bench-analyze")
    args = ap.parse_args()
    return {"corpus": cmd_corpus, "bench-select": cmd_bench_select,
            "bench-analyze": cmd_bench_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
