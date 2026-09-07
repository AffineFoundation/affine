"""Does agreement with a frontier model's ACTION track the benchmark where the
teacher score does not? (operator directive 2026-09-07 "Go. Run it.")

Follows research/results/frontier_rule_probe/report_modulation.txt: the
frontier-action modulator is cheap and adds no noise, but two live miners with
identical frontier agreement cannot tell whether agree_F is a capability
signal. Two panels of benched models can:

  rt7     the 29 live-board models of research/results/rt7_live_isomorphism.json
          (pre-reset era, records in affine/state.wvk9/evals, schema-1 corpus).
          On this panel Spearman(min-era score, swe) = -0.42: the teacher term
          inverts. 80 turns x 4 miner samples per record.
  reigns  the 9 post-reset kings (reign 0..8, affine/state/bench_history.jsonl),
          each crowned in a stored 1300-turn duel where the previous king is the
          other side. 150 bash turns per record, both sides scored.

For every selected turn the frontier (engy glm-5.2) is sampled N_SAMPLES times
at T=0.8 plus one greedy pass. Per model:

  agree_F   mean over turns/samples of best token-Jaccard(miner action, frontier actions)
  excess_F  agree_F minus the teacher refs' own agree_F on the same turns (removes
            turn difficulty; each model sits on a different slice)
  agree_T   same against the teacher refs (the thing min(R,G) implicitly rewards)

then Spearman with swe-rebench, permutation p. Actions only: no thoughts, no
logprobs, no GPU.

  python research/scripts/frontier_agree_bench.py select --panel rt7 reigns
  python research/scripts/frontier_agree_bench.py sample --record chal-00343 --era wvk9
  python research/scripts/frontier_agree_bench.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "scripts"))

import frontier_rule_probe as frp  # noqa: E402

OUT_DEFAULT = REPO / "research/results/frontier_agree_bench"
WVK9_EVALS = REPO / "affine/state.wvk9/evals"
RT7_JSON = REPO / "research/results/rt7_live_isomorphism.json"
BENCH_HISTORY = REPO / "affine/state/bench_history.jsonl"
SHARDS_DIR = Path("/tmp/rt7/shards")
N_SAMPLES = 2
ALBEDO_GENESIS = "dendriteholdings/albedo-qwen3.6-35b-king-genesis"
ALBEDO_GENESIS_SWE = 0.20  # AGENTS.md section 3b


def load_record(rec: str, era: str) -> dict:
    d = WVK9_EVALS if era == "wvk9" else frp.EVALS_DIR
    return json.load(gzip.open(d / f"{rec}.json.gz"))


# ------------------------------------------------------------------ select
def schema1_turns(manifest_sha: str) -> dict[str, dict]:
    """turn_id -> turn for a schema-1 (per-turn shard) manifest, from the shards
    mirrored at data.affine.io/turns/shards/ (downloaded to SHARDS_DIR)."""
    man = json.loads((SHARDS_DIR / f"man_{manifest_sha}.json").read_text())
    out: dict[str, dict] = {}
    for sh in man["shards"]:
        if not sh.get("active"):
            continue
        with gzip.open(SHARDS_DIR / Path(sh["key"]).name, "rt") as f:
            for line in f:
                t = json.loads(line)
                out[f"{t['traj_id']}:{t['turn_idx']}"] = t
    return out


def rt7_panel() -> dict[str, dict]:
    """repo(lower) -> {swe, label, s, margin} for the 29 RT-7 models (+ Albedo genesis)."""
    rows = json.load(open(RT7_JSON))["rows"]
    panel = {r["repo"].lower(): {"swe": r["swe"], "label": r["label"], "s": r["s"],
                                 "margin": r["margin"]} for r in rows}
    panel[ALBEDO_GENESIS.lower()] = {"swe": ALBEDO_GENESIS_SWE, "label": "albedo-genesis",
                                     "s": None, "margin": None}
    return panel


def reigns_panel() -> dict[str, dict]:
    """revision -> pooled swe (resolved/attempted over both bench suites) + label."""
    out: dict[str, dict] = {}
    for line in open(BENCH_HISTORY):
        r = json.loads(line)
        res = r.get("result") or {}
        if r.get("state") != "DONE" or not res.get("ok") or res.get("score") is None:
            continue
        e = out.setdefault(r["revision"], {"resolved": 0, "attempted": 0,
                                            "label": r["label"], "repo": r["repo"]})
        e["resolved"] += int(res["n_resolved"])
        e["attempted"] += int(res["n_instances"])
    for e in out.values():
        e["swe"] = e["resolved"] / e["attempted"]
    return out


def rt7_records() -> list[tuple[str, str]]:
    idx = [json.loads(l) for l in open(WVK9_EVALS / "index.jsonl")]
    panel = rt7_panel()
    seen, recs = set(), []
    for e in idx:  # index is chronological; first record per repo
        repo = e["repo"].lower()
        if repo in panel and repo not in seen and (WVK9_EVALS / f"{e['challenge_id']}.json.gz").exists():
            seen.add(repo)
            recs.append((e["challenge_id"], "wvk9"))
    return recs


def reigns_records() -> list[tuple[str, str]]:
    panel = reigns_panel()
    idx = [json.loads(l) for l in open(frp.EVALS_DIR / "index.jsonl")]
    recs = []
    for e in idx:
        if e["revision"] in panel and e.get("challenger_wins"):
            recs.append((e["challenge_id"], "live"))
    return recs


def cmd_select(args) -> int:
    out = Path(args.out)
    (out / "turns").mkdir(parents=True, exist_ok=True)
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    rng = random.Random(args.seed)
    cfg = frp.load_config()
    rt7 = rt7_panel()
    reigns = reigns_panel()
    todo: list[tuple[str, str, str]] = []
    if "rt7" in args.panel:
        todo += [(r, e, "rt7") for r, e in rt7_records()]
    if "reigns" in args.panel:
        todo += [(r, e, "reigns") for r, e in reigns_records()]
    for rec, era, panel in todo:
        dst = out / "turns" / f"{rec}.jsonl"
        if dst.exists() and not args.force:
            print(f"{rec}: already selected, skip")
            continue
        d = load_record(rec, era)
        req, v = d["request"], d["verdict"]
        refs = d["teacher_refs"]
        k_by = {r["turn_id"]: r for r in d["king_rows"]}
        c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
        tids = [t for t in d["turn_ids"] if refs.get(t)
                and frp.not_forfeit(k_by.get(t)) and frp.not_forfeit(c_by.get(t))]
        sha = v["slice"]["manifest_sha256"]
        if panel == "rt7":
            turns_by_id = schema1_turns(sha)
            picked = tids  # all 80
            mat = {t: {"prefix": turns_by_id[t]["prefix"], "action_kind": "bash"} for t in picked}
        else:
            scratch = Path(f"/tmp/frontier_probe_corpus/{sha[:12]}")
            corpus = None
            for key in (f"corpus/manifests/{sha}.json", f"turns/manifests/{sha}.json"):
                c = frp.CorpusSync(cfg.dataset.corpus_base_url, key, scratch, lazy_chunks=True)
                if not c.ready:
                    try:
                        c.refresh()
                    except Exception as e:  # noqa: BLE001 — try the other location
                        print(f"  {key}: {e!r}")
                if c.ready:
                    corpus = c
                    break
            if corpus is None:
                raise SystemExit(f"{rec}: manifest {sha[:12]} not syncable")
            rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
            pool = sorted(t for t in tids if rows[t].get("action_kind", "bash") == "bash")
            rng.shuffle(pool)
            picked = pool[: args.n_per]
            mats = corpus.materialize_turns([rows[t] for t in picked])
            mat = {t: m for t, m in zip(picked, mats)}
        sides = {
            "challenger": {"repo": req["challenger_repo"], "revision": req["challenger_revision"]},
            "king": {"repo": req["king_repo"], "revision": req["king_revision"]},
        }
        for s in sides.values():
            b = rt7.get(s["repo"].lower()) if panel == "rt7" else reigns.get(s["revision"])
            s["swe"] = b["swe"] if b else None
            s["label"] = b["label"] if b else None
        with open(dst, "w") as f:
            for t in picked:
                f.write(json.dumps({
                    "record": rec, "turn_id": t, "kind": mat[t].get("action_kind", "bash"),
                    "prefix": mat[t]["prefix"],
                    "refs_y": [r["y"] for r in refs[t]],
                    "king_y": [p["y_a"] for p in k_by[t]["pairs"]],
                    "challenger_y": [p["y_a"] for p in c_by[t]["pairs"]],
                }) + "\n")
        meta[rec] = {"panel": panel, "era": era, "sides": sides, "n_turns": len(picked),
                     "margin": v["margin"], "z": v["z"], "manifest_sha256": sha}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{rec}: {len(picked)} turns  chal={sides['challenger']['label']} swe={sides['challenger']['swe']}"
              f"  king={sides['king']['label']} swe={sides['king']['swe']}")
    return 0


# ------------------------------------------------------------------ sample
async def cmd_sample(args) -> int:
    out = Path(args.out)
    turns = frp.read_jsonl(out / "turns" / f"{args.record}.jsonl")
    if not turns:
        raise SystemExit(f"no turns for {args.record}; run select first")
    dst = out / "frontier" / f"{args.record}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = {r["turn_id"] for r in frp.read_jsonl(dst) if not r.get("error")}
    todo = [t for t in turns if t["turn_id"] not in done]
    key = os.environ.get(args.engy_env, "")
    if not key:
        raise SystemExit(f"{args.engy_env} not set (source .env)")
    engy = frp.Engy(key, args.concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0
    print(f"{args.record}: sampling {len(todo)} turns ({len(done)} already done)")

    async def one(t: dict) -> None:
        nonlocal n_done
        try:
            *sampled, greedy = await asyncio.gather(
                *[engy.chat(t["prefix"], frp.FRONTIER_TEMP) for _ in range(N_SAMPLES)],
                engy.chat(t["prefix"], 0.0))
            rec = {"turn_id": t["turn_id"], "record": args.record, "kind": t["kind"],
                   "samples": [frp.choice_to_rollout(c, t["kind"]) for c in sampled],
                   "greedy": frp.choice_to_rollout(greedy, t["kind"]), "error": None}
        except Exception as e:  # noqa: BLE001 — one bad turn must not kill the run
            rec = {"turn_id": t["turn_id"], "record": args.record, "kind": t["kind"],
                   "samples": [], "greedy": None, "error": repr(e)[:300]}
        async with lock:
            frp.append_jsonl(dst, rec)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(todo)} ({time.time() - t0:.0f}s) usage={engy.usage}", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    rows = [r for r in frp.read_jsonl(dst) if not r.get("error")]
    parsed = st.mean(sum(1 for s in r["samples"] if s["parsed"]) for r in rows) if rows else 0
    print(f"{args.record}: {len(rows)} turns ok in {time.time() - t0:.0f}s; usage={engy.usage}; "
          f"parsed of {N_SAMPLES}: {parsed:.2f}")
    if len(rows) >= len(turns):
        dst.with_suffix(".done").write_text(json.dumps(engy.usage))
    return 0


# ------------------------------------------------------------------ analyze
def permutation_p(a: list[float], b: list[float], observed: float, trials: int = 20000) -> float:
    rng = random.Random(0)
    sh = list(b)
    hits = 0
    for _ in range(trials):
        rng.shuffle(sh)
        if abs(frp.spearman(a, sh)) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (trials + 1)


def cmd_analyze(args) -> int:
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    # model key -> accumulated per-turn stats (a model can appear in several records)
    models: dict[str, dict] = {}
    for rec, m in meta.items():
        fr = {}
        for r in frp.read_jsonl(out / "frontier" / f"{rec}.jsonl"):
            if not r.get("error"):
                fr[r["turn_id"]] = r
        for t in frp.read_jsonl(out / "turns" / f"{rec}.jsonl"):
            f = fr.get(t["turn_id"])
            if not f:
                continue
            yF = [s["y"] for s in f["samples"] if s["parsed"]]
            yG = f["greedy"]["y"] if f.get("greedy") and f["greedy"]["parsed"] else None
            if not yF:
                continue
            teacher_F = st.mean(frp.agree(y, yF) for y in t["refs_y"])
            teacher_G = st.mean(frp.jaccard(y, yG) for y in t["refs_y"]) if yG else None
            for side in ("challenger", "king"):
                s = m["sides"][side]
                key = s["revision"] if m["panel"] == "reigns" else s["repo"].lower()
                e = models.setdefault(key, {"panel": m["panel"], "label": s["label"], "repo": s["repo"],
                                            "swe": s["swe"], "records": set(), "aF": [], "aG": [],
                                            "aT": [], "xF": [], "xG": [], "exactF": []})
                e["records"].add(rec)
                ys = [y for y in t[f"{side}_y"] if y]
                if not ys:
                    continue
                aF = st.mean(frp.agree(y, yF) for y in ys)
                aT = st.mean(frp.agree(y, t["refs_y"]) for y in ys)
                e["aF"].append(aF)
                e["aT"].append(aT)
                e["xF"].append(aF - teacher_F)
                e["exactF"].append(st.mean(any(y.strip() == x.strip() for x in yF) for y in ys))
                if yG:
                    aG = st.mean(frp.jaccard(y, yG) for y in ys)
                    e["aG"].append(aG)
                    e["xG"].append(aG - teacher_G)
    L: list[str] = []
    P = L.append
    P("Frontier-action agreement vs swe-rebench — per benched model (actions only; engy glm-5.2, "
      f"{N_SAMPLES} samples @T=0.8 + greedy per turn)")
    report: dict = {}
    for panel in ("rt7", "reigns"):
        rows = [dict(e, key=k) for k, e in models.items()
                if e["panel"] == panel and e["swe"] is not None and len(e["aF"]) >= 20]
        P("")
        P(f"===== panel {panel}: {len(rows)} benched models =====")
        if len(rows) < 4:
            P("  too few"); continue
        P(f"{'label':14} {'swe':>5} {'turns':>5} {'agree_F':>8} {'excess_F':>9} {'agree_G':>8} {'excess_G':>9} {'agree_T':>8} {'exact_F':>8}")
        summ = []
        for e in sorted(rows, key=lambda e: -e["swe"]):
            row = {"label": e["label"], "repo": e["repo"], "swe": e["swe"], "n": len(e["aF"]),
                   "agree_F": st.mean(e["aF"]), "excess_F": st.mean(e["xF"]),
                   "agree_G": st.mean(e["aG"]) if e["aG"] else float("nan"),
                   "excess_G": st.mean(e["xG"]) if e["xG"] else float("nan"),
                   "agree_T": st.mean(e["aT"]), "exact_F": st.mean(e["exactF"]),
                   "records": sorted(e["records"])}
            summ.append(row)
            P(f"{row['label'][:14]:14} {row['swe']:5.2f} {row['n']:5d} {row['agree_F']:8.3f} {row['excess_F']:+9.3f} "
              f"{row['agree_G']:8.3f} {row['excess_G']:+9.3f} {row['agree_T']:8.3f} {row['exact_F']:8.3f}")
        swe = [r["swe"] for r in summ]
        P("")
        P(f"{'metric':10} {'Spearman vs swe':>15} {'perm p':>8}")
        cors = {}
        for met in ("agree_F", "excess_F", "agree_G", "excess_G", "agree_T", "exact_F"):
            xs = [r[met] for r in summ]
            if any(x != x for x in xs):
                continue
            rho = frp.spearman(xs, swe)
            p = permutation_p(xs, swe, rho)
            cors[met] = {"rho": rho, "p": p}
            P(f"{met:10} {rho:+15.3f} {p:8.4f}")
        # the teacher-side comparator on the same models
        if panel == "rt7":
            rt = rt7_panel()
            sub = [(rt[r["repo"].lower()]["s"], r["swe"]) for r in summ
                   if r["repo"].lower() in rt and rt[r["repo"].lower()]["s"] is not None]
            if len(sub) >= 4:
                rho = frp.spearman([a for a, _ in sub], [b for _, b in sub])
                cors["S_v2_era"] = {"rho": rho, "p": permutation_p([a for a, _ in sub], [b for _, b in sub], rho)}
                P(f"{'S (v2 era)':10} {rho:+15.3f} {cors['S_v2_era']['p']:8.4f}   <- the teacher-side score on the same models")
        # agree_F − agree_T: does siding with the frontier over the teacher track the bench?
        xs = [r["agree_F"] - r["agree_T"] for r in summ]
        rho = frp.spearman(xs, swe)
        cors["F_minus_T"] = {"rho": rho, "p": permutation_p(xs, swe, rho)}
        P(f"{'F − T':10} {rho:+15.3f} {cors['F_minus_T']['p']:8.4f}")
        report[panel] = {"models": summ, "correlations": cors}
    text = "\n".join(L) + "\n"
    (out / "report.txt").write_text(text)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--panel", nargs="+", default=["rt7", "reigns"])
    s.add_argument("--n-per", type=int, default=150)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("--record", required=True)
    s.add_argument("--engy-env", default="ENGY_2")
    s.add_argument("--concurrency", type=int, default=24)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "select":
        return cmd_select(args)
    if args.cmd == "sample":
        return asyncio.run(cmd_sample(args))
    return cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
