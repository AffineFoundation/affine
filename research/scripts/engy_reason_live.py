"""Engy-teacher Reason on the LIVE SN120 panel (the RT-7 falsifier).

RT-7: under the production teacher (GLM-4.5-Air) the live board inverts —
Spearman(duel margin, swe_lite) = -0.42. This probe rescores the *same stored
thoughts* from published duel artifacts with engy `glm-5.2` (a teacher nobody
optimized against) and asks whether the inversion disappears.

Every artifact contributes two panel models (challenger + king on the same
slice), so we get an unpaired Reason per model and a difficulty-controlled
paired margin per artifact.

Stages (run from research/, venv active, ENGY exported):
  python scripts/engy_reason_live.py panel     # select models, pull artifacts
  python scripts/engy_reason_live.py refs      # sample k teacher refs per turn
  python scripts/engy_reason_live.py score     # teacher-forced lp calls
  python scripts/engy_reason_live.py analyze   # Spearman table
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engy_reason_probe import (  # noqa: E402
    Engy, TEACHER, K_REFS, TAU, RESULTS, BENCH_URL, EVALS_URL,
    load_done, append_jsonl, turn_score, spearman, fetch_json)
from harness.chat import gen_prompt, force_text, split_rollout  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "affine"))
from evalsrv.corpus import CorpusSync  # noqa: E402

SEED = 120
TURNS_PER_MODEL = 30
MAX_ZERO_SWE = 10
CORPUS_BASE = "https://s3.hippius.com/affine-sn120"
MANIFEST_KEY = "turns/manifest.json"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "live_corpus"

PANEL_PATH = RESULTS / "engy_live_panel.json"
THOUGHTS_PATH = RESULTS / "engy_live_thoughts.jsonl"
REFS_PATH = RESULTS / "engy_refs_live.jsonl"
SCORES_PATH = RESULTS / "engy_scores_live.jsonl"


# ------------------------------------------------------------------ artifacts
def fetch_artifact(chal_id: str) -> dict:
    with urllib.request.urlopen(f"{EVALS_URL}/{chal_id}.json.gz",
                                timeout=300) as r:
        return json.loads(gzip.decompress(r.read()))


def published_chals() -> dict[str, dict]:
    """chal_id -> index meta for every published artifact."""
    with urllib.request.urlopen(f"{EVALS_URL}/index.jsonl", timeout=120) as r:
        lines = r.read().decode().splitlines()
    out = {}
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("challenge_id"):
            out[rec["challenge_id"]] = rec
    return out


def swe_models() -> dict[str, dict]:
    bench = fetch_json(BENCH_URL)
    out = {}
    for m in bench["models"]:
        s = m["suites"].get("swe_rebench_lite", {}).get("score")
        if s is not None:
            out[m["model_repo"].lower()] = {
                "repo": m["model_repo"], "swe": s, "label": m.get("label")}
    return out


def build_panel() -> None:
    """Pick panel models, download their artifacts, extract thoughts."""
    swe = swe_models()
    chals = published_chals()
    rng = random.Random(SEED)

    entries = []                 # candidate (repo_lc, chal_id, role)
    for repo_lc, m in swe.items():
        lab = m["label"] or ""
        if lab.startswith("chal-") and lab in chals:
            entries.append((repo_lc, lab, "challenger"))
    # kings: resolve via each candidate artifact's request.king_repo lazily
    # (handled while downloading below — a king with a swe score and no slot
    # yet joins the panel from the same artifact).

    pos = [e for e in entries if swe[e[0]]["swe"] > 0]
    zero = [e for e in entries if swe[e[0]]["swe"] == 0]
    rng.shuffle(zero)
    chosen = pos + zero[:MAX_ZERO_SWE]
    print(f"panel candidates: {len(pos)} swe>0, {len(zero)} zeros "
          f"(keeping {min(len(zero), MAX_ZERO_SWE)})")

    panel, seen_repos = [], set()
    thoughts_f = open(THOUGHTS_PATH, "w")

    def extract(chal_id: str, role: str, rows: list[dict],
                repo: str, sw: float, paired_rows: list[dict]):
        by = {r["turn_id"]: r for r in rows
              if r.get("valid") and r.get("pairs")
              and r["pairs"][0].get("z_a", "").strip()}
        other = {r["turn_id"] for r in paired_rows
                 if r.get("valid") and r.get("pairs")}
        tids = sorted(set(by) & other)
        rng2 = random.Random(f"{SEED}:{chal_id}")
        rng2.shuffle(tids)
        tids = tids[:TURNS_PER_MODEL]
        for tid in tids:
            thoughts_f.write(json.dumps({
                "chal_id": chal_id, "role": role, "repo": repo,
                "turn_id": tid, "z": by[tid]["pairs"][0]["z_a"]}) + "\n")
        return len(tids)

    for repo_lc, chal_id, _role in chosen:
        if repo_lc in seen_repos:
            continue
        try:
            art = fetch_artifact(chal_id)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {chal_id}: {e}")
            continue
        req = art.get("request", {})
        c_repo = req.get("challenger_repo", swe[repo_lc]["repo"])
        n = extract(chal_id, "challenger", art["challenger_rows"],
                    c_repo, swe[repo_lc]["swe"], art["king_rows"])
        panel.append({"repo": c_repo, "swe": swe[repo_lc]["swe"],
                      "chal_id": chal_id, "role": "challenger", "n_turns": n})
        seen_repos.add(repo_lc)
        k_repo = (req.get("king_repo") or "").lower()
        if k_repo in swe and k_repo not in seen_repos:
            n = extract(chal_id, "king", art["king_rows"],
                        req["king_repo"], swe[k_repo]["swe"],
                        art["challenger_rows"])
            panel.append({"repo": req["king_repo"], "swe": swe[k_repo]["swe"],
                          "chal_id": chal_id, "role": "king", "n_turns": n})
            seen_repos.add(k_repo)
        print(f"  {chal_id}: challenger={c_repo} "
              f"king={req.get('king_repo')} panel={len(panel)}")

    thoughts_f.close()
    PANEL_PATH.write_text(json.dumps(panel, indent=1))
    print(f"panel: {len(panel)} models -> {PANEL_PATH}")


# ------------------------------------------------------------------- prefixes
class LivePrefixes:
    """Materialize live-corpus turn prefixes on demand (lazy chunk fetch)."""

    def __init__(self):
        self.sync = CorpusSync(CORPUS_BASE, MANIFEST_KEY, CORPUS_DIR,
                               lazy_chunks=True)
        if not self.sync.ready:
            self.sync.refresh()
        rows = self.sync.load_index_rows()
        self.by_tid = {f"{r['traj_id']}:{r['turn_idx']}": r for r in rows}
        self.cache: dict[str, list[dict]] = {}

    def get(self, tid: str) -> list[dict] | None:
        if tid in self.cache:
            return self.cache[tid]
        row = self.by_tid.get(tid)
        if row is None:
            return None
        try:
            turn = self.sync.materialize_turns([row])[0]
        except Exception:  # noqa: BLE001
            return None
        self.cache[tid] = turn["prefix"]
        return turn["prefix"]

    def prefetch(self, tids: list[str], workers: int = 16) -> None:
        """Parallel chunk downloads, then chunk-grouped materialization
        (the sync's parsed-line cache only holds 4 chunks at a time)."""
        rows = [(t, self.by_tid[t]) for t in tids
                if t in self.by_tid and t not in self.cache]
        keys = sorted({r["chunk_key"] for _, r in rows})
        print(f"prefetch: {len(rows)} turns across {len(keys)} chunks")

        def fetch(key: str):
            try:
                self.sync.ensure_chunk(key)
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {key} failed: {e}")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fetch, keys))
        for tid, row in sorted(rows, key=lambda x: x[1]["chunk_key"]):
            self.get(tid)
        print(f"prefetch done: {sum(1 for t in tids if t in self.cache)}"
              f"/{len(tids)} prefixes available")


def load_thoughts() -> list[dict]:
    return [json.loads(l) for l in open(THOUGHTS_PATH)]


# --------------------------------------------------------------------- stages
async def stage_refs(engy: Engy):
    thoughts = load_thoughts()
    tids = sorted({t["turn_id"] for t in thoughts})
    pre = LivePrefixes()
    done = load_done(REFS_PATH, ("turn_id", "ref_idx"))
    todo = [(tid, i) for tid in tids for i in range(K_REFS)
            if (tid, i) not in done]
    print(f"refs: {len(tids)} turns, {len(done)} cached, {len(todo)} to do")
    lock = asyncio.Lock()
    n_done = n_fail = 0

    async def one(tid: str, i: int):
        nonlocal n_done, n_fail
        prefix = pre.get(tid)
        if prefix is None:
            rec = {"turn_id": tid, "ref_idx": i, "z": "", "y": "",
                   "error": "prefix_unavailable"}
        else:
            prompt = gen_prompt(TEACHER, prefix)
            z = y = ""
            for _ in range(4):
                text = await engy.sample(prompt)
                z, y = split_rollout(text)
                if y:
                    break
            rec = {"turn_id": tid, "ref_idx": i, "z": z, "y": y}
        async with lock:
            append_jsonl(REFS_PATH, rec)
            n_done += 1
            n_fail += 0 if rec["y"] else 1
            if n_done % 50 == 0:
                print(f"  refs {n_done}/{len(todo)} (fail={n_fail})")

    pre.prefetch(tids)
    await asyncio.gather(*[one(t, i) for t, i in todo])
    print(f"refs done: {n_done} (fail={n_fail})")


async def stage_score(engy: Engy):
    thoughts = load_thoughts()
    pre = LivePrefixes()
    pre.prefetch(sorted({t["turn_id"] for t in thoughts}))
    refs = {k: v for k, v in
            load_done(REFS_PATH, ("turn_id", "ref_idx")).items() if v["y"]}
    done = {k: v for k, v in
            load_done(SCORES_PATH, ("turn_id", "ref_idx", "model")).items()
            if v.get("lp_per_byte") is not None}

    tasks = []
    empties = set()
    for t in thoughts:
        tid = t["turn_id"]
        if pre.get(tid) is None:
            continue
        model = f"{t['chal_id']}:{t['role']}"
        for i in range(K_REFS):
            ref = refs.get((tid, i))
            if not ref:
                continue
            if (tid, i, model) not in done:
                tasks.append((tid, i, model, t["z"], ref["y"]))
            if (tid, i, "__empty__") not in done and (tid, i) not in empties:
                empties.add((tid, i))
                tasks.append((tid, i, "__empty__", "", ref["y"]))
    print(f"score: {len(done)} cached, {len(tasks)} calls")
    lock = asyncio.Lock()
    n = 0

    async def one(tid, i, model, z, y):
        nonlocal n
        full = force_text(TEACHER, pre.get(tid), z, y)
        try:
            s = await engy.score_span(full, y)
        except RuntimeError as e:
            s = {"sum_lp": None, "n_tokens": 0, "lp_per_byte": None,
                 "error": str(e)[:200]}
        async with lock:
            append_jsonl(SCORES_PATH, {"turn_id": tid, "ref_idx": i,
                                       "model": model, **s})
            n += 1
            if n % 100 == 0:
                print(f"  score {n}/{len(tasks)}")

    await asyncio.gather(*[one(*t) for t in tasks])
    print(f"score done: {n}")


def analyze():
    panel = json.loads(PANEL_PATH.read_text())
    by_tm: dict[tuple[str, str], dict[int, float]] = {}
    for line in open(SCORES_PATH):
        r = json.loads(line)
        if r.get("lp_per_byte") is None:
            continue
        by_tm.setdefault((r["turn_id"], r["model"]), {})[r["ref_idx"]] = \
            r["lp_per_byte"]
    empty = {t: v for (t, m), v in by_tm.items() if m == "__empty__"}

    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for (tid, model), lps in by_tm.items():
        if model == "__empty__" or tid not in empty:
            continue
        a = [lps[i] - empty[tid][i] for i in lps if i in empty[tid]]
        if not a:
            continue
        per_model.setdefault(model, {})[tid] = {
            "lme": turn_score(a, TAU), "mean": turn_score(a, None)}

    # Merge by repo: the same model can appear in several artifacts (kings
    # persist across duels; genesis was added from three early slices).
    agg: dict[str, dict] = {}
    for p in panel:
        key = f"{p['chal_id']}:{p['role']}"
        turns = per_model.get(key, {})
        if not turns:
            continue
        partner_key = (f"{p['chal_id']}:king" if p["role"] == "challenger"
                       else f"{p['chal_id']}:challenger")
        partner = per_model.get(partner_key, {})
        shared = sorted(set(turns) & set(partner))
        a = agg.setdefault(p["repo"], {
            "repo": p["repo"], "swe": p["swe"], "role": p["role"],
            "lme": [], "mean": [], "margins": []})
        a["lme"] += [t["lme"] for t in turns.values()]
        a["mean"] += [t["mean"] for t in turns.values()]
        a["margins"] += [turns[t]["lme"] - partner[t]["lme"] for t in shared]

    rows = []
    for a in agg.values():
        if len(a["lme"]) < 10:
            continue
        rows.append({
            "repo": a["repo"], "swe": a["swe"], "role": a["role"],
            "n_turns": len(a["lme"]),
            "reason_lme": st.mean(a["lme"]),
            "reason_mean": st.mean(a["mean"]),
            "margin_lme": (st.mean(a["margins"])
                           if len(a["margins"]) >= 10 else None),
        })
    rows.sort(key=lambda r: -r["swe"])

    print(f"\n== LIVE panel: engy glm-5.2 Reason vs swe (n={len(rows)}) ==")
    print(f"{'repo':<58} {'swe':>5} {'R_lme':>9} {'R_mean':>9} "
          f"{'margin':>9} {'role':>10}")
    for r in rows:
        mg = f"{r['margin_lme']:+9.5f}" if r["margin_lme"] is not None else \
            "        -"
        print(f"{r['repo']:<58} {r['swe']:>5.2f} {r['reason_lme']:>9.5f} "
              f"{r['reason_mean']:>9.5f} {mg} {r['role']:>10}")

    swe_v = [r["swe"] for r in rows]
    print(f"\nSpearman(Reason_lme tau={TAU}, swe)  = "
          f"{spearman([r['reason_lme'] for r in rows], swe_v):+.3f}   (n={len(rows)})")
    print(f"Spearman(Reason_mean v3-ish, swe)  = "
          f"{spearman([r['reason_mean'] for r in rows], swe_v):+.3f}")
    sub = [r for r in rows if r["margin_lme"] is not None]
    if len(sub) >= 5:
        print(f"Spearman(paired margin_lme, swe)   = "
              f"{spearman([r['margin_lme'] for r in sub], [r['swe'] for r in sub]):+.3f}   (n={len(sub)})")
    out = RESULTS / "engy_reason_live_table.json"
    out.write_text(json.dumps({"tau": TAU, "k": K_REFS, "rows": rows},
                              indent=1))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["panel", "refs", "score", "analyze"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    if args.stage == "panel":
        build_panel()
        return
    if args.stage == "analyze":
        analyze()
        return
    key = os.environ.get("ENGY") or sys.exit("export ENGY first")
    engy = Engy(key, args.concurrency)
    if args.stage == "refs":
        asyncio.run(stage_refs(engy))
    else:
        asyncio.run(stage_score(engy))


if __name__ == "__main__":
    main()
