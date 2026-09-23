"""N1 stage-2, Set B — benchmark-panel alignment rows ($0, no API calls).

Reuses the frontier-arbiter panel selection (research/results/frontier_arbiter/
panel/turns.jsonl.gz: kings 11-20 as challenger in their crowning duel and as
king afterwards, k=3 teacher refs per turn; genesis samples from
panel/frontier/qwen3.6-35b-a3b.jsonl) and emits one row per (model, turn):
the rendered teacher prompt, the model's action with body spans, the 3 teacher
ref actions with spans, and the model's benchmark card axes exactly as panel.py
computed them (bench_axes + axis_means on the cached matrix.json: centered /
raw / n cells for total, agentic, agentic_no_tau2, chat).

Models: kings 11-20 (digest12, <=150 turns each, 30 per dialect then refilled),
genesis (its parsed samples, <=150), and the teacher as a model row on every
selected turn (candidates = its 3 refs; the scorer reads them leave-one-out:
ref_i's term minus the mean of the other two).

    python research/scripts/frontier_arbiter/iprm_probe/setB_build.py [--per-model 150]
    -> research/results/frontier_arbiter/iprm/setB_turns.jsonl.gz (+ setB_meta.json)
(gzipped: ~1,700 unique prompts x ~50k chars would be >100 MB plain.)
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import panel as P  # noqa: E402
import setAB_render as R  # noqa: E402

OUT = C.REPO / "research" / "results" / "frontier_arbiter" / "iprm"
PER_DIALECT = 30


def model_turns(turns: list[dict], genesis: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """model key -> {turn_id: {"z", "y", "record", "role"}} for every panel model."""
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for t in turns:
        for side, s in t["sides"].items():
            if s.get("y"):
                out[s["model"]][t["turn_id"]] = {"z": s.get("z") or "", "y": s["y"], "record": t["record"],
                                                 "side": side}
    for tid, s in genesis.items():
        out[P.GENESIS_KEY][tid] = {"z": s.get("z") or "", "y": s["y"], "record": None, "side": "genesis_sample"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-model", type=int, default=150)
    ap.add_argument("--per-dialect", type=int, default=PER_DIALECT)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=Path, default=OUT / "setB_turns.jsonl.gz")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    turns = P.load_turns()
    by_tid = {t["turn_id"]: t for t in turns}
    gen = {}
    for s in C.read_jsonl(P.frontier_path(C.GENESIS_ENGY)):
        if s.get("parsed") and not s.get("error") and s["turn_id"] in by_tid and s["turn_id"] not in gen:
            gen[s["turn_id"]] = s
    cards, cinfo = P.bench_axes()
    labels = {d: f"king{r}" for r, (d, _, _) in P.KINGS.items()}
    labels[P.TEACHER_KEY] = "teacher"
    labels[P.GENESIS_KEY] = "genesis"
    panel_keys = [P.TEACHER_KEY, P.GENESIS_KEY] + [P.KINGS[r][0] for r in P.KINGS]
    panel_keys = [k for k in panel_keys if k in cards]
    axes = P.axis_means(cards, panel_keys)
    bench = {k: {ax: axes[k].get(ax) for ax in P.AXES} | {ax + "_raw": axes[k].get(ax + "_raw") for ax in P.AXES}
             | {ax + "_n": axes[k].get(ax + "_n") for ax in P.AXES} | {"cells": cards[k]["cells"]}
             for k in panel_keys}

    mt = model_turns(turns, gen)
    rng = random.Random(a.seed)
    models = [P.KINGS[r][0] for r in P.KINGS] + [P.GENESIS_KEY]
    selection: dict[str, list[str]] = {}
    for mk in models:
        pool = mt.get(mk, {})
        cands: dict[str, list[str]] = collections.defaultdict(list)
        for tid in pool:
            cands[by_tid[tid]["kind"]].append(tid)
        selection[mk] = sorted(P._pick_stratified(cands, rng, a.per_dialect, a.per_model))
    all_tids = sorted({tid for tids in selection.values() for tid in tids})

    prompts: dict[str, str] = {}
    rows_out = 0
    counts: collections.Counter = collections.Counter()
    body_kinds: collections.Counter = collections.Counter()
    n_fallback = 0
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with gzip.open(a.out, "wt", compresslevel=6) as fo:
        def emit(mk: str, tid: str, cand: dict | None) -> None:
            nonlocal rows_out, n_fallback
            t = by_tid[tid]
            prompt = prompts.get(tid)
            if prompt is None:
                prompt = prompts[tid] = C.gen_prompt(t["prefix"])
            kind = t["kind"]
            refs = [R.render_candidate(prompt, f"ref_{j}", rf.get("z") or "", rf["y"], kind, P.TEACHER_KEY)
                    for j, rf in enumerate(t["refs"])]
            cl = ([cand] if cand else []) + refs
            row = {"set": "B", "model": mk, "label": labels.get(mk, mk), "turn_id": tid, "record": t["record"],
                   "reign": t["reign"], "role": t["role"], "dialect": kind, "source": t["source"],
                   "group": t["group"], "phase": t.get("phase"), "depth": t["depth"],
                   "n_prefix_chars": t["n_prefix_chars"], "n_prompt_chars": len(prompt),
                   "bench": bench.get(mk), "prompt": prompt, "candidates": cl}
            fo.write(json.dumps(row) + "\n")
            rows_out += 1
            for c in cl:
                body_kinds[c["rel"]["body_kind"]] += 1
                n_fallback += c["y_kind"] != c["turn_kind"]
            counts[mk] += 1
            by_kind[mk][kind] += 1

        for mk in models:
            for tid in selection[mk]:
                s = mt[mk][tid]
                if tid not in prompts:
                    prompts[tid] = C.gen_prompt(by_tid[tid]["prefix"])
                cand = R.render_candidate(prompts[tid], "model", s["z"], s["y"], by_tid[tid]["kind"], mk,
                                          side=s["side"], sample_record=s["record"])
                emit(mk, tid, cand)
            print(f"{labels.get(mk, mk):8} {counts[mk]:4d} rows {dict(by_kind[mk])}", flush=True)
        for tid in all_tids:
            emit(P.TEACHER_KEY, tid, None)
        print(f"{'teacher':8} {counts[P.TEACHER_KEY]:4d} rows (LOO over its 3 refs) {dict(by_kind[P.TEACHER_KEY])}")
    meta = {"n_rows": rows_out, "n_unique_turns": len(all_tids), "per_model": a.per_model,
            "per_dialect": a.per_dialect, "seed": a.seed, "rows_by_model": {labels.get(k, k): v for k, v in counts.items()},
            "by_dialect_by_model": {labels.get(k, k): dict(v) for k, v in by_kind.items()},
            "by_dialect": dict(sum(by_kind.values(), collections.Counter())),
            "bench_axes": {labels.get(k, k): {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in bench.items()},
            "bench_axis_cells": axes["_axes"], "bench_cell_means": axes["_cell_mean"],
            "matrix_generated_at": cinfo.get("generated_at"), "renderings": ["no_thought", "with_thought"],
            "n_candidate_rows": sum(body_kinds.values()), "body_kinds": dict(body_kinds), "n_text_fallback": n_fallback,
            "n_terminus_empty_batch": body_kinds.get("commands_array", 0),
            "notes": ["one row per (model, turn); teacher rows carry only the 3 refs (LOO: ref_i − mean of the other two ≡ 0 on average for k=3, so the teacher anchors excess at 0)",
                      "genesis rows = the panel's parsed qwen3.6-35b-a3b samples (Engy) on crowning-duel turns",
                      "identical candidate texts on a turn (boxed answers) dedupe to one echo in score_sets.py"],
            "source_panel": str(P.turns_path().relative_to(C.REPO))}
    (OUT / "setB_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"total {rows_out} rows over {len(all_tids)} unique turns -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
