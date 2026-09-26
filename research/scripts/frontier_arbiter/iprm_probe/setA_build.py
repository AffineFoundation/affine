"""N1 stage-2, Set A — duel-turn separation rows ($0, no API calls).

From recent wvk-22 stored duels (sd_min_rga, as_generated) take 100 turns each,
stratified over bash / tool_call / terminus_json / text, with a valid king AND
challenger row and 3 stored teacher refs; materialize the prefix from the
pinned public corpus; render the teacher prompt; emit per turn the candidate
actions (king, challenger, ref_0..2 with their thoughts, attack `ls -la`,
attack repeat-last) with body spans (setAB_render) and the LIVE per-turn legs
of the sd-meter (z_R / typ_c / z_A / score / bind, from evalsrv.sdmeter on the
stored pair fields, anchors LOO over the whole duel) plus the raw lp fields.

    python research/scripts/frontier_arbiter/iprm_probe/setA_build.py \
        [--records chal-00614 ...] [--per-record 100] [--per-dialect 25]
    -> research/results/frontier_arbiter/iprm/setA_turns.jsonl (+ setA_meta.json)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import setAB_render as R  # noqa: E402
from panel import _pick_stratified  # noqa: E402
from evalsrv import sdmeter  # noqa: E402  (live tree via common's sys.path)

OUT = C.REPO / "research" / "results" / "frontier_arbiter" / "iprm"
RECORDS = ("chal-00614", "chal-00621", "chal-00623", "chal-00625", "chal-00630")
MAX_PREFIX_CHARS = 160_000
PAIR_SHARED = ("lpC_ya_za", "lpC_ya_e", "n_bytes_ya", "lpC_za_x", "lpC_za_e", "mc_za",
               "n_content_za", "n_tokens_za", "mean_lift_za", "eta")
PAIR_PER_REF = ("lpC_yc_za", "lpC_ya_zc", "lpC_yc_zc", "lpC_yc_e", "lpC_zc_x", "mc_zc", "n_content_zc")
REF_FIELDS = ("lp_own", "lp_empty", "lp_thought", "lp_thought_e", "n_bytes_y", "lp_cross",
              "mc_thought", "n_content_thought", "n_tokens_thought")


def side_live(row: dict, refs: list[dict], tau: float, cfg: dict, mu: dict | None,
              sigma: dict | None) -> dict:
    pairs = row["pairs"]
    legs = sdmeter.side_legs(row, tau, cfg["a_norm_bytes"])
    score = sdmeter.turn_score(legs, mu, sigma, cfg)
    p0 = pairs[0]
    return {"legs": legs, "sd": score,
            "lp": {k: p0.get(k) for k in PAIR_SHARED},
            "lp_per_ref": [{k: p.get(k) for k in PAIR_PER_REF} for p in pairs],
            "a_match": row.get("a_match"), "ref_pair": row.get("ref_pair"),
            "cap_tokens": row.get("cap_tokens")}


def ref_live(refs: list[dict], j: int, tau: float, cfg: dict, sigma: dict | None) -> dict:
    """Ref j scored as a miner against the other k-1 refs (sdmeter's positive
    control, per ref instead of averaged)."""
    t = sdmeter.ref_loo_terms(refs, tau, cfg["a_norm_bytes"])
    out = {"lp": {k: refs[j].get(k) for k in REF_FIELDS}}
    if t is None:
        out["legs"], out["sd"] = None, None
        return out
    legs = {"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
            "n_content": refs[j].get("n_content_thought"), "n_tokens": refs[j].get("n_tokens_thought")}
    others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None] for leg in ("R", "A", "Mc")}
    mu = {leg: st.mean(v) for leg, v in others.items() if v}
    out["legs"] = legs
    out["sd"] = sdmeter.turn_score(legs, mu, sigma, cfg)
    return out


def build_record(rec: str, rng: random.Random, per_record: int, per_dialect: int) -> tuple[list[dict], dict]:
    d = C.load_verdict(rec)
    req, v = d["request"], d["verdict"]
    dp = v["duel_params"]
    tau = dp.get("tau")
    cfg = {**sdmeter.DEFAULTS, **(dp.get("sd_meter") or {})}
    refs_by = d["teacher_refs"]
    k_by = {r["turn_id"]: r for r in d["king_rows"]}
    c_by = {r["turn_id"]: r for r in d["challenger_rows"]}
    sl = v["slice"]
    corpus = C.corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    kind_by_tid = {t: rows[t].get("action_kind", "bash") for t in d["turn_ids"] if t in rows}
    anchors = sdmeter.loo_anchors(refs_by, kind_by_tid, tau, cfg["a_norm_bytes"])

    cands: dict[str, list[str]] = collections.defaultdict(list)
    n_seen = n_long = 0
    for t in d["turn_ids"]:
        if t not in rows or len(refs_by.get(t) or []) < 3:
            continue
        if not (C.not_forfeit(k_by.get(t)) and C.not_forfeit(c_by.get(t))):
            continue
        if not all(r.get("y") for r in refs_by[t]):
            continue
        kind = kind_by_tid[t]
        if kind not in R.SET_DIALECTS:
            continue
        n_seen += 1
        if rows[t]["n_prefix_chars"] > MAX_PREFIX_CHARS:
            n_long += 1
            continue
        cands[kind].append(t)
    picked = _pick_stratified(cands, rng, per_dialect, per_record)
    mats = C.materialize(d, picked)
    king_digest, chal_digest = req["king_revision"][:12], req["challenger_revision"][:12]
    out = []
    n_no_repeat = 0
    for t in picked:
        m = mats[t]
        prefix, kind = m["prefix"], kind_by_tid[t]
        prompt = C.gen_prompt(prefix)
        refs = refs_by[t]
        mu, sigma = anchors.mu.get(t), anchors.sigma.get(kind)
        cl = []
        for side, row, digest in (("king", k_by[t], king_digest), ("challenger", c_by[t], chal_digest)):
            p = row["pairs"][0]
            cl.append(R.render_candidate(prompt, side, p.get("z_a") or "", p.get("y_a") or "", kind, digest,
                                         live=side_live(row, refs, tau, cfg, mu, sigma)))
        for j, rf in enumerate(refs):
            cl.append(R.render_candidate(prompt, f"ref_{j}", rf.get("z") or "", rf["y"], kind, "teacher",
                                         live=ref_live(refs, j, tau, cfg, sigma)))
        y_ls, meta_ls = R.generic_action(kind, prefix)
        cl.append(R.render_candidate(prompt, "attack_ls", "", y_ls, kind, None, **meta_ls))
        y_rep = R.repeat_last_action(kind, prefix)
        if y_rep:
            cl.append(R.render_candidate(prompt, "attack_repeat", "", y_rep, kind, None))
        else:
            n_no_repeat += 1
        out.append({
            "set": "A", "record": rec, "turn_id": t, "dialect": kind, "source": m.get("source"),
            "group": R.group_of(m.get("source")), "phase": m.get("phase"), "depth": R.depth_of(prefix),
            "n_prefix_chars": rows[t]["n_prefix_chars"], "n_prompt_chars": len(prompt),
            "king_model": king_digest, "challenger_model": chal_digest,
            "prompt": prompt, "candidates": cl,
            "live_anchors": {"mu": mu, "sigma": sigma},
        })
    cands_all = [c for r in out for c in r["candidates"]]
    meta = {"record": rec, "king": king_digest, "challenger": chal_digest, "n_turns": len(out),
            "n_candidates": n_seen, "n_dropped_long": n_long, "n_no_repeat_last": n_no_repeat,
            "by_dialect": dict(collections.Counter(r["dialect"] for r in out)),
            "n_candidate_rows": len(cands_all),
            "body_kinds": dict(collections.Counter(c["rel"]["body_kind"] for c in cands_all)),
            "n_text_fallback": sum(1 for c in cands_all if c["y_kind"] != c["turn_kind"]),
            "n_terminus_empty_batch": sum(1 for c in cands_all if c["rel"]["body_kind"] == "commands_array"),
            "n_attack_ls_tool_missing": sum(1 for c in cands_all if c.get("attack_tool_in_schema") is False),
            "manifest_sha256": sl["manifest_sha256"], "score_mode": dp.get("score_mode"),
            "thought_rendering": dp.get("thought_rendering"), "tau": tau, "sd_meter": cfg,
            "margin": v.get("margin"), "z": v.get("z"), "challenger_wins": v.get("challenger_wins"),
            "sigma_by_dialect": anchors.sigma}
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="*", default=list(RECORDS))
    ap.add_argument("--per-record", type=int, default=100)
    ap.add_argument("--per-dialect", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=Path, default=OUT / "setA_turns.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        a.out.unlink()
    metas = {}
    total = collections.Counter()
    for rec in a.records:
        rows, meta = build_record(rec, rng, a.per_record, a.per_dialect)
        for r in rows:
            C.append_jsonl(a.out, r)
        metas[rec] = meta
        total.update(meta["by_dialect"])
        print(f"{rec}: {meta['n_turns']} turns {meta['by_dialect']} (cands {meta['n_candidates']}, "
              f"dropped long {meta['n_dropped_long']}, no repeat-last {meta['n_no_repeat_last']}) "
              f"king={meta['king']} chal={meta['challenger']} z={meta['z']:+.2f}", flush=True)
    (OUT / "setA_meta.json").write_text(json.dumps(
        {"records": metas, "by_dialect": dict(total), "n_turns": sum(total.values()),
         "per_record": a.per_record, "per_dialect": a.per_dialect, "seed": a.seed,
         "max_prefix_chars": MAX_PREFIX_CHARS, "renderings": ["no_thought", "with_thought"],
         "candidate_roles": ["king", "challenger", "ref_0", "ref_1", "ref_2", "attack_ls", "attack_repeat"],
         "n_candidate_rows": sum(m["n_candidate_rows"] for m in metas.values()),
         "body_kinds": dict(sum((collections.Counter(m["body_kinds"]) for m in metas.values()), collections.Counter())),
         "n_text_fallback": sum(m["n_text_fallback"] for m in metas.values()),
         "n_terminus_empty_batch": sum(m["n_terminus_empty_batch"] for m in metas.values()),
         "n_attack_ls_tool_missing": sum(m["n_attack_ls_tool_missing"] for m in metas.values()),
         "notes": ["tool_call actions are Qwen3 XML (<function=…><parameter=…>), body = every parameter value (list of spans)",
                   "terminus empty batches ('commands': []) have a one-token body: the scorer's overlap fallback scores the ' [],' token; use --level action as a sensitivity check",
                   "y_kind != turn dialect marks a text fallback reply on a tool_call turn (text_fallback_at_tool_turns)",
                   "live legs: evalsrv.sdmeter side_legs/turn_score with LOO anchors over the whole duel; refs carry their own LOO legs"]},
        indent=1))
    print(f"total {sum(total.values())} turns {dict(total)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
