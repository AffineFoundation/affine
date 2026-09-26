"""Design S9 — what the typicality leg costs and buys (2026-09-21).

Replays every stored wvk-22 verdict (score_mode = "sd_min_rga", both sides
valid) through the LIVE sd-meter code path (evalsrv.sdmeter of the live tree
at /tmp/box/affine) and then re-decides each duel under three variants of the
per-turn min:

    live  turn = min(z_R, typ_c, z_A)      (the wvk-22 rule)
    (a)   turn = min(z_R, z_A)             no typicality
    (b)   turn = z_R                       Reason only
    (c)   turn = min(z_R, typ_c)           no action leg

Everything else is held fixed: the LOO anchors (μ per turn, σ per dialect
from the k(k−1) cross echoes on the refs), the forfeit floor, the pairing rule,
the bar max(k_sigma·SE, min_margin_sd) and the live gates (thought floor, B
licence) read from the stored verdict. Per verdict the script reports margin /
SE / z / rule_passes / binding leg per variant, which crowns flip, the
Spearman rank correlation of per-turn paired differences with live, the SE
ratio, and the teacher-vs-king positive control (each ref j scored as the
miner against the other k−1 refs, exactly as the live shadow computes it).
Two synthetic miners are scored from the king's own legs: FLAT (z_R = 0,
z_A = the king's — a content-free thought over the king's actions) and
A-MIMIC (z_R = the king's, z_A = 0 — the king's thought over a teacher-mean
action). A per-dialect table pools the paired differences over all verdicts.
The stored attack arms of the 4 older duels (research/results/
frontier_rule_probe: filler / generic / parrot thoughts, teacher echoes) are
re-scored under teacher R-only vs min(R,G) for the anti-filler question.

    cd /workspace && source .venv/bin/activate
    python research/scripts/frontier_arbiter/no_typicality_replay.py [--recs chal-00588,...]

Inputs: research/data/frontier_arbiter/evals/chal-XXXXX.json.gz (scp'd from the
validator box by common.fetch_verdict), the pinned corpus manifest's turn index
(downloaded from data.affine.io into --index-dir; turn_id -> action_kind, the
per-dialect σ pooling key). Outputs: research/results/frontier_arbiter/
no_typicality/report.{txt,json}. No GPU, no API spend, no git.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontier_arbiter.common import EVALS_CACHE, REPO, clme, fetch_verdict, read_jsonl  # noqa: E402
from evalsrv import sdmeter  # noqa: E402
from affine.score import action_leg, centered_reason, is_forfeit  # noqa: E402

OUT = REPO / "research/results/frontier_arbiter/no_typicality"
PROBE = REPO / "research/results/frontier_rule_probe"
CORPUS_BASE = "https://data.affine.io"
FIRST_WVK22 = "chal-00587"

VARIANTS = {
    "live": ("R", "Gc", "A"),
    "a_no_typ": ("R", "A"),
    "b_r_only": ("R",),
    "c_no_a": ("R", "Gc"),
}
VARIANT_LABEL = {
    "live": "live min(z_R,typ,z_A)",
    "a_no_typ": "(a) min(z_R,z_A)",
    "b_r_only": "(b) z_R only",
    "c_no_a": "(c) min(z_R,typ)",
}
DIALECTS = ("bash", "tool_call", "text", "boxed", "terminus_json")


# ------------------------------------------------------------------ inputs
def wvk22_records() -> list[str]:
    """Cached verdicts from FIRST_WVK22 on whose stored decision came from the
    sd-meter and that carry both sides' rows (protocol rejections have none)."""
    out = []
    for p in sorted(EVALS_CACHE.glob("chal-*.json.gz")):
        rec = p.name.split(".")[0]
        if rec < FIRST_WVK22:
            continue
        d = json.load(gzip.open(p))
        dp = (d.get("verdict") or {}).get("duel_params") or {}
        if dp.get("score_mode") != "sd_min_rga":
            continue
        if not d.get("king_rows") or not d.get("challenger_rows"):
            continue
        out.append(rec)
    return out


def load(rec: str) -> dict:
    return json.load(gzip.open(fetch_verdict(rec)))


class KindIndex:
    """turn_id -> action_kind from the corpus manifest a verdict pinned."""

    def __init__(self, index_dir: Path):
        self.dir = index_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.maps: dict[str, dict[str, str]] = {}

    def for_manifest(self, sha: str, base: str) -> dict[str, str]:
        if sha in self.maps:
            return self.maps[sha]
        dst = self.dir / f"{sha[:12]}.parquet"
        if not dst.exists():
            m = httpx.get(f"{base}/corpus/manifests/{sha}.json", timeout=120).json()
            idx = m["index"]
            body = httpx.get(f"{base}/{idx['key']}", timeout=600).content
            if hashlib.sha256(body).hexdigest() != idx["sha256"]:
                raise RuntimeError(f"index sha mismatch for manifest {sha[:12]}")
            dst.write_bytes(body)
        t = pq.read_table(dst, columns=["turn_id", "action_kind"])
        self.maps[sha] = dict(zip(t.column("turn_id").to_pylist(),
                                  t.column("action_kind").to_pylist()))
        return self.maps[sha]


def sd_cfg(verdict: dict) -> dict:
    """The [duel.sd_meter] knobs exactly as stamped on the verdict."""
    stamp = verdict["duel_params"]["sd_meter"]
    cfg = dict(sdmeter.DEFAULTS)
    cfg.update({k: stamp[k] for k in stamp if k in cfg})
    cfg["frozen"] = {}
    return cfg


# ------------------------------------------------------------------ variant scoring
def variant_score(comp: dict, legs: tuple[str, ...], floor: float) -> dict:
    """Re-min one side's sd-meter components over a subset of legs.
    comp = sdmeter.turn_score output ({score, bind, z_R, typ_c, z_A})."""
    if comp["bind"] == "forfeit":
        return {"score": floor, "bind": "forfeit"}
    vals = {"R": comp["z_R"], "Gc": comp["typ_c"], "A": comp["z_A"]}
    live = {k: vals[k] for k in legs if vals[k] is not None}
    if not live:
        return {"score": None, "bind": None}
    bind = min(live, key=live.get)
    return {"score": live[bind], "bind": bind}


def teacher_components(turn_refs: dict, kind_by_tid: dict, loo: sdmeter.Anchors,
                       tau: float | None, cfg: dict) -> dict[str, list[dict]]:
    """Per turn, the k leave-one-out teacher components (ref j as the miner
    against the other refs) — the same construction as
    sdmeter.shadow_verdict.teacher_scores, kept per j so variants can re-min."""
    out = {}
    for tid, refs in turn_refs.items():
        t = sdmeter.ref_loo_terms(refs, tau, cfg["a_norm_bytes"])
        if t is None:
            continue
        kind = kind_by_tid.get(tid) or "bash"
        sig = loo.sigma.get(kind)
        per = []
        for j in range(len(refs)):
            legs = {"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                    "n_content": refs[j].get("n_content_thought"),
                    "n_tokens": refs[j].get("n_tokens_thought")}
            others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None]
                      for leg in ("R", "A", "Mc")}
            mu = {leg: st.mean(v) for leg, v in others.items() if v}
            s = sdmeter.turn_score(legs, mu, sig, cfg)
            if s["score"] is not None:
                per.append(s)
        if per:
            out[tid] = per
    return out


def fair_control_components(turn_refs: dict, king_rows: list[dict], kind_by_tid: dict,
                            loo: sdmeter.Anchors, tau: float | None, cfg: dict
                            ) -> dict[str, list[tuple[dict, dict]]]:
    """Same-k control: for each held-out ref j, the KING is re-scored on the
    same k−1 refs (pairs i ≠ j: centred LME over 2 a_i, A over 2 b_i, same
    μ_j and σ) so teacher_j and king_j share the reference set. The live
    shadow control scores the king on all k refs against a k−1-ref μ, which
    shifts the centred-LME and LME scales between the two sides. Returns per
    turn a list of (teacher_j components, king_j components)."""
    a_norm = cfg["a_norm_bytes"]
    krow = {r["turn_id"]: r for r in king_rows}
    out = {}
    for tid, refs in turn_refs.items():
        t = sdmeter.ref_loo_terms(refs, tau, a_norm)
        row = krow.get(tid)
        if t is None or row is None or is_forfeit(row):
            continue
        pairs = row["pairs"]
        if len(pairs) != len(refs) or any(
                abs(p["lpC_yc_e"] - r["lp_empty"]) > 1e-9 for p, r in zip(pairs, refs)):
            continue
        kind = kind_by_tid.get(tid) or "bash"
        sig = loo.sigma.get(kind)
        p0 = pairs[0]
        per = []
        for j in range(len(refs)):
            others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None]
                      for leg in ("R", "A", "Mc")}
            mu = {leg: st.mean(v) for leg, v in others.items() if v}
            t_legs = {"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                      "n_content": refs[j].get("n_content_thought"),
                      "n_tokens": refs[j].get("n_tokens_thought")}
            sub = [p for i, p in enumerate(pairs) if i != j]
            k_legs = {"R": centered_reason(sub, tau), "A": action_leg(sub, tau, a_norm),
                      "mc": p0.get("mc_za"), "n_content": p0.get("n_content_za"),
                      "n_tokens": p0.get("n_tokens_za")}
            ts = sdmeter.turn_score(t_legs, mu, sig, cfg)
            ks = sdmeter.turn_score(k_legs, mu, sig, cfg)
            if ts["score"] is not None and ks["score"] is not None:
                per.append((ts, ks))
        if per:
            out[tid] = per
    return out


def fair_control_diffs(per_turn: dict[str, list[tuple[dict, dict]]], legs: tuple[str, ...],
                       floor: float) -> dict[str, float]:
    """Per turn mean_j (teacher_j − king_j) under one variant."""
    out = {}
    for tid, per in per_turn.items():
        ds = []
        for ts, ks in per:
            a, b = variant_score(ts, legs, floor), variant_score(ks, legs, floor)
            if a["score"] is not None and b["score"] is not None:
                ds.append(a["score"] - b["score"])
        if ds:
            out[tid] = st.mean(ds)
    return out


def teacher_variant(per_turn: dict[str, list[dict]], legs: tuple[str, ...], floor: float) -> dict:
    out = {}
    for tid, per in per_turn.items():
        vs = [variant_score(p, legs, floor) for p in per]
        vs = [v for v in vs if v["score"] is not None]
        if not vs:
            continue
        binds = Counter(v["bind"] for v in vs)
        out[tid] = {"score": st.mean(v["score"] for v in vs), "bind": binds.most_common(1)[0][0]}
    return out


def bind_frac(scores: dict[str, dict]) -> dict:
    valid = [s for s in scores.values() if s["bind"] not in ("forfeit", None) and s["score"] is not None]
    n = len(valid)
    return {k: (sum(1 for s in valid if s["bind"] == k) / n if n else None) for k in ("R", "Gc", "A")}


def paired_diffs(c: dict[str, dict], k: dict[str, dict]) -> dict[str, float]:
    """Per-turn challenger − king differences under the live pairing rule."""
    out = {}
    for tid in sorted(set(c) & set(k)):
        cs, ks = c[tid], k[tid]
        if cs["score"] is None or ks["score"] is None:
            continue
        if cs["bind"] == "forfeit" and ks["bind"] == "forfeit":
            out[tid] = 0.0
        else:
            out[tid] = cs["score"] - ks["score"]
    return out


def pooled_stats(diffs: list[float]) -> dict:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": None, "se": None, "z": None}
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    return {"n": n, "mean": m, "se": se, "z": (m / se) if se > 0 else None}


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ------------------------------------------------------------------ one verdict
def replay(rec: str, kinds: KindIndex) -> dict:
    d = load(rec)
    v = d["verdict"]
    dp = v["duel_params"]
    sl = v["slice"]
    kind_map = kinds.for_manifest(sl["manifest_sha256"], sl.get("corpus_base_url") or CORPUS_BASE)
    tids = d["turn_ids"]
    missing = [t for t in tids if t not in kind_map]
    kind_by_tid = {t: kind_map.get(t, "bash") for t in tids}
    cfg = sd_cfg(v)
    tau = dp["tau"]
    floor = cfg["forfeit_sd"]
    gates_ok = v["rejection_reason"] not in ("thought_too_short", "causality_fail")
    chall_rows, king_rows, refs = d["challenger_rows"], d["king_rows"], d["teacher_refs"]

    # 1. Parity: the live code path end to end.
    sh = sdmeter.shadow_verdict(chall_rows, king_rows, refs, kind_by_tid, tau, cfg,
                                live_gates_pass=gates_ok)
    head = sh["by_anchor"]["loo"]
    stored_sd = v["shadow"]["sd_meter"]
    stored_head = stored_sd["by_anchor"]["loo"]
    parity = {
        "margin": close(head["margin"], v["margin"]),
        "se": close(head["se"], v["se"]),
        "z": close(head["z"], v["z"]),
        "wins": bool(head["would_crown"]) == bool(v["challenger_wins"]),
        "teacher_vs_king_z": close(head["teacher_vs_king"]["z"], stored_head["teacher_vs_king"]["z"]),
        "sigma_by_dialect": sh["sigma_by_dialect"] == stored_sd["sigma_by_dialect"],
        "n_loo_by_dialect": sh["n_loo_turns_by_dialect"] == stored_sd["n_loo_turns_by_dialect"],
        "kinds_missing_from_index": len(missing),
        "replayed": {"margin": head["margin"], "se": head["se"], "z": head["z"]},
        "stored": {"margin": v["margin"], "se": v["se"], "z": v["z"],
                   "challenger_wins": v["challenger_wins"],
                   "teacher_vs_king": stored_head["teacher_vs_king"]},
    }
    parity["all"] = all(parity[k] for k in ("margin", "se", "z", "wins", "teacher_vs_king_z",
                                             "sigma_by_dialect", "n_loo_by_dialect"))

    # 2. Components per side per turn (the live turn_score, kept whole).
    a_norm = cfg["a_norm_bytes"]
    c_legs = {r["turn_id"]: sdmeter.side_legs(r, tau, a_norm) for r in chall_rows}
    k_legs = {r["turn_id"]: sdmeter.side_legs(r, tau, a_norm) for r in king_rows}
    loo = sdmeter.loo_anchors(refs, kind_by_tid, tau, a_norm)

    def comps(legs_by_tid):
        return {tid: sdmeter.turn_score(l, loo.mu.get(tid), loo.sigma.get(kind_by_tid[tid]), cfg)
                for tid, l in legs_by_tid.items()}
    c_comp, k_comp = comps(c_legs), comps(k_legs)
    t_comp = teacher_components(refs, kind_by_tid, loo, tau, cfg)
    fair_comp = fair_control_components(refs, king_rows, kind_by_tid, loo, tau, cfg)

    live_diffs = None
    by_variant = {}
    for name, legs in VARIANTS.items():
        c_s = {tid: variant_score(c, legs, floor) for tid, c in c_comp.items()}
        k_s = {tid: variant_score(c, legs, floor) for tid, c in k_comp.items()}
        t_s = teacher_variant(t_comp, legs, floor)
        if name == "live":
            # internal parity: re-min over all three legs == the live score
            assert all(close(c_s[t]["score"], c_comp[t]["score"]) for t in c_s), rec
        pr = sdmeter._paired(c_s, k_s, cfg)
        tk = sdmeter._paired(t_s, {t: s for t, s in k_s.items() if t in t_s}, cfg)
        tc = sdmeter._paired(t_s, {t: s for t, s in c_s.items() if t in t_s}, cfg)
        diffs = paired_diffs(c_s, k_s)
        if name == "live":
            live_diffs = diffs
            rho, se_ratio, z_ratio = 1.0, 1.0, 1.0
        else:
            common = sorted(set(diffs) & set(live_diffs))
            rho = float(spearmanr([diffs[t] for t in common], [live_diffs[t] for t in common])[0])
            se_ratio = pr["se"] / head["se"] if pr["se"] and head["se"] else None
            z_ratio = (pr["z"] / head["z"]) if pr["z"] is not None and head["z"] else None
        # per-dialect paired diffs (challenger − king, teacher − king, fair teacher − king)
        t_diffs = paired_diffs(t_s, {t: s for t, s in k_s.items() if t in t_s})
        f_diffs = fair_control_diffs(fair_comp, legs, floor)
        fair = pooled_stats(list(f_diffs.values()))
        per_dialect = {}
        for kind in DIALECTS:
            ck = [dv for t, dv in diffs.items() if kind_by_tid[t] == kind]
            tk_ = [dv for t, dv in t_diffs.items() if kind_by_tid[t] == kind]
            fk = [dv for t, dv in f_diffs.items() if kind_by_tid[t] == kind]
            both = [s for side in (c_s, k_s) for t, s in side.items()
                    if kind_by_tid[t] == kind and s["bind"] not in ("forfeit", None)]
            per_dialect[kind] = {"chal_minus_king": pooled_stats(ck), "teacher_minus_king": pooled_stats(tk_),
                                 "fair_teacher_minus_king": pooled_stats(fk),
                                 "gc_bind": (sum(1 for s in both if s["bind"] == "Gc") / len(both)) if both else None,
                                 "n_valid_sides": len(both),
                                 "diffs_ck": ck, "diffs_tk": tk_, "diffs_fk": fk}
        by_variant[name] = {
            "legs": list(legs),
            "margin": pr["margin"], "se": pr["se"], "z": pr["z"], "bar": pr.get("bar"),
            "n_paired_turns": pr["n_paired_turns"], "n_forfeit_turns": pr["n_forfeit_turns"],
            "rule_passes": pr["rule_passes"],
            "would_crown": bool(pr["rule_passes"]) and gates_ok,
            "bind_frac_challenger": bind_frac(c_s), "bind_frac_king": bind_frac(k_s),
            "bind_frac_teacher": bind_frac(t_s),
            "mean_challenger": sdmeter._mean([s["score"] for s in c_s.values()]),
            "mean_king": sdmeter._mean([s["score"] for s in k_s.values()]),
            "mean_teacher": sdmeter._mean([s["score"] for s in t_s.values()]),
            "teacher_vs_king": {k: tk.get(k) for k in ("margin", "se", "z", "n_paired_turns")},
            "fair_teacher_vs_king": {"margin": fair["mean"], "se": fair["se"], "z": fair["z"],
                                     "n_paired_turns": fair["n"]},
            "teacher_vs_challenger": {k: tc.get(k) for k in ("margin", "se", "z", "n_paired_turns")},
            "rho_vs_live": rho, "se_ratio_vs_live": se_ratio, "z_ratio_vs_live": z_ratio,
            "per_dialect": per_dialect,
        }

    # 3. Synthetic miners built from the king's own components.
    def synth(kind: str, legs: tuple[str, ...]) -> dict:
        """FLAT: z_R := 0 (content-free thought), king's z_A, typ := floor (a
        thought with no content tokens gets typ_c = forfeit_sd in the live
        rule). A-MIMIC: king's z_R and typ (same thought), z_A := 0."""
        m_s, k_s = {}, {}
        for tid, c in k_comp.items():
            k_s[tid] = variant_score(c, legs, floor)
            if c["bind"] == "forfeit":
                m_s[tid] = {"score": floor, "bind": "forfeit"}
                continue
            if kind == "flat":
                comp = {"bind": "x", "z_R": 0.0, "typ_c": floor, "z_A": c["z_A"]}
            else:
                comp = {"bind": "x", "z_R": c["z_R"], "typ_c": c["typ_c"], "z_A": 0.0}
            m_s[tid] = variant_score(comp, legs, floor)
        pr = sdmeter._paired(m_s, k_s, cfg)
        return {k: pr.get(k) for k in ("margin", "se", "z", "n_paired_turns", "rule_passes")}
    synthetic = {kind: {name: synth(kind, legs) for name, legs in VARIANTS.items()}
                 for kind in ("flat", "a_mimic")}

    return {
        "rec": rec,
        "live": {"challenger_wins": v["challenger_wins"], "margin": v["margin"], "se": v["se"],
                 "z": v["z"], "rejection_reason": v["rejection_reason"], "gates_ok": gates_ok,
                 "n_paired_turns": v["n_paired_turns"], "n_forfeit_turns": v["n_forfeit_turns"],
                 "dialects": sl.get("dialects"), "corpus_epoch": sl.get("corpus_epoch")},
        "parity": parity,
        "sigma_by_dialect": sh["sigma_by_dialect"],
        "variants": by_variant,
        "synthetic": synthetic,
    }


# ------------------------------------------------------------------ older-duel attack arms (teacher R-only vs min(R,G))
def attack_arms_r_only(band_c: float = 4.0, band_floor: float = 0.002, tau: float = 0.03) -> dict:
    """filler (king thought + fixed suffix), generic (fixed task-agnostic
    thought), parrot (king thought + frontier action y_F^0 pasted) on the 4
    stored wvk-11 duels, scored under teacher R-only (centred tempered LME)
    and min(R,G) from the stored swarm echoes."""
    rows = []
    for rec in ("chal-00286", "chal-00287", "chal-00288", "chal-00289"):
        turns = {t["turn_id"]: t for t in read_jsonl(PROBE / "turns" / f"{rec}.jsonl")}
        echoes = {}
        for e in read_jsonl(PROBE / "echoes" / f"{rec}.jsonl"):
            if "lp" in e:
                echoes[e["turn_id"]] = e
        for tid, t in turns.items():
            e = echoes.get(tid)
            if not e:
                continue
            lp = e["lp"]
            refs = t["refs"]
            k = len(refs)
            eC = [r["lp_empty"] for r in refs]
            tC = [r["lp_thought"] for r in refs]
            mu = st.mean(tC)
            sd = st.stdev(tC) if len(tC) >= 2 else 0.0
            w = max(band_c * sd, band_floor)

            def G(m):
                return min(m - (mu - w), (mu + w) - m)
            kp = t["king"]
            kR = clme([p["lpC_yc_za"] - p["lpC_yc_e"] for p in kp], tau)
            kG = G(kp[0]["lpC_za_x"])
            row = {"kind": t["kind"], "king": {"R": kR, "G": kG, "minRG": min(kR, kG)}}
            ok = True
            for vn in ("filler", "generic", "parrot"):
                try:
                    a = [lp[f"var.{vn}.a.{i}"] - eC[i] for i in range(k)]
                    m = lp[f"var.{vn}.m"]
                except KeyError:
                    ok = False
                    break
                R = clme(a, tau)
                row[vn] = {"R": R, "G": G(m), "minRG": min(R, G(m))}
            if ok:
                rows.append(row)
    out = {"n_turns": len(rows), "band_c": band_c, "by_kind": {}}

    def stats(sub):
        res = {}
        for vn in ("filler", "generic", "parrot"):
            res[vn] = {
                "R_only": {
                    "mean": st.mean(r[vn]["R"] for r in sub),
                    "positive_share": sum(1 for r in sub if r[vn]["R"] > 0) / len(sub),
                    "beats_king_share": sum(1 for r in sub if r[vn]["R"] > r["king"]["R"]) / len(sub),
                    "ge_king_share": sum(1 for r in sub if r[vn]["R"] >= r["king"]["R"]) / len(sub),
                    "mean_minus_king": st.mean(r[vn]["R"] - r["king"]["R"] for r in sub),
                },
                "min_RG": {
                    "mean": st.mean(r[vn]["minRG"] for r in sub),
                    "positive_share": sum(1 for r in sub if r[vn]["minRG"] > 0) / len(sub),
                    "beats_king_share": sum(1 for r in sub if r[vn]["minRG"] > r["king"]["minRG"]) / len(sub),
                    "mean_minus_king": st.mean(r[vn]["minRG"] - r["king"]["minRG"] for r in sub),
                },
            }
        res["king"] = {"R_mean": st.mean(r["king"]["R"] for r in sub),
                       "minRG_mean": st.mean(r["king"]["minRG"] for r in sub), "n": len(sub)}
        return res
    out["ALL"] = stats(rows)
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        if len(sub) >= 20:
            out["by_kind"][kind] = stats(sub)
    return out


# ------------------------------------------------------------------ report
def fmt(x, w=8, p=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return " " * (w - 1) + "-"
    return f"{x:{'+' if sign else ''}{w}.{p}f}"


def pct(x, w=5):
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.0%}"


def median(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return st.median(vals) if vals else None


def write_report(results: list[dict], attacks: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    P = lines.append
    n = len(results)
    recs = [r["rec"] for r in results]
    P(f"No-typicality replay (design S9) — {n} stored wvk-22 verdicts {recs[0]}..{recs[-1]} "
      f"(score_mode sd_min_rga, both sides valid), replayed through the live evalsrv.sdmeter code path")
    P("")
    P("DEFINITIONS")
    P("  z_R    = (R − μ_R)/σ_R: the live centred Reason (does the thought predict which action the teacher takes) in teacher-sd units")
    P("  typ_c  = 2 − |m_c − μ_c|/σ_c: content-masked typicality (does the thought SOUND like the teacher's own thought), floor −12 below 10 content tokens")
    P("  z_A    = (A − μ_A)/σ_A: summed action lift (would the teacher, thinking its own thought, take the miner's action) in sd units")
    P("  μ/σ    = leave-one-out anchors from the k = 3 refs' cross echoes (μ per turn, σ pooled per dialect over the duel)")
    P("  variants: live = min(z_R,typ_c,z_A); (a) = min(z_R,z_A); (b) = z_R; (c) = min(z_R,typ_c). Forfeit = −12 on every variant.")
    P("  margin = paired mean(chal − king) over turns; SE = sd/√n; z = margin/SE; bar = max(2·SE, 0.2); rule_passes = margin > bar")
    P("  would_crown = rule_passes AND the live gates (thought floor, B licence) — the stored verdict's rejection_reason")
    P("  bind = the leg that is the min on a turn (share of valid turns, challenger side)")
    P("  rho = Spearman of per-turn paired differences (variant vs live); SE ratio = SE_variant / SE_live")
    P("  control = teacher vs king: each ref j scored as the miner against the other k−1 refs (its own value left out of μ), paired against the king")
    P("  FLAT = synthetic miner with z_R = 0 and the king's z_A on every turn (content-free thought, king's actions); its typ_c is the floor in the live rule")
    P("  A-MIMIC = synthetic miner with the king's z_R and typ_c and z_A = 0 (king's thought, teacher-mean action)")
    P("")

    # ---- parity
    P("== 1. PARITY (live code path vs stored verdict; tolerance 1e-9 relative) ==")
    ok = sum(1 for r in results if r["parity"]["all"])
    P(f"  {ok}/{n} verdicts reproduce margin, SE, z, decision, teacher-vs-king z, σ_by_dialect and n_loo_by_dialect exactly")
    for r in results:
        p = r["parity"]
        if not p["all"]:
            P(f"  MISMATCH {r['rec']}: " + ", ".join(k for k in ("margin", "se", "z", "wins", "teacher_vs_king_z", "sigma_by_dialect", "n_loo_by_dialect") if not p[k])
              + f"; kinds missing from pinned index = {p['kinds_missing_from_index']}; replayed z {fmt(p['replayed']['z'])} stored {fmt(p['stored']['z'])}")
    P("")

    # ---- per-verdict table
    P("== 2. PER-VERDICT DECISIONS UNDER EACH VARIANT (margin / SE / z / bar-pass / binding leg of the challenger) ==")
    hdr = f"{'rec':11}{'live':>5}|"
    for name in VARIANTS:
        hdr += f" {VARIANT_LABEL[name][:22]:^44}|"
    P(hdr)
    sub = f"{'':16}|"
    for name in VARIANTS:
        sub += f" {'margin':>7} {'SE':>6} {'z':>6} {'pass':>4} {'crown':>5} {'R/G/A bind':>11}|"
    P(sub)
    for r in results:
        line = f"{r['rec']:11}{'WIN' if r['live']['challenger_wins'] else '-':>5}|"
        for name in VARIANTS:
            vv = r["variants"][name]
            bf = vv["bind_frac_challenger"]
            b = "/".join(pct(bf[k], 3).strip() if bf[k] is not None else "-" for k in ("R", "Gc", "A"))
            crown = "CROWN" if vv["would_crown"] else ("gate" if vv["rule_passes"] and not vv["would_crown"] else "-")
            line += f" {fmt(vv['margin'], 7, 3, True)} {fmt(vv['se'], 6, 3)} {fmt(vv['z'], 6, 2, True)} {'yes' if vv['rule_passes'] else 'no':>4} {crown:>5} {b:>11}|"
        P(line)
    P("")

    # ---- crown changes
    P("== 3. CROWN CHANGES vs LIVE ==")
    live_wins = [r["rec"] for r in results if r["live"]["challenger_wins"]]
    P(f"  live crowns ({len(live_wins)}): {', '.join(live_wins)}")
    summary = {}
    for name in VARIANTS:
        if name == "live":
            continue
        lost = [r["rec"] for r in results if r["live"]["challenger_wins"] and not r["variants"][name]["would_crown"]]
        new = [r["rec"] for r in results if not r["live"]["challenger_wins"] and r["variants"][name]["would_crown"]]
        wins = [r["rec"] for r in results if r["variants"][name]["would_crown"]]
        se_r = median([r["variants"][name]["se_ratio_vs_live"] for r in results])
        rho = median([r["variants"][name]["rho_vs_live"] for r in results])
        zr = median([r["variants"][name]["z_ratio_vs_live"] for r in results])
        sign_flips = sum(1 for r in results if (r["variants"][name]["margin"] or 0) * (r["live"]["margin"] or 0) < 0)
        summary[name] = {"crowns": wins, "lost_crowns": lost, "new_crowns": new, "median_se_ratio": se_r,
                         "median_rho": rho, "median_z_ratio": zr, "margin_sign_flips": sign_flips}
        P(f"  {VARIANT_LABEL[name]:24} crowns {len(wins):2d}  lost {len(lost)}: {', '.join(lost) or '-'}  |  new {len(new)}: {', '.join(new) or '-'}"
          f"  |  median SE ratio {fmt(se_r, 5, 2)}  median rho {fmt(rho, 5, 2)}  median z ratio {fmt(zr, 5, 2)}  margin sign flips {sign_flips}/{n}")
    P("")
    P("  per-verdict rho / SE ratio / z ratio vs live:")
    P(f"  {'rec':11}" + "".join(f" {VARIANT_LABEL[nm][:16]:>28}" for nm in VARIANTS if nm != "live"))
    P(f"  {'':11}" + "".join(f" {'rho':>8} {'SEratio':>9} {'zratio':>9}" for nm in VARIANTS if nm != "live"))
    for r in results:
        P(f"  {r['rec']:11}" + "".join(
            f" {fmt(r['variants'][nm]['rho_vs_live'], 8, 3)} {fmt(r['variants'][nm]['se_ratio_vs_live'], 9, 3)} {fmt(r['variants'][nm]['z_ratio_vs_live'], 9, 2)}"
            for nm in VARIANTS if nm != "live"))
    P("")

    # ---- control
    P("== 4. TEACHER-vs-KING CONTROL (paired margin / z; the teacher's held-out reply scored as the miner) ==")
    P(f"  {'rec':11}" + "".join(f" {VARIANT_LABEL[nm][:20]:>22}" for nm in VARIANTS) + f" {'teacher mean (live/a/b/c)':>30}")
    P(f"  {'':11}" + "".join(f" {'margin':>9} {'z':>7}     " for nm in VARIANTS))
    for r in results:
        line = f"  {r['rec']:11}"
        for nm in VARIANTS:
            tk = r["variants"][nm]["teacher_vs_king"]
            line += f" {fmt(tk['margin'], 9, 3, True)} {fmt(tk['z'], 7, 2, True)}     "
        line += "  " + "/".join(fmt(r["variants"][nm]["mean_teacher"], 6, 2, True).strip() for nm in VARIANTS)
        P(line)
    ctrl = {}
    for nm in VARIANTS:
        zs = [r["variants"][nm]["teacher_vs_king"]["z"] for r in results]
        ms = [r["variants"][nm]["teacher_vs_king"]["margin"] for r in results]
        pos = sum(1 for z in zs if z is not None and z > 2)
        neg = sum(1 for z in zs if z is not None and z < -2)
        ctrl[nm] = {"median_z": median(zs), "median_margin": median(ms), "n_z_gt_2": pos, "n_z_lt_minus2": neg,
                    "teacher_mean_median": median([r["variants"][nm]["mean_teacher"] for r in results]),
                    "king_mean_median": median([r["variants"][nm]["mean_king"] for r in results])}
        P(f"  {VARIANT_LABEL[nm]:24} median control z {fmt(ctrl[nm]['median_z'], 7, 2, True)}  median margin {fmt(ctrl[nm]['median_margin'], 7, 3, True)}"
          f"  teacher above king at z>2 on {pos}/{n}, below at z<−2 on {neg}/{n}"
          f"  (median per-turn mean: teacher {fmt(ctrl[nm]['teacher_mean_median'], 6, 2, True)}, king {fmt(ctrl[nm]['king_mean_median'], 6, 2, True)})")
    P("  Note: the live-style control scores the teacher's held-out ref against k−1 = 2 refs while the king is scored against all 3 (the live shadow does the")
    P("  same); the centred LME (R) and LME (A) change scale with k, so the sign of a near-zero control is not readable from this table alone.")
    P("")
    P("  FAIR CONTROL (same k−1 refs on both sides: for each held-out ref j the king is re-scored on pairs i ≠ j with the same μ_j, σ; mean over j per turn):")
    P(f"  {'rec':11}" + "".join(f" {VARIANT_LABEL[nm][:20]:>22}" for nm in VARIANTS))
    P(f"  {'':11}" + "".join(f" {'margin':>9} {'z':>7}     " for nm in VARIANTS))
    for r in results:
        line = f"  {r['rec']:11}"
        for nm in VARIANTS:
            tk = r["variants"][nm]["fair_teacher_vs_king"]
            line += f" {fmt(tk['margin'], 9, 3, True)} {fmt(tk['z'], 7, 2, True)}     "
        P(line)
    for nm in VARIANTS:
        zs = [r["variants"][nm]["fair_teacher_vs_king"]["z"] for r in results]
        ms = [r["variants"][nm]["fair_teacher_vs_king"]["margin"] for r in results]
        pos = sum(1 for z in zs if z is not None and z > 2)
        neg = sum(1 for z in zs if z is not None and z < -2)
        ctrl[nm]["fair"] = {"median_z": median(zs), "median_margin": median(ms), "n_z_gt_2": pos, "n_z_lt_minus2": neg}
        P(f"  {VARIANT_LABEL[nm]:24} FAIR median control z {fmt(median(zs), 7, 2, True)}  median margin {fmt(median(ms), 7, 3, True)}"
          f"  teacher above king at z>2 on {pos}/{n}, below at z<−2 on {neg}/{n}")
    P("")

    # ---- synthetic miners
    P("== 5. SYNTHETIC MINERS FROM THE KING'S OWN LEGS (paired margin vs the real king, z; 'pass' = clears the crown bar) ==")
    P(f"  {'rec':11}" + "".join(f" {'FLAT ' + VARIANT_LABEL[nm][:14]:>24}" for nm in VARIANTS) + "".join(f" {'A-MIMIC ' + VARIANT_LABEL[nm][:11]:>24}" for nm in ("live", "a_no_typ")))
    for r in results:
        line = f"  {r['rec']:11}"
        for nm in VARIANTS:
            s = r["synthetic"]["flat"][nm]
            line += f" {fmt(s['margin'], 9, 3, True)} z{fmt(s['z'], 8, 1, True)}{' P' if s['rule_passes'] else '  '}   "
        for nm in ("live", "a_no_typ"):
            s = r["synthetic"]["a_mimic"][nm]
            line += f" {fmt(s['margin'], 9, 3, True)} z{fmt(s['z'], 8, 1, True)}{' P' if s['rule_passes'] else '  '}   "
        P(line)
    synth = {}
    for kind in ("flat", "a_mimic"):
        synth[kind] = {}
        for nm in VARIANTS:
            ms = [r["synthetic"][kind][nm]["margin"] for r in results]
            zs = [r["synthetic"][kind][nm]["z"] for r in results]
            passes = sum(1 for r in results if r["synthetic"][kind][nm]["rule_passes"])
            beats = sum(1 for m in ms if m is not None and m > 0)
            synth[kind][nm] = {"median_margin": median(ms), "median_z": median(zs), "n_pass": passes, "n_margin_positive": beats}
            P(f"  {kind.upper():8} {VARIANT_LABEL[nm]:24} median margin {fmt(median(ms), 7, 3, True)}  median z {fmt(median(zs), 7, 1, True)}"
              f"  margin > 0 on {beats}/{n}  clears the bar on {passes}/{n}")
    P("")

    # ---- per dialect
    P("== 6. PER-DIALECT SEPARATION (paired differences pooled over all verdicts; mean / z; n turns) ==")
    P("  T−K = live-style teacher − king control (mean, z); fair z = same-k control; C−K z = challenger − king pooled over every challenger;")
    P("  Gc-bind = share of valid turns (both sides) on which typicality is the binding leg under live")
    per_dialect = {}
    P(f"  {'dialect':14}{'n':>7}{'Gc-bind':>9}" + "".join(f" {VARIANT_LABEL[nm][:16]:>34}" for nm in VARIANTS))
    P(f"  {'':30}" + "".join(f" {'T−K mean':>9} {'z':>6} {'fair z':>7} {'C−K z':>8}" for nm in VARIANTS))
    for kind in DIALECTS:
        row = {}
        line = f"  {kind:14}"
        n_turns = sum(len(r["variants"]["live"]["per_dialect"][kind]["diffs_tk"]) for r in results)
        line += f"{n_turns:>7}"
        gc = [r["variants"]["live"]["per_dialect"][kind].get("gc_bind") for r in results]
        gc = [g for g in gc if g is not None]
        line += f"{pct(st.mean(gc), 8) if gc else '       -'} "
        for nm in VARIANTS:
            tk = [dv for r in results for dv in r["variants"][nm]["per_dialect"][kind]["diffs_tk"]]
            fk = [dv for r in results for dv in r["variants"][nm]["per_dialect"][kind]["diffs_fk"]]
            ck = [dv for r in results for dv in r["variants"][nm]["per_dialect"][kind]["diffs_ck"]]
            s_tk, s_fk, s_ck = pooled_stats(tk), pooled_stats(fk), pooled_stats(ck)
            row[nm] = {"teacher_minus_king": {k: s_tk[k] for k in ("n", "mean", "se", "z")},
                       "fair_teacher_minus_king": {k: s_fk[k] for k in ("n", "mean", "se", "z")},
                       "chal_minus_king": {k: s_ck[k] for k in ("n", "mean", "se", "z")}}
            line += f" {fmt(s_tk['mean'], 9, 3, True)} {fmt(s_tk['z'], 6, 1, True)} {fmt(s_fk['z'], 7, 1, True)} {fmt(s_ck['z'], 8, 1, True)}"
        row["gc_bind_live"] = st.mean(gc) if gc else None
        row["n_turns"] = n_turns
        per_dialect[kind] = row
        P(line)
    P("  separation kept = fair control z under the variant ÷ fair control z under live, per dialect:")
    for kind in DIALECTS:
        zl = per_dialect[kind]["live"]["fair_teacher_minus_king"]["z"]
        parts = []
        for nm in ("a_no_typ", "b_r_only", "c_no_a"):
            zv = per_dialect[kind][nm]["fair_teacher_minus_king"]["z"]
            parts.append(f"{VARIANT_LABEL[nm][:4]} {fmt(zv, 6, 1, True)} ({fmt(zv / zl if zl else None, 5, 2)}x)")
        P(f"  {kind:14} live fair control z {fmt(zl, 7, 1, True)}   " + "   ".join(parts))
    P("")

    # ---- attacks
    P("== 7. ATTACK ARMS ==")
    P("  (i) 4 stored wvk-11 duels (research/results/frontier_rule_probe, canonical rendering, teacher echoes), king's thought replaced:")
    P("      filler = king thought + fixed filler suffix (the reign-41 attack); generic = fixed task-agnostic thought; parrot = king thought + frontier action pasted")
    P(f"      re-scored here under teacher R-only (centred LME, tau 0.03) and min(R,G) (band_c {attacks['band_c']}); n = {attacks['n_turns']} turns")
    P(f"      {'arm':8} {'R-only mean':>12} {'R>0':>6} {'R>king':>7} {'R≥king':>7} {'R−king':>9} | {'minRG mean':>11} {'minRG>0':>8} {'>king':>6} {'−king':>9}")
    for vn in ("filler", "generic", "parrot"):
        a = attacks["ALL"][vn]
        P(f"      {vn:8} {fmt(a['R_only']['mean'], 12, 4, True)} {pct(a['R_only']['positive_share'], 6)} {pct(a['R_only']['beats_king_share'], 7)} {pct(a['R_only']['ge_king_share'], 7)} {fmt(a['R_only']['mean_minus_king'], 9, 4, True)} |"
          f" {fmt(a['min_RG']['mean'], 11, 4, True)} {pct(a['min_RG']['positive_share'], 8)} {pct(a['min_RG']['beats_king_share'], 6)} {fmt(a['min_RG']['mean_minus_king'], 9, 4, True)}")
    P(f"      king: R mean {fmt(attacks['ALL']['king']['R_mean'], 7, 4, True)}, minRG mean {fmt(attacks['ALL']['king']['minRG_mean'], 7, 4, True)}")
    for kind, a in attacks["by_kind"].items():
        P(f"      [{kind}, n={a['king']['n']}] filler R>0 {pct(a['filler']['R_only']['positive_share'], 4)} R≥king {pct(a['filler']['R_only']['ge_king_share'], 4)} | "
          f"generic R>0 {pct(a['generic']['R_only']['positive_share'], 4)} R>king {pct(a['generic']['R_only']['beats_king_share'], 4)} | "
          f"parrot R>king {pct(a['parrot']['R_only']['beats_king_share'], 4)} minRG>king {pct(a['parrot']['min_RG']['beats_king_share'], 4)}")
    P("      As published: research/results/frontier_rule_probe/report.txt Test 4 (ALL): minRG filler ≤ 0 on 91 %, generic ≤ 0 on 99 %, parrot < king on 60 %;")
    P("      the 'RF' row (centred Reason against FRONTIER refs, the R-only-style leg) has filler ≤ 0 on only 8 % (positive on 92 %) and parrot < king on 29 %")
    P("      (parrot ≥ king on 71 %). research/results/frontier_arbiter/rule/report.txt Test 3 (ALL): live_c4 filler ≤ king 90 % / generic 97 % / parrot 58 %;")
    P("      exact/G|RF filler ≤ king 70 % / generic 66 % / parrot 38 %; V (signed contrast) 49 / 48 / 27 %.")
    P("  (ii) wvk-22 verdicts, synthetic FLAT and A-MIMIC miners — section 5.")
    P("")

    # ---- verdict
    P("== 8. READING ==")
    a, b, c = summary["a_no_typ"], summary["b_r_only"], summary["c_no_a"]
    P(f"  Dropping typicality (a): {len(a['lost_crowns'])} of {len(live_wins)} live crowns lost, {len(a['new_crowns'])} new crowns; median SE ratio {fmt(a['median_se_ratio'], 4, 2)}, "
      f"rho {fmt(a['median_rho'], 4, 2)}; control z median {fmt(ctrl['live']['median_z'], 5, 1, True)} → {fmt(ctrl['a_no_typ']['median_z'], 5, 1, True)}"
      f" (fair {fmt(ctrl['live']['fair']['median_z'], 5, 1, True)} → {fmt(ctrl['a_no_typ']['fair']['median_z'], 5, 1, True)}).")
    P(f"  R only (b): {len(b['lost_crowns'])} lost / {len(b['new_crowns'])} new; SE ratio {fmt(b['median_se_ratio'], 4, 2)}, rho {fmt(b['median_rho'], 4, 2)}; control z → {fmt(ctrl['b_r_only']['median_z'], 5, 1, True)}"
      f" (fair {fmt(ctrl['b_r_only']['fair']['median_z'], 5, 1, True)}).")
    P(f"  No A (c): {len(c['lost_crowns'])} lost / {len(c['new_crowns'])} new; SE ratio {fmt(c['median_se_ratio'], 4, 2)}, rho {fmt(c['median_rho'], 4, 2)}; control z → {fmt(ctrl['c_no_a']['median_z'], 5, 1, True)}"
      f" (fair {fmt(ctrl['c_no_a']['fair']['median_z'], 5, 1, True)}).")
    fl = synth["flat"]
    P(f"  FLAT miner vs king: live median margin {fmt(fl['live']['median_margin'], 6, 2, True)} (typ floor binds), (a) {fmt(fl['a_no_typ']['median_margin'], 6, 2, True)} "
      f"(clears bar on {fl['a_no_typ']['n_pass']}/{n}), (b) {fmt(fl['b_r_only']['median_margin'], 6, 2, True)} (clears bar on {fl['b_r_only']['n_pass']}/{n}).")
    am = synth["a_mimic"]
    P(f"  A-MIMIC vs king: live {fmt(am['live']['median_margin'], 6, 2, True)}, (a) {fmt(am['a_no_typ']['median_margin'], 6, 2, True)} (clears bar on {am['a_no_typ']['n_pass']}/{n}).")
    gc_first = st.mean(r["variants"]["live"]["bind_frac_king"]["Gc"] for r in results[:5])
    gc_last = st.mean(r["variants"]["live"]["bind_frac_king"]["Gc"] for r in results[-5:])
    fair_first = st.mean(r["variants"]["live"]["fair_teacher_vs_king"]["z"] for r in results[:5])
    fair_last = st.mean(r["variants"]["live"]["fair_teacher_vs_king"]["z"] for r in results[-5:])
    P(f"  Drift over the window: typicality binds on the king's turns {gc_first:.0%} (first 5 verdicts) → {gc_last:.0%} (last 5); "
      f"live fair control z {fair_first:+.1f} → {fair_last:+.1f} — successive kings moved into and past the teacher's own typicality.")
    heavy = sum(1 for r in results if (r["variants"]["b_r_only"]["se_ratio_vs_live"] or 0) > 1)
    P(f"  (b) has no min to cap positive z_R outliers (σ_R is tiny on tool_call/terminus_json), so its SE exceeds live's on {heavy}/{n} verdicts; "
      f"forfeit_sd = −12 was calibrated on the live turn-score sd and would need re-setting for (a)/(b), whose valid-turn sd is ~0.3–0.6× live.")

    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    slim = []
    for r in results:
        rr = json.loads(json.dumps(r))
        for nm in rr["variants"]:
            for kind in rr["variants"][nm]["per_dialect"]:
                for key in ("diffs_ck", "diffs_tk", "diffs_fk"):
                    rr["variants"][nm]["per_dialect"][kind].pop(key, None)
        slim.append(rr)
    (out_dir / "report.json").write_text(json.dumps({
        "records": recs, "n": n, "parity_ok": ok, "summary": summary, "control": ctrl,
        "synthetic": synth, "per_dialect": per_dialect, "attack_arms_old_duels": attacks,
        "verdicts": slim}, indent=1, default=str))


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", default="", help="comma-separated chal ids (default: every cached wvk-22 verdict)")
    ap.add_argument("--index-dir", default="/tmp/no_typicality/index")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    recs = [r.strip() for r in args.recs.split(",") if r.strip()] or wvk22_records()
    kinds = KindIndex(Path(args.index_dir))
    results = []
    for rec in recs:
        r = replay(rec, kinds)
        results.append(r)
        v = r["variants"]["live"]
        print(f"{rec}: parity={'ok' if r['parity']['all'] else 'MISMATCH'} live z {fmt(v['z'], 6, 2, True)} "
              f"(a) z {fmt(r['variants']['a_no_typ']['z'], 6, 2, True)} (b) z {fmt(r['variants']['b_r_only']['z'], 6, 2, True)} "
              f"(c) z {fmt(r['variants']['c_no_a']['z'], 6, 2, True)}  control z live {fmt(v['teacher_vs_king']['z'], 6, 1, True)} "
              f"(a) {fmt(r['variants']['a_no_typ']['teacher_vs_king']['z'], 6, 1, True)}", flush=True)
    attacks = attack_arms_r_only()
    write_report(results, attacks, Path(args.out))
    print(f"wrote {Path(args.out) / 'report.txt'}")


if __name__ == "__main__":
    main()
