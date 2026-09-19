#!/usr/bin/env python
"""Curriculum ledger: king performance per (verdict, side, turn).

Inputs (all public; local copies on the validator box):
  * history.jsonl        verdict / crowned / crown_revoked rows
                         (public: https://affine.io/data/history_full.jsonl.gz)
  * evals/{cid}.json.gz  the duel artifact: turn_ids, teacher_refs, king_rows,
                         challenger_rows, request, verdict, slice
  * the corpus index of the manifest each verdict stamps
                         (slice.manifest_sha256 -> corpus/manifests/{sha}.json
                         -> views/duel_turns@v4/index/*.parquet), plus the view
                         chunks for rollout -> harness

One row per (challenge_id, side, turn_id) for every DRAWN turn; per-turn
scores are recomputed from the stored `pairs` with the verdict's own
`duel_params` through affine.score (the same code the duel ran). No
randomness, no clock in the hashed outputs, sorted rows, sha256 over the
canonical row stream -- anyone can rebuild it and must get the same sha.

    python ops/curriculum/ledger.py --out affine/state/curriculum/ledger
    python ops/curriculum/ledger.py --check <ledger_sha256>
"""

from __future__ import annotations

import argparse
import glob
import re
import gzip
import hashlib
import json
import os
import statistics as st
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA_BASE, DEFAULT_GROUP, EVALS_DIR, HISTORY_PATH, INDEX_CACHE, LOCAL_CHUNKS,
    PUBLIC_EVALS_BASE, STATE_DIR, base_stratum, cached_public, canonical_json,
    clean_float, depth_bin, fetch_bytes, group_of_stratum, index_table_for_manifest,
    load_curriculum_cfg, load_sources_toml, load_src2grp, load_static_mix, log,
    manifest_by_sha, norm_text, sha256_bytes, write_json, write_parquet, wvk_of,
)

from affine import score as S  # noqa: E402
from evalsrv import amatch  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(TOKEN_RE.findall(a)), set(TOKEN_RE.findall(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def action_divergence(y_side: str | None, ref_ys: list[str], kind: str | None
                      ) -> tuple[float | None, float | None]:
    """(1 − soft A_match, 1 − exact A_match) for one side on one turn. Soft =
    mean token-Jaccard between the side's normalised action and each valid
    reference action (exact agreement is ≈ 0 on bash/tool turns, so the
    soft form carries the signal). None when the dialect has no normal
    form (`text`) or nothing parses."""
    refs = [n for n in (amatch.norm_action(y, kind) for y in ref_ys) if n is not None]
    mine = amatch.norm_action(y_side, kind)
    if mine is None or not refs:
        return None, None
    soft = sum(token_jaccard(mine, r) for r in refs) / len(refs)
    exact = sum(1 for r in refs if r == mine) / len(refs)
    return 1.0 - soft, 1.0 - exact

ROW_SORT = ("challenge_id", "side", "turn_id")
LEDGER_VERSION = 4   # 4 (2026-09-16, rule v2): rows gain div_action / div_action_exact / div_score / b_lift / teacher_own_lift (ROWS CHANGE -> new ledger sha); rollup + div_* means


# -- history --------------------------------------------------------------------
def load_history_rows(src: str) -> list[dict]:
    if src.startswith("http"):
        raw = fetch_bytes(src)
        text = gzip.decompress(raw).decode("utf-8") if src.endswith(".gz") else raw.decode("utf-8")
    else:
        p = Path(src)
        text = gzip.open(p, "rt", encoding="utf-8").read() if p.suffix == ".gz" \
            else p.read_text(encoding="utf-8")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [r for r in rows if r.get("event") in ("verdict", "crowned", "crown_revoked")]


def reign_chain(rows: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """revision -> {reign, hotkey, crowned_at, revoked}; crowning cid -> same."""
    by_rev: dict[str, dict] = {}
    by_cid: dict[str, dict] = {}
    for r in rows:
        if r.get("event") != "crowned":
            continue
        rec = {"reign": int(r.get("reign_number") or 0), "hotkey": r.get("hotkey") or "",
               "crowned_at": r.get("at"), "revoked": False, "revision": r.get("revision")}
        by_rev[str(r.get("revision"))] = rec
        by_cid[str(r.get("challenge_id"))] = rec
    for r in rows:
        if r.get("event") == "crown_revoked":
            rec = by_rev.get(str(r.get("revision")))
            if rec is None:
                rec = {"reign": int(r.get("reign_number") or 0), "hotkey": r.get("hotkey") or "",
                       "crowned_at": r.get("at"), "revoked": True, "revision": r.get("revision")}
                by_rev[str(r.get("revision"))] = rec
                by_cid[str(r.get("challenge_id"))] = rec
            rec["revoked"] = True
    return by_rev, by_cid


# -- artifacts ------------------------------------------------------------------
def artifact_bytes(evals: str, cid: str) -> bytes | None:
    if evals.startswith("http"):
        try:
            return fetch_bytes(f"{evals.rstrip('/')}/{cid}.json.gz")
        except Exception as e:  # noqa: BLE001
            log(f"{cid}: fetch failed ({e})")
            return None
    p = Path(evals) / f"{cid}.json.gz"
    return p.read_bytes() if p.is_file() else None


def list_local_cids(evals: str) -> list[str]:
    if evals.startswith("http"):
        return []
    return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(evals, "chal-*.json.gz")))


# -- rollout -> harness (view chunks) --------------------------------------------
class RolloutMeta:
    """rollout_id -> (policy_id, harness) from the view chunks the stamped
    manifests list. Parsed once per chunk, cached as small JSON files."""

    def __init__(self, cache_dir: Path, base: str = DATA_BASE):
        self.cache_dir = cache_dir / "rollouts"
        self.base = base
        self.meta: dict[str, tuple[str, str]] = {}
        self.loaded: set[str] = set()

    def add_manifest(self, manifest: dict) -> None:
        for sh in manifest.get("shards", []):
            key = str(sh.get("key") or "")
            if not key.startswith("views/") or key in self.loaded:
                continue
            self.loaded.add(key)
            self.meta.update(self._chunk(key, sh.get("sha256")))

    def _chunk(self, key: str, sha: str | None) -> dict[str, tuple[str, str]]:
        name = os.path.basename(key)
        cpath = self.cache_dir / (name + ".meta.json")
        if cpath.is_file():
            return {k: tuple(v) for k, v in json.loads(cpath.read_text()).items()}
        local = LOCAL_CHUNKS / name
        if not local.is_file():
            local = cached_public(key, sha256=sha, gz_payload_sha=True, base=self.base,
                                 cache_dir=INDEX_CACHE)
        out: dict[str, tuple[str, str]] = {}
        with gzip.open(local, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pol = r.get("policy") or {}
                out[str(r["rollout_id"])] = (str(pol.get("id") or ""), str(pol.get("harness") or ""))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: list(v) for k, v in sorted(out.items())}))
        os.replace(tmp, cpath)
        log(f"rollout meta: parsed {name} ({len(out)} rollouts)")
        return out


# -- per-turn scoring -----------------------------------------------------------
def side_terms(row: dict | None, dp: dict) -> dict:
    """One side's recomputed terms on one turn (None row = no row written =
    the teacher produced zero references there; both sides lose the slot)."""
    if row is None:
        return {"scored": False, "forfeit": None, "turn_score": None, "r_leg": None,
                "g_leg": None, "b_pass": None, "thought_chars": None,
                "b_king_lift": None, "y_side": None}
    tau = dp.get("tau", 0.03)
    band_c = dp.get("band_c", 2.0)
    band_floor = dp.get("band_floor", 0.002)
    floor = dp.get("forfeit_turn_score", -0.1)
    mode = dp.get("score_mode", "min_rg")
    norm = dp.get("action_norm_bytes", None)
    if S.is_forfeit(row):
        return {"scored": floor is not None, "forfeit": True, "turn_score": floor,
                "r_leg": None, "g_leg": None, "b_pass": None, "thought_chars": None,
                "b_king_lift": None, "y_side": None}
    pairs = row["pairs"]
    score = S.side_turn_score(row, tau, mode, band_c, band_floor, floor, norm)
    r = S.centered_reason(pairs, tau)
    g = S.grounding(pairs, band_c, band_floor)
    bp = S.b_gate_pass(pairs[0])
    return {"scored": True, "forfeit": False, "turn_score": score, "r_leg": r, "g_leg": g,
            "b_pass": bp, "thought_chars": len((pairs[0].get("z_a") or "").strip()),
            "b_king_lift": S.teacher_causality(pairs[0]), "y_side": pairs[0].get("y_a")}


# -- build ----------------------------------------------------------------------
class IndexView:
    """turn_id -> meta for the manifest a verdict stamps (one table at a time;
    verdicts are processed in challenge order so the manifest changes rarely)."""

    def __init__(self, base: str, meta: RolloutMeta):
        self.base = base
        self.meta = meta
        self.sha: str | None = None
        self.table: pa.Table | None = None
        self.manifest: dict | None = None
        self.epoch: int | None = None

    def load(self, sha: str) -> None:
        if sha == self.sha:
            return
        self.manifest = manifest_by_sha(sha, self.base)
        self.table = index_table_for_manifest(self.manifest, self.base)
        self.sha = sha
        self.epoch = int(self.manifest.get("corpus_epoch") or 0)
        self.meta.add_manifest(self.manifest)
        log(f"index: manifest {sha[:12]} epoch {self.epoch} ({self.table.num_rows} turns)")

    def lookup(self, turn_ids: list[str]) -> dict[str, dict]:
        assert self.table is not None
        mask = pc.is_in(self.table.column("turn_id"), value_set=pa.array(turn_ids, pa.string()))
        sub = self.table.filter(mask).to_pylist()
        return {r["turn_id"]: r for r in sub}


def build_rows(history_rows: list[dict], evals: str, *, since: str, until: str,
               cfg: dict, base: str, index_cache: Path, max_verdicts: int | None = None
               ) -> tuple[list[dict], dict]:
    raw = load_sources_toml()
    mix = load_static_mix(raw)
    src2grp = load_src2grp(raw)
    groups = set(mix)
    by_rev, by_cid = reign_chain(history_rows)
    # A crowning duel is written as one `crowned` row (with its verdict), not
    # as a `verdict` row; a revoked crown keeps its verdict on the
    # `crown_revoked` row. All three carry the scored verdict.
    verdicts = {r["challenge_id"]: r for r in history_rows
                if isinstance(r.get("verdict"), dict) and str(r.get("challenge_id") or "").startswith("chal-")}
    cids = sorted(c for c in verdicts if since <= c <= until)
    if not evals.startswith("http"):
        have = set(list_local_cids(evals))
        cids = [c for c in cids if c in have]
    if max_verdicts:
        cids = cids[-max_verdicts:]
    meta = RolloutMeta(index_cache, base)
    index = IndexView(base, meta)
    rows: list[dict] = []
    scored_cids: list[str] = []
    inputs: dict = {"evals_sha256": {}, "manifests": {}}
    n_join = n_drawn = 0
    for cid in cids:
        hv = verdicts[cid]
        v = hv["verdict"]
        blob = artifact_bytes(evals, cid)
        if blob is None:
            log(f"{cid}: no artifact; skipped")
            continue
        art = json.loads(gzip.decompress(blob))
        if not art.get("king_rows") or not art.get("challenger_rows"):
            log(f"{cid}: rejected before scoring ({v.get('rejection_reason')}); skipped")
            continue
        inputs["evals_sha256"][cid] = sha256_bytes(blob)
        sl = art.get("slice") or v.get("slice") or {}
        msha = str(sl.get("manifest_sha256") or "")
        if not msha:
            log(f"{cid}: no manifest sha on the slice; skipped")
            continue
        index.load(msha)
        inputs["manifests"][msha] = index.epoch
        dp = v.get("duel_params") or {}
        req = art.get("request") or {}
        king_rev = str(req.get("king_revision") or "")
        reign = by_rev.get(king_rev) or {}
        z = v.get("z")
        near_king = (v.get("rejection_reason") in (None, "")) and z is not None and abs(z) < cfg["near_king_z"]
        crowning = by_cid.get(cid)
        chal_becomes_king = bool(v.get("challenger_wins")) and crowning is not None
        k_rows = {r["turn_id"]: r for r in art["king_rows"]}
        c_rows = {r["turn_id"]: r for r in art["challenger_rows"]}
        refs = art.get("teacher_refs") or {}
        turn_ids = list(art.get("turn_ids") or [])
        metas = index.lookup(turn_ids)
        common = {
            "challenge_id": cid, "at": hv.get("at"), "wvk": wvk_of(hv.get("at")),
            "manifest_sha256": msha, "corpus_epoch": index.epoch,
            "hotkey": hv.get("hotkey") or "", "revision": str(hv.get("revision") or ""),
            "z": clean_float(z), "margin": clean_float(v.get("margin")),
            "rejection_reason": v.get("rejection_reason"),
            "king_digest_at_time": king_rev, "reign": reign.get("reign"),
            "reign_revoked": bool(reign.get("revoked", False)),
            "gated_near_king": bool(near_king),
        }
        scored_cids.append(cid)
        for tid in turn_ids:
            n_drawn += 1
            m = metas.get(tid)
            if m is not None:
                n_join += 1
            m = m or {}
            ref = refs.get(tid) or []
            ys = {norm_text(r.get("y")) for r in ref}
            n_refs = len(ref)
            identical = n_refs >= 2 and len(ys) == 1
            live = n_refs >= 2 and not identical
            stratum = str(m.get("stratum") or "")
            src = m.get("stratum_src")
            base_s = base_stratum(stratum, src) if m else ""
            group = group_of_stratum(base_s, m.get("source"), src2grp, groups) if m else ""
            pol_id, harness = meta.meta.get(str(m.get("rollout_id") or ""), ("", ""))
            dbin = depth_bin(m.get("n_prefix_chars")) if m else ""
            cell = "|".join([group, str(m.get("source") or ""), harness,
                             str(m.get("action_kind") or ""), dbin]) if m else ""
            kt = side_terms(k_rows.get(tid), dp)
            ct = side_terms(c_rows.get(tid), dp)
            own = [r["lp_own"] - r["lp_empty"] for r in ref
                   if r.get("lp_own") is not None and r.get("lp_empty") is not None]
            t_own = (sum(own) / len(own)) if own else None
            kind = str(m.get("action_kind") or "") or None
            ref_ys = [r.get("y") or "" for r in ref]
            d = (ct["turn_score"] - kt["turn_score"]) if (kt["scored"] and ct["scored"]) else None
            turn_common = {
                "turn_id": tid, "joined": m != {}, "stratum": stratum, "base_stratum": base_s,
                "cell": cell, "group": group, "source": str(m.get("source") or ""),
                "harness": harness, "policy_id": pol_id,
                "action_kind": str(m.get("action_kind") or ""), "depth_bin": dbin,
                "prefix_chars": int(m.get("n_prefix_chars") or 0) if m else None,
                "n_refs_valid": n_refs, "refs_identical": bool(identical), "live": bool(live),
                "d": clean_float(d),
            }
            for side, terms, is_king in (("king", kt, not reign.get("revoked", False)),
                                         ("challenger", ct, chal_becomes_king and not crowning.get("revoked", False))):
                div_soft, div_exact = (action_divergence(terms["y_side"], ref_ys, kind)
                                       if terms["scored"] and not terms["forfeit"] else (None, None))
                b_lift = terms["b_king_lift"]
                # (c) reference-implied score deficit: the teacher's own thought
                # explains its own action (lp_own − lp_empty, per byte) more than
                # the side's thought explains the side's action (B); ≥ 0
                div_score = (max(0.0, t_own - b_lift) if (t_own is not None and b_lift is not None
                                                          and terms["scored"] and not terms["forfeit"]) else None)
                rows.append({**common, **turn_common, "side": side, "is_king_row": bool(is_king),
                             "div_action": clean_float(div_soft), "div_action_exact": clean_float(div_exact),
                             "div_score": clean_float(div_score), "b_lift": clean_float(b_lift),
                             "teacher_own_lift": clean_float(t_own),
                             "scored": bool(terms["scored"]), "forfeit": terms["forfeit"],
                             "turn_score": clean_float(terms["turn_score"]),
                             "r_leg": clean_float(terms["r_leg"]), "g_leg": clean_float(terms["g_leg"]),
                             "b_pass": terms["b_pass"], "thought_chars": terms["thought_chars"],
                             "miss": None})
        log(f"{cid}: {len(turn_ids)} turns, joined {sum(1 for t in turn_ids if t in metas)}, "
            f"king {'revoked ' if reign.get('revoked') else ''}reign {reign.get('reign')}"
            f"{' (crowning duel)' if chal_becomes_king else ''}")
    stats = {"n_verdicts": len(scored_cids), "first_challenge_id": scored_cids[0] if scored_cids else None,
             "last_challenge_id": scored_cids[-1] if scored_cids else None,
             "n_drawn": n_drawn, "n_joined": n_join,
             "join_rate": clean_float(n_join / n_drawn) if n_drawn else None,
             "scored_cids": scored_cids}
    return rows, {"inputs": inputs, "stats": stats}


def apply_theta(rows: list[dict], theta_pct: float) -> float | None:
    king_live = [r["turn_score"] for r in rows
                 if r["is_king_row"] and r["scored"] and r["live"] and r["turn_score"] is not None]
    if not king_live:
        for r in rows:
            r["miss"] = bool(r["forfeit"]) if r["scored"] else None
        return None
    theta = float(np.percentile(np.array(sorted(king_live), dtype=float), 100 * theta_pct))
    theta = clean_float(theta)
    for r in rows:
        if not r["scored"]:
            r["miss"] = None
        else:
            r["miss"] = bool(r["forfeit"]) or (bool(r["live"]) and r["turn_score"] < theta)
    return theta


def sort_rows(rows: list[dict]) -> None:
    rows.sort(key=lambda r: tuple(r[k] for k in ROW_SORT))


def rows_sha256(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(canonical_json(r))
        h.update(b"\n")
    return h.hexdigest()


# -- rollups --------------------------------------------------------------------
def rollup(rows: list[dict], scored_cids: list[str], cfg: dict) -> list[dict]:
    """Per key (stratum / cell / group): decayed observation mass n (king rows
    only), miss rate M, live rate S, near-king paired gap Dbar, recurrence
    over the last `recurrence_window_verdicts` verdicts (all draws)."""
    half = float(cfg["half_life_verdicts"])
    order = {c: i for i, c in enumerate(scored_cids)}
    newest = len(scored_cids) - 1
    recent = set(scored_cids[-int(cfg["recurrence_window_verdicts"]):])
    acc: dict[tuple[str, str], dict] = {}

    def slot(level: str, key: str) -> dict:
        k = (level, key)
        if k not in acc:
            acc[k] = {"level": level, "key": key, "n_obs": 0, "n_w": 0.0, "miss_w": 0.0,
                      "live_w": 0.0, "forfeit_w": 0.0, "score_w": 0.0, "n_d": 0, "d_w": 0.0,
                      "livevalid_w": 0.0, "livemiss_w": 0.0, "livescore_w": 0.0, "live_scores": [],
                      "dplus_sum": 0.0, "act_w": 0.0, "act_sum": 0.0, "act_exact_sum": 0.0,
                      "sc_w": 0.0, "sc_sum": 0.0,
                      "dw_sum": 0.0, "draws_50": 0, "turns_50": defaultdict(int),
                      "n_draws_total": 0, "turns_all": set(), "group": "", "cell": ""}
        return acc[k]

    for r in rows:
        if not r["joined"]:
            continue
        keys = (("stratum", r["base_stratum"]), ("cell", r["cell"]), ("group", r["group"]))
        age = newest - order[r["challenge_id"]]
        a = 0.5 ** (age / half)
        for level, key in keys:
            s = slot(level, key)
            s["group"] = r["group"]
            if level == "stratum":
                s["cell"] = r["cell"]
            if r["side"] == "king":       # draws are per turn, count once
                s["n_draws_total"] += 1
                s["turns_all"].add(r["turn_id"])
                if r["challenge_id"] in recent:
                    s["draws_50"] += 1
                    s["turns_50"][r["turn_id"]] += 1
            if r["is_king_row"] and r["scored"]:
                s["n_obs"] += 1
                s["n_w"] += a
                s["miss_w"] += a * (1.0 if r["miss"] else 0.0)
                s["live_w"] += a * (1.0 if r["live"] else 0.0)
                s["forfeit_w"] += a * (1.0 if r["forfeit"] else 0.0)
                s["score_w"] += a * float(r["turn_score"])
                if r["div_action"] is not None:
                    s["act_w"] += a
                    s["act_sum"] += a * float(r["div_action"])
                    s["act_exact_sum"] += a * float(r["div_action_exact"] or 0.0)
                if r["div_score"] is not None:
                    s["sc_w"] += a
                    s["sc_sum"] += a * float(r["div_score"])
                if r["live"] and not r["forfeit"]:
                    # "answers badly": live turn, answered, scored under theta
                    s["livevalid_w"] += a
                    s["livemiss_w"] += a * (1.0 if r["miss"] else 0.0)
                    s["livescore_w"] += a * float(r["turn_score"])
                    s["live_scores"].append(float(r["turn_score"]))
            if r["side"] == "challenger" and r["gated_near_king"] and r["d"] is not None:
                s["n_d"] += 1
                s["d_w"] += a
                s["dw_sum"] += a * float(r["d"])
                s["dplus_sum"] += a * max(0.0, float(r["d"]))
    out = []
    for (level, key), s in sorted(acc.items()):
        n = s["n_w"]
        out.append({
            "level": level, "key": key, "group": s["group"], "cell": s["cell"] if level == "stratum" else "",
            "n_obs": s["n_obs"], "n_w": clean_float(n),
            "M": clean_float(s["miss_w"] / n) if n > 0 else None,
            "S": clean_float(s["live_w"] / n) if n > 0 else None,
            "forfeit_rate": clean_float(s["forfeit_w"] / n) if n > 0 else None,
            "mean_score": clean_float(s["score_w"] / n) if n > 0 else None,
            "n_d": s["n_d"], "n_d_w": clean_float(s["d_w"]),
            "Dbar": clean_float(s["dw_sum"] / s["d_w"]) if s["d_w"] > 0 else None,
            "Dbar_plus": clean_float(s["dplus_sum"] / s["d_w"]) if s["d_w"] > 0 else None,
            "n_act_w": clean_float(s["act_w"]),
            "div_action": clean_float(s["act_sum"] / s["act_w"]) if s["act_w"] > 0 else None,
            "div_action_exact": clean_float(s["act_exact_sum"] / s["act_w"]) if s["act_w"] > 0 else None,
            "n_sc_w": clean_float(s["sc_w"]),
            "div_score": clean_float(s["sc_sum"] / s["sc_w"]) if s["sc_w"] > 0 else None,
            "M_live": clean_float(s["livemiss_w"] / s["livevalid_w"]) if s["livevalid_w"] > 0 else None,
            "mean_live_score": clean_float(s["livescore_w"] / s["livevalid_w"]) if s["livevalid_w"] > 0 else None,
            "q25_live_score": clean_float(float(np.percentile(np.array(sorted(s["live_scores"])), 25)))
            if len(s["live_scores"]) >= 4 else None,
            "n_live_answered": len(s["live_scores"]),
            "draws_50": s["draws_50"], "distinct_turns_50": len(s["turns_50"]),
            "max_turn_draws_50": max(s["turns_50"].values()) if s["turns_50"] else 0,
            "n_draws_total": s["n_draws_total"], "distinct_turns_total": len(s["turns_all"]),
        })
    return out


# -- io -------------------------------------------------------------------------
ROW_SCHEMA = pa.schema([
    ("challenge_id", pa.string()), ("side", pa.string()), ("turn_id", pa.string()),
    ("at", pa.string()), ("wvk", pa.int32()), ("manifest_sha256", pa.string()),
    ("corpus_epoch", pa.int32()), ("hotkey", pa.string()), ("revision", pa.string()),
    ("z", pa.float64()), ("margin", pa.float64()), ("rejection_reason", pa.string()),
    ("king_digest_at_time", pa.string()), ("reign", pa.int32()), ("reign_revoked", pa.bool_()),
    ("is_king_row", pa.bool_()), ("gated_near_king", pa.bool_()), ("joined", pa.bool_()),
    ("stratum", pa.string()), ("base_stratum", pa.string()), ("cell", pa.string()),
    ("group", pa.string()), ("source", pa.string()), ("harness", pa.string()),
    ("policy_id", pa.string()), ("action_kind", pa.string()), ("depth_bin", pa.string()),
    ("prefix_chars", pa.int64()), ("n_refs_valid", pa.int32()), ("refs_identical", pa.bool_()),
    ("live", pa.bool_()), ("scored", pa.bool_()), ("forfeit", pa.bool_()),
    ("turn_score", pa.float64()), ("r_leg", pa.float64()), ("g_leg", pa.float64()),
    ("b_pass", pa.bool_()), ("thought_chars", pa.int64()), ("miss", pa.bool_()),
    ("d", pa.float64()), ("div_action", pa.float64()), ("div_action_exact", pa.float64()),
    ("div_score", pa.float64()), ("b_lift", pa.float64()), ("teacher_own_lift", pa.float64()),
])

ROLLUP_SCHEMA = pa.schema([
    ("level", pa.string()), ("key", pa.string()), ("group", pa.string()), ("cell", pa.string()),
    ("n_obs", pa.int64()), ("n_w", pa.float64()), ("M", pa.float64()), ("S", pa.float64()),
    ("forfeit_rate", pa.float64()), ("mean_score", pa.float64()), ("n_d", pa.int64()),
    ("n_d_w", pa.float64()), ("Dbar", pa.float64()), ("Dbar_plus", pa.float64()),
    ("n_act_w", pa.float64()), ("div_action", pa.float64()), ("div_action_exact", pa.float64()),
    ("n_sc_w", pa.float64()), ("div_score", pa.float64()), ("M_live", pa.float64()), ("mean_live_score", pa.float64()),
    ("q25_live_score", pa.float64()), ("n_live_answered", pa.int64()),
    ("draws_50", pa.int64()), ("distinct_turns_50", pa.int64()),
    ("max_turn_draws_50", pa.int64()), ("n_draws_total", pa.int64()),
    ("distinct_turns_total", pa.int64()),
])


def to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    return pa.table(cols, schema=schema)


def rows_from_parquet(path: Path) -> list[dict]:
    t = pq.read_table(path)
    rows = t.to_pylist()
    for r in rows:
        for k in ("z", "margin", "turn_score", "r_leg", "g_leg", "d"):
            r[k] = clean_float(r[k])
    return rows


def write_ledger(rows: list[dict], roll: list[dict], meta: dict, out_dir: Path) -> dict:
    sort_rows(rows)
    lsha = rows_sha256(rows)
    roll_sha = sha256_bytes(b"".join(canonical_json(r) + b"\n" for r in roll))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"{lsha}.rows.parquet"
    roll_path = out_dir / f"{lsha}.rollup.parquet"
    write_parquet(to_table(rows, ROW_SCHEMA), rows_path)
    write_parquet(to_table(roll, ROLLUP_SCHEMA), roll_path)
    doc = {
        "ledger_version": LEDGER_VERSION,
        "ledger_sha256": lsha,
        "rows_sha256": lsha,
        "rollup_sha256": roll_sha,
        "rows_parquet_sha256": sha256_bytes(rows_path.read_bytes()),
        "rollup_parquet_sha256": sha256_bytes(roll_path.read_bytes()),
        "n_rows": len(rows), "n_rollup": len(roll),
        **meta,
    }
    write_json(doc, out_dir / f"{lsha}.json")
    write_json({"ledger_sha256": lsha, "json": f"{lsha}.json",
                "rows": rows_path.name, "rollup": roll_path.name,
                "last_challenge_id": meta["window"]["last_challenge_id"],
                "n_verdicts": meta["window"]["n_verdicts"]},
               out_dir / "latest.json")
    return doc


def build(args) -> dict:
    cfg = load_curriculum_cfg()
    hist = load_history_rows(args.history)
    rows, extra = build_rows(hist, args.evals, since=args.since, until=args.until, cfg=cfg,
                             base=args.data_base, index_cache=Path(args.index_cache),
                             max_verdicts=args.max_verdicts)
    theta = apply_theta(rows, float(cfg["theta_pct"]))
    sort_rows(rows)
    scored = extra["stats"].pop("scored_cids")
    roll = rollup(rows, scored, cfg)
    hist_sel = [r for r in hist if r.get("event") in ("crowned", "crown_revoked")
                or (r.get("event") == "verdict" and r.get("challenge_id") in set(scored))]
    hist_sel.sort(key=lambda r: (str(r.get("challenge_id")), str(r.get("at")), str(r.get("event"))))
    kings: dict[str, set[str]] = defaultdict(set)
    king_digest: dict[str, str] = {}
    for r in rows:
        if r["side"] == "king":
            key = f"{r['reign']}{'-revoked' if r['reign_revoked'] else ''}"
            kings[key].add(r["challenge_id"])
            king_digest[key] = r["king_digest_at_time"]
    last = rows[-1] if rows else None
    newest_key = (f"{last['reign']}{'-revoked' if last['reign_revoked'] else ''}") if last else None
    meta = {
        "window": {**extra["stats"], "since": args.since, "until": args.until},
        "kings_in_window": {k: len(v) for k, v in sorted(kings.items())},
        "newest_king": ({"reign": last["reign"], "digest": last["king_digest_at_time"],
                         "revoked": last["reign_revoked"], "n_verdicts": len(kings[newest_key])}
                        if last else None),
        "theta": theta, "theta_pct": cfg["theta_pct"],
        "knobs": {k: cfg[k] for k in ("half_life_verdicts", "recurrence_window_verdicts", "near_king_z")},
        "inputs": {**extra["inputs"],
                   "history_rows_sha256": sha256_bytes(b"".join(canonical_json(r) + b"\n" for r in hist_sel)),
                   "history_source": args.history, "evals_source": args.evals,
                   "data_base": args.data_base},
        "row_sort": list(ROW_SORT),
        "definitions": {
            "live": "n_refs_valid >= 2 and the reference actions are not all identical (whitespace-normalized)",
            "miss": "forfeit OR (live AND turn_score < theta); theta = theta_pct quantile of turn_score over live king rows",
            "is_king_row": "the side that held the throne at duel time (revoked reigns weigh 0); a crowning duel's challenger rows too",
            "d": "challenger turn_score - king turn_score on the same turn (both sides scored)",
            "gated_near_king": "verdict had no rejection_reason and |z| < near_king_z",
            "rollup.n_w": "sum over king rows of 0.5^(age_in_verdicts / half_life_verdicts)",
            "div_action": "1 - mean token-Jaccard between the side's normalised action and each valid reference action (evalsrv.amatch normal form; None for text)",
            "div_score": "max(0, mean_i(lp_own_i - lp_empty_i) - B_side): the teacher's thought explains its own action more than the side's thought explains the side's action (per byte)",
            "rollup.Dbar_plus": "decayed mean over gated near-king challenger rows of max(0, challenger - king turn score): 'a challenger can do better here'",
            "rollup.M_live": "miss rate among live, answered king rows (score < theta): 'answers badly'; M - M_live*S_answered ~ forfeits",
            "rollup.q25_live_score": "unweighted 25th percentile of live answered king scores in the key (compare with the global theta)",
        },
    }
    doc = write_ledger(rows, roll, meta, Path(args.out))
    log(f"ledger {doc['ledger_sha256']}: {doc['n_rows']} rows, {doc['n_rollup']} rollup keys, "
        f"theta {theta}, join {extra['stats']['join_rate']}")
    for k in ("ledger_sha256", "rows_parquet_sha256", "rollup_parquet_sha256", "rollup_sha256"):
        print(f"{k} {doc[k]}")
    return doc


def check(args) -> int:
    """Rebuild in a temp dir and compare the content sha with `--check`."""
    with tempfile.TemporaryDirectory() as tmp:
        args.out = tmp
        doc = build(args)
    ok = doc["ledger_sha256"] == args.check
    print(f"check {'OK' if ok else 'MISMATCH'}: rebuilt {doc['ledger_sha256']} expected {args.check}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--history", default=str(HISTORY_PATH),
                    help=f"history.jsonl path or URL (public: {PUBLIC_EVALS_BASE}/data/history_full.jsonl.gz)")
    ap.add_argument("--evals", default=str(EVALS_DIR),
                    help=f"evals dir or URL prefix (public: {PUBLIC_EVALS_BASE}/evals)")
    ap.add_argument("--index-cache", default=str(INDEX_CACHE))
    ap.add_argument("--data-base", default=DATA_BASE)
    ap.add_argument("--since", default=None, help="first challenge id (default: [curriculum].first_challenge)")
    ap.add_argument("--until", default="chal-99999")
    ap.add_argument("--max-verdicts", type=int, default=None, help="debug: only the last N verdicts")
    ap.add_argument("--out", default=str(STATE_DIR / "ledger"))
    ap.add_argument("--check", default=None, metavar="SHA256",
                    help="rebuild and compare the ledger sha; exit 1 on mismatch")
    args = ap.parse_args()
    if args.since is None:
        args.since = load_curriculum_cfg()["first_challenge"]
    if args.check:
        sys.exit(check(args))
    build(args)


if __name__ == "__main__":
    main()
