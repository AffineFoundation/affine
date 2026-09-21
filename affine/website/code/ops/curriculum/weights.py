#!/usr/bin/env python
"""Rule v1 weights: ledger rollup + the live corpus index -> shadow vector.

Outputs (under --out):
  rule.json               version, mode, knobs, theta, window, input shas, weights sha
  weights.parquet         per base stratum: n, M, S, M~, S~, w, rank, m_shadow, m_applied
  groups.json             share raw -> after_floor -> after_clamp -> applied + reason
  recurrence.json         ledger recurrence (last 50 verdicts) + projected draws
  deficit_by_source.json  cell deficit M~*S~ per (source, harness)

    python ops/curriculum/weights.py --ledger-json affine/state/curriculum/ledger/<sha>.json \
        --out affine/state/curriculum/work/weights
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rule  # noqa: E402
from common import (  # noqa: E402
    DATA_BASE, FITTED_SCORE_MODES, INDEX_CACHE, INPUT_UNITS, PROBES_PATH, RESCORING_METHOD, STATE_DIR,
    base_stratum, canonical_json,
    clean_float, depth_bin, fetch_bytes, group_of_stratum, index_table_for_manifest,
    load_curriculum_cfg, load_sources_toml, load_src2grp, load_static_mix, log,
    manifest_by_sha, norm_text, sha256_bytes, write_json, write_parquet,
)
from ledger import RolloutMeta  # noqa: E402

WEIGHTS_SCHEMA = pa.schema([
    ("stratum", pa.string()), ("group", pa.string()), ("cell", pa.string()),
    ("source", pa.string()), ("harness", pa.string()), ("action_kind", pa.string()),
    ("depth_bin", pa.string()), ("n_turns", pa.int64()), ("n_slice_keys", pa.int64()),
    ("bucketed", pa.bool_()), ("n_obs", pa.int64()), ("n_w", pa.float64()),
    ("M", pa.float64()), ("S", pa.float64()), ("forfeit_rate", pa.float64()), ("forfeit_share", pa.float64()),
    ("M_t", pa.float64()), ("S_t", pa.float64()),
    ("prior_M", pa.float64()), ("prior_S", pa.float64()), ("probe_yield", pa.float64()),
    ("w", pa.float64()), ("w_counted", pa.float64()), ("w_m_only", pa.float64()), ("w_v11", pa.float64()),
    ("F_t", pa.float64()), ("Dp_t", pa.float64()), ("M12_t", pa.float64()), ("w_v12", pa.float64()),
    ("Dbar_plus", pa.float64()), ("div_action", pa.float64()), ("div_action_t", pa.float64()),
    ("div_score", pa.float64()), ("div_score_t", pa.float64()), ("D_v2", pa.float64()), ("w_v2", pa.float64()),
    ("rank_pct", pa.float64()), ("m_shadow", pa.int32()),
    ("m_applied", pa.int32()), ("draws_50", pa.int64()), ("distinct_turns_50", pa.int64()),
    ("max_turn_draws_50", pa.int64()), ("Dbar", pa.float64()),
])


def live_manifest(base: str, sha: str | None) -> tuple[dict, str]:
    if sha:
        return manifest_by_sha(sha, base), sha
    raw = fetch_bytes(f"{base}/corpus/manifest.json")
    sha = sha256_bytes(raw)
    return manifest_by_sha(sha, base), sha


def load_rollup(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {"stratum": {}, "cell": {}, "group": {}, "group_era": {}}
    for r in pq.read_table(path).to_pylist():
        out.setdefault(r["level"], {})[r["key"]] = r
    return out


def read_probes(path: Path) -> bytes:
    """The teacher-probe side table as one pinned byte string (the live file
    grows while the probe runs; the snapshot hashes exactly what it used)."""
    if not path.is_file():
        return b""
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def probe_yields(probes: bytes, base_of: dict[str, str]) -> dict[str, float]:
    """Teacher-probe live yield per base stratum (S prior for unscored strata)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for line in probes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        s = base_of.get(str(p.get("turn_id") or ""))
        if s is None:
            continue
        live = int(p.get("n_valid") or 0) >= 2 and not bool(p.get("identical"))
        acc[s].append(1.0 if live else 0.0)
    return {s: sum(v) / len(v) for s, v in acc.items() if v}


def build_strata(index_rows: list[dict], meta: RolloutMeta, src2grp: dict[str, str],
                 groups: set[str]) -> tuple[dict[str, dict], dict[str, str], dict[str, int]]:
    """Per base stratum: group, modal cell, n_turns, slice keys. Also
    turn_id -> base stratum and the live slice-key count per group."""
    strata: dict[str, dict] = {}
    base_of: dict[str, str] = {}
    keys_by_group: dict[str, set[str]] = defaultdict(set)
    cells: dict[str, Counter] = defaultdict(Counter)
    slice_keys: dict[str, set[str]] = defaultdict(set)
    for r in index_rows:
        stratum = str(r.get("stratum") or "")
        s = base_stratum(stratum, r.get("stratum_src"))
        src = str(r.get("source") or "")
        g = group_of_stratum(s, src, src2grp, groups)
        _, harness = meta.meta.get(str(r.get("rollout_id") or ""), ("", ""))
        cell = "|".join([g, src, harness, str(r.get("action_kind") or ""),
                         depth_bin(r.get("n_prefix_chars"))])
        rec = strata.setdefault(s, {"stratum": s, "group": g, "n_turns": 0, "source": src})
        rec["n_turns"] += 1
        cells[s][cell] += 1
        slice_keys[s].add(stratum)
        keys_by_group[g].add(stratum)
        base_of[str(r["turn_id"])] = s
    for s, rec in strata.items():
        cell = sorted(cells[s].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rec["cell"] = cell
        parts = cell.split("|")
        rec["harness"], rec["action_kind"], rec["depth_bin"] = parts[2], parts[3], parts[4]
        rec["n_slice_keys"] = len(slice_keys[s])
        # a slice key shared by several base strata = a fold bucket
        rec["bucketed"] = any(k.split(":", 1)[-1].startswith("b") and ":" in k and k != s
                              for k in slice_keys[s])
    current_keys = {g: len(v) for g, v in keys_by_group.items()}
    return strata, base_of, current_keys


def compute(args) -> dict:
    raw_toml = load_sources_toml()
    cfg = load_curriculum_cfg(raw_toml)
    mode = args.mode or cfg["mode"]
    static = load_static_mix(raw_toml)
    src2grp = load_src2grp(raw_toml)
    groups = set(static)
    ledger_doc = json.loads(Path(args.ledger_json).read_text())
    lsha = ledger_doc["ledger_sha256"]
    roll = load_rollup(Path(args.ledger_json).with_name(f"{lsha}.rollup.parquet"))

    manifest, msha = live_manifest(args.data_base, args.manifest_sha)
    epoch = int(manifest.get("corpus_epoch") or 0)
    table = index_table_for_manifest(manifest, args.data_base)
    meta = RolloutMeta(Path(args.index_cache), args.data_base)
    meta.add_manifest(manifest)
    index_rows = table.to_pylist()
    log(f"weights: manifest {msha[:12]} epoch {epoch}, {len(index_rows)} turns")
    strata, base_of, current_keys = build_strata(index_rows, meta, src2grp, groups)
    log(f"weights: {len(strata)} base strata, slice keys per group {dict(sorted(current_keys.items()))}")
    probe_bytes = read_probes(Path(args.probes) if args.probes else PROBES_PATH)
    probes = probe_yields(probe_bytes, base_of)
    probes_sha = sha256_bytes(probe_bytes)

    # ledger stats onto the live strata (strata that left D are dropped; new
    # strata get the prior)
    for s, rec in strata.items():
        r = roll["stratum"].get(s) or {}
        rec.update({"n_obs": int(r.get("n_obs") or 0), "n_w": float(r.get("n_w") or 0.0),
                    "M": r.get("M"), "S": r.get("S"), "Dbar": r.get("Dbar"),
                    "forfeit_rate": r.get("forfeit_rate"), "Dbar_plus": r.get("Dbar_plus"),
                    "n_d_w": float(r.get("n_d_w") or 0.0),
                    "div_action": r.get("div_action"), "n_act_w": float(r.get("n_act_w") or 0.0),
                    "div_score": r.get("div_score"), "n_sc_w": float(r.get("n_sc_w") or 0.0),
                    # share of the king's misses that were forfeits ("cannot
                    # answer") rather than live turns under theta ("answers badly")
                    "forfeit_share": (float(r["forfeit_rate"]) / float(r["M"])
                                      if r.get("M") and r.get("forfeit_rate") is not None else None),
                    "draws_50": int(r.get("draws_50") or 0),
                    "distinct_turns_50": int(r.get("distinct_turns_50") or 0),
                    "max_turn_draws_50": int(r.get("max_turn_draws_50") or 0),
                    "probe_yield": probes.get(s)})
    keep = ("n_w", "M", "S", "forfeit_rate", "Dbar_plus", "n_d_w", "div_action", "n_act_w", "div_score", "n_sc_w")
    cells = {c: {"group": r["group"], **{k: r.get(k) for k in keep}} for c, r in roll["cell"].items()}
    grp_stats = {g: {k: r.get(k) for k in keep} for g, r in roll["group"].items()}
    tot_w = sum(float(r["n_w"] or 0) for r in roll["group"].values())
    corpus_s = (sum(float(r["n_w"] or 0) * float(r["S"] or 0) for r in roll["group"].values()) / tot_w
                if tot_w > 0 else 0.85)
    n0 = float(cfg["n_0"])
    rule.shrunk_rates(strata, cells, grp_stats, n0=n0, corpus_s=corpus_s, probe_s=probes)
    s_gate = float(cfg.get("v11_s_gate", 0.5))
    # v1.2 inputs: forfeit rate and the positive challenger gap, shrunk like M,
    # the gap scaled so its corpus mean equals the corpus mean forfeit rate
    corpus_f = (sum(float(r["n_w"] or 0) * float(r.get("forfeit_rate") or 0) for r in roll["group"].values()) / tot_w
                if tot_w > 0 else 0.03)
    tot_dw = sum(float(r.get("n_d_w") or 0) for r in roll["group"].values())
    corpus_dp = (sum(float(r.get("n_d_w") or 0) * float(r.get("Dbar_plus") or 0) for r in roll["group"].values()) / tot_dw
                 if tot_dw > 0 else 0.0)
    dplus_scale = (corpus_f / corpus_dp) if corpus_dp > 0 else 0.0
    rule.shrink_field(strata, cells, grp_stats, field="forfeit_rate", n_field="n_w", out="F_t",
                      corpus_prior=corpus_f, n0=n0)
    rule.shrink_field(strata, cells, grp_stats, field="Dbar_plus", n_field="n_d_w", out="Dp_t",
                      corpus_prior=corpus_dp, n0=n0)
    for rec in strata.values():
        rec["M12_t"], rec["w_v12"] = rule.stratum_weight_v12(
            rec["F_t"], rec["Dp_t"], rec["S_t"], dplus_scale=dplus_scale, eps=float(cfg["eps"]),
            gamma=float(cfg["gamma"]), s_gate=s_gate)
    for rec in strata.values():
        rec["w"] = rule.stratum_weight(rec["M_t"], rec["S_t"], eps=float(cfg["eps"]),
                                       gamma=float(cfg["gamma"]))
        rec["w_m_only"] = rule.stratum_weight(rec["M_t"], 1.0, eps=float(cfg["eps"]), gamma=float(cfg["gamma"]))
        rec["w_v11"] = rule.stratum_weight_v11(rec["M_t"], rec["S_t"], eps=float(cfg["eps"]),
                                               gamma=float(cfg["gamma"]), s_gate=s_gate)
    # ---- rule v2 (Jacob 2026-09-16 16:38 UTC, "sample more where the divergence
    # between king and teacher is greatest, ranked intelligently, with a non-zero
    # chance of visiting every turn"): D_s = weighted mix of four king-vs-teacher
    # distances, each shrunk stratum -> cell -> group -> corpus and divided by
    # its corpus mean; w_s = eps/N + (1-eps) * D^gamma / sum D^gamma.
    v2_w = dict(cfg.get("v2_component_weights") or {"action": 0.25, "forfeit": 0.25, "score": 0.25, "gap": 0.25})
    v2_eps = float(cfg.get("v2_eps", 0.20))
    v2_gamma = float(cfg.get("v2_gamma", 1.0))

    def corpus_mean(field: str, n_field: str) -> float:
        tw = sum(float(r.get(n_field) or 0) for r in roll["group"].values())
        return (sum(float(r.get(n_field) or 0) * float(r.get(field) or 0) for r in roll["group"].values()) / tw
                if tw > 0 else 0.0)
    v2_means = {"action": corpus_mean("div_action", "n_act_w"), "forfeit": corpus_f,
                "score": corpus_mean("div_score", "n_sc_w"), "gap": corpus_dp}
    rule.shrink_field(strata, cells, grp_stats, field="div_action", n_field="n_act_w", out="div_action_t",
                      corpus_prior=v2_means["action"], n0=n0)
    rule.shrink_field(strata, cells, grp_stats, field="div_score", n_field="n_sc_w", out="div_score_t",
                      corpus_prior=v2_means["score"], n0=n0)
    for rec in strata.values():
        rec["D_v2"] = rule.divergence_v2({"action": rec["div_action_t"], "forfeit": rec["F_t"],
                                          "score": rec["div_score_t"], "gap": rec["Dp_t"]}, v2_means, v2_w)
    rule.weights_v2(strata, eps=v2_eps, gamma=v2_gamma)

    # The COUNTED rule ([curriculum].counted_rule, default v1): its weight
    # drives the published shares and the multiplicity; the others are
    # published next to it. Switching rules at fold 3 = one toml line.
    counted = str(cfg.get("counted_rule") or "v1")
    counted_field = {"v1": "w", "v1.1": "w_v11", "v1.2": "w_v12", "v2": "w_v2"}[counted]
    for rec in strata.values():
        rec["w_counted"] = rec[counted_field]
    rule.multiplicity(strata, m_max=int(cfg["m_max"]), field="w_counted")

    tot_keys = sum(current_keys.values()) or 1
    current = {g: current_keys.get(g, 0) / tot_keys for g in sorted(groups | set(current_keys))}
    # Σ w per group. Operator decision 2026-09-15 00:39 UTC (lever 2): sum
    # over the SLICE KEYS -- the phase-9 buckets, the unit the duel draws one
    # turn from -- with a bucket's weight = the mean w of the base strata it
    # merges. The plan's Σ over base strata is kept as a diagnostic column:
    # it re-imports coding's raw 14k stratum count and pulls the vector back
    # toward coding / terminal, against the directive.
    raw_base = rule.raw_group_shares({k: {**r, "w": r["w_counted"]} for k, r in strata.items()})
    raw_slicekeys = rule.shares_by_slice_key(strata, index_rows, "w_counted")
    share_unit = str(cfg.get("share_unit") or "slice_keys")
    raw_shares = raw_slicekeys if share_unit == "slice_keys" else raw_base
    # Phase 10 (fold worker, 2026-09-16): the stop-state block keeps >= stop_state_floor of
    # the slice; the rule honours it as a constraint so the published vector already does.
    ss_groups = tuple(cfg.get("stop_state_groups") or ())
    block_floors = ({"stop_state": (ss_groups, float(cfg.get("stop_state_floor") or 0.0))}
                    if ss_groups and float(cfg.get("stop_state_floor") or 0) > 0 else {})
    vec = rule.group_vector(raw_shares, static, current, floor_frac=float(cfg["floor_frac_of_static"]),
                            floor_ct=float(cfg["floor_coding_terminal"]), cap=float(cfg["group_cap"]),
                            max_shift=float(cfg["max_share_shift"]), block_floors=block_floors)
    fill_kw = dict(floor_frac=float(cfg["floor_frac_of_static"]), floor_ct=float(cfg["floor_coding_terminal"]),
                   cap=float(cfg["group_cap"]), max_shift=float(cfg["max_share_shift"]), block_floors=block_floors)
    # Decomposition (coordinator 2026-09-15 00:46 UTC): what drives the vector.
    #   m_only     -- the same rule with S~ = 1 (king miss rate alone)
    #   v1.1       -- S~ as a GATE (live share >= s_gate -> eligible), weight = miss rate; informational
    #   floors_only -- the live slice share with only the plan's floors / cap applied (no rule)
    vec_m_only = rule.group_vector(rule.shares_by_slice_key(strata, index_rows, "w_m_only"), static, current, **fill_kw)
    vec_v11 = rule.group_vector(rule.shares_by_slice_key(strata, index_rows, "w_v11"), static, current, **fill_kw)
    vec_v12 = rule.group_vector(rule.shares_by_slice_key(strata, index_rows, "w_v12"), static, current, **fill_kw)
    vec_v1 = rule.group_vector(rule.shares_by_slice_key(strata, index_rows, "w"), static, current, **fill_kw)
    vec_v2 = rule.group_vector(rule.shares_by_slice_key(strata, index_rows, "w_v2"), static, current, **fill_kw)
    all_groups = sorted(set(current) | groups)
    floors_only = rule.constrained_fill(
        {g: current.get(g, 0.0) for g in all_groups},
        {g: (fill_kw["floor_frac"] * static.get(g, 0.0) if current.get(g, 0.0) > 0 else 0.0) for g in all_groups},
        {g: (fill_kw["cap"] if current.get(g, 0.0) > 0 else 0.0) for g in all_groups})
    means = rule.group_means_by_slice_key(strata, index_rows, ("M_t", "S_t", "w", "w_v11", "F_t", "Dp_t", "M12_t", "w_v12",
                                                              "div_action_t", "div_score_t", "D_v2", "w_v2"))
    # recurrence guard on the counted vector: lower k before share (hard cap)
    guard = rule.recurrence_guard(strata, vec["after_clamp"], group_cap=float(cfg["recurrence_group_cap"]),
                                  turn_cap=float(cfg["recurrence_turn_cap"]))
    if guard["actions"]:
        log(f"weights: recurrence guard -- {len(guard['actions'])} actions, e.g. {guard['actions'][:3]}")
    vec["after_clamp"] = rule.restore_block_floors(
        guard["shares"], block_floors, fixed=set(guard["groups_cut"]), floor=vec["floor"], cap=float(cfg["group_cap"]),
        current=current, max_shift=float(cfg["max_share_shift"]))
    if guard["actions"]:
        vec["blocks"] = {name: {**b, "after_guard": sum(vec["after_clamp"].get(g, 0.0) for g in b["members"])}
                         for name, b in vec["blocks"].items()}
    applied = vec["after_clamp"] if mode == "apply" else current
    proj_shadow = rule.recurrence_projection(strata, vec["after_clamp"])
    proj_applied = rule.recurrence_projection(strata, applied)

    # -- weights table (hashed) --
    for rec in strata.values():
        rec["m_shadow"] = int(rec["m"])
        rec["m_applied"] = int(rec["m"]) if mode == "apply" else max(1, int(rec["n_slice_keys"])) \
            if not rec["bucketed"] else 1
    wrows = []
    for s in sorted(strata):
        rec = strata[s]
        wrows.append({f.name: rec.get(f.name) for f in WEIGHTS_SCHEMA})
        for k in ("n_w", "M", "S", "forfeit_rate", "forfeit_share", "M_t", "S_t", "prior_M", "prior_S",
                  "probe_yield", "w", "w_counted", "w_m_only", "w_v11", "F_t", "Dp_t", "M12_t", "w_v12", "Dbar_plus",
                  "div_action", "div_action_t", "div_score", "div_score_t", "D_v2", "w_v2", "rank_pct", "Dbar"):
            wrows[-1][k] = clean_float(wrows[-1][k])
    knobs = {k: cfg[k] for k in ("rule_version", "half_life_verdicts", "n_0", "gamma", "eps", "theta_pct",
                                 "m_max", "floor_coding_terminal", "floor_frac_of_static", "group_cap",
                                 "max_share_shift", "min_new_verdicts")}
    knobs["share_unit"] = share_unit
    knobs["counted_rule"] = counted
    knobs["stop_state_groups"] = list(ss_groups)
    knobs["stop_state_floor"] = cfg.get("stop_state_floor")
    hashed = {"rule_version": int(cfg["rule_version"]), "knobs": knobs, "ledger_sha256": lsha,
              "manifest_sha256": msha, "theta": ledger_doc.get("theta"), "probes_sha256": probes_sha,
              # the group vector is an output of the rule too: a share change (guard, block floor)
              # with an unchanged strata table must still be a new snapshot
              "shares_after_clamp": {g: clean_float(v) for g, v in sorted(vec["after_clamp"].items())},
              "strata": wrows}
    wsha = sha256_bytes(canonical_json(hashed))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher_probe.jsonl.gz").write_bytes(gzip.compress(probe_bytes, mtime=0))
    write_parquet(pa.table({f.name: [r[f.name] for r in wrows] for f in WEIGHTS_SCHEMA},
                           schema=WEIGHTS_SCHEMA), out / "weights.parquet")
    g_rows = {}
    for g in sorted(set(vec["raw"]) | groups):
        ss = [r for r in strata.values() if r["group"] == g]
        gr = roll["group"].get(g) or {}
        g_rows[g] = {
            "share_raw": clean_float(vec["raw"].get(g, 0.0)),
            "share_raw_slice_keys": clean_float(raw_slicekeys.get(g, 0.0)),
            "share_raw_base_strata": clean_float(raw_base.get(g, 0.0)),
            "share_after_floor": clean_float(vec["after_floor"].get(g, 0.0)),
            "share_after_clamp": clean_float(vec["after_clamp"].get(g, 0.0)),
            "share_applied": clean_float(applied.get(g, 0.0)),
            "share_current": clean_float(current.get(g, 0.0)),
            "share_static": clean_float(static.get(g, 0.0)),
            "floor": clean_float(vec["floor"].get(g, 0.0)),
            "reason": vec["reasons"].get(g, "no_supply"),
            "n_strata": len(ss), "n_slice_keys": current_keys.get(g, 0),
            "n_turns": sum(int(r["n_turns"]) for r in ss),
            "sum_w": clean_float(sum(float(r["w"]) for r in ss)),
            "mean_w": clean_float(sum(float(r["w"]) for r in ss) / len(ss)) if ss else None,
            "M_t": clean_float(rule.shrink(float(gr.get("n_w") or 0), gr.get("M"), rule.CORPUS_PRIOR_M, n0)),
            "S_t": clean_float(rule.shrink(float(gr.get("n_w") or 0), gr.get("S"), corpus_s, n0)),
            "n_w": clean_float(gr.get("n_w")), "n_obs": int(gr.get("n_obs") or 0),
            # decomposition over the group's slice keys (what the sampler draws)
            "mean_M_t_slice_keys": clean_float((means.get(g) or {}).get("M_t")),
            "mean_S_t_slice_keys": clean_float((means.get(g) or {}).get("S_t")),
            "mean_w_slice_keys": clean_float((means.get(g) or {}).get("w")),
            "share_m_only_raw": clean_float(vec_m_only["raw"].get(g, 0.0)),
            "share_m_only_after_clamp": clean_float(vec_m_only["after_clamp"].get(g, 0.0)),
            "share_v1_raw": clean_float(vec_v1["raw"].get(g, 0.0)),
            "share_v1_after_floor": clean_float(vec_v1["after_floor"].get(g, 0.0)),
            "share_v1_after_clamp": clean_float(vec_v1["after_clamp"].get(g, 0.0)),
            "v1_reason": vec_v1["reasons"].get(g, "no_supply"),
            "share_v11_raw": clean_float(vec_v11["raw"].get(g, 0.0)),
            "share_v11_after_floor": clean_float(vec_v11["after_floor"].get(g, 0.0)),
            "share_v11_after_clamp": clean_float(vec_v11["after_clamp"].get(g, 0.0)),
            "v11_reason": vec_v11["reasons"].get(g, "no_supply"),
            "v11_eligible_strata": sum(1 for r in ss if float(r["w_v11"]) > 0),
            "share_floors_only": clean_float(floors_only.get(g, 0.0)),
            # v1.2: forfeit + scaled positive challenger gap, S~ gate (informational)
            "mean_F_t_slice_keys": clean_float((means.get(g) or {}).get("F_t")),
            "mean_Dp_t_slice_keys": clean_float((means.get(g) or {}).get("Dp_t")),
            "mean_M12_t_slice_keys": clean_float((means.get(g) or {}).get("M12_t")),
            "share_v12_raw": clean_float(vec_v12["raw"].get(g, 0.0)),
            "share_v12_after_floor": clean_float(vec_v12["after_floor"].get(g, 0.0)),
            "share_v12_after_clamp": clean_float(vec_v12["after_clamp"].get(g, 0.0)),
            "v12_reason": vec_v12["reasons"].get(g, "no_supply"),
            # v2: divergence rule (informational unless counted)
            "mean_div_action_t_slice_keys": clean_float((means.get(g) or {}).get("div_action_t")),
            "mean_div_score_t_slice_keys": clean_float((means.get(g) or {}).get("div_score_t")),
            "mean_D_v2_slice_keys": clean_float((means.get(g) or {}).get("D_v2")),
            "share_v2_raw": clean_float(vec_v2["raw"].get(g, 0.0)),
            "share_v2_after_floor": clean_float(vec_v2["after_floor"].get(g, 0.0)),
            "share_v2_after_clamp": clean_float(vec_v2["after_clamp"].get(g, 0.0)),
            "v2_reason": vec_v2["reasons"].get(g, "no_supply"),
            "king_div_action": clean_float(gr.get("div_action")), "king_div_action_exact": clean_float(gr.get("div_action_exact")),
            "king_div_score": clean_float(gr.get("div_score")),
            # inputs split by scoring era (wvk / score_mode) -- an era shift shows here first
            "king_by_era": {k.split("|", 1)[1]: {kk: (clean_float(er.get(kk)) if kk != "n_obs" else int(er.get(kk) or 0))
                                                 for kk in ("n_obs", "M", "forfeit_rate", "mean_score", "Dbar_plus",
                                                            "div_action", "div_score", "S")}
                            for k, er in sorted(roll.get("group_era", {}).items()) if k.split("|", 1)[0] == g},
            "king_Dbar": clean_float(gr.get("Dbar")), "king_Dbar_plus": clean_float(gr.get("Dbar_plus")),
            "king_n_d": int(gr.get("n_d") or 0),
            # the king's miss rate, raw from the ledger (decayed means over king rows)
            "king_M": clean_float(gr.get("M")), "king_S": clean_float(gr.get("S")),
            "king_forfeit_rate": clean_float(gr.get("forfeit_rate")),
            "king_M_live_answered": clean_float(gr.get("M_live")),
            "king_mean_live_score": clean_float(gr.get("mean_live_score")),
            "king_q25_live_score": clean_float(gr.get("q25_live_score")),
            "king_n_live_answered": int(gr.get("n_live_answered") or 0),
            "m_hist_shadow": dict(sorted(Counter(int(r["m_shadow"]) for r in ss).items())),
        }
    groups_doc = {"mode": mode, "rule_version": int(cfg["rule_version"]), "weights_sha256": wsha,
                  "input_units": INPUT_UNITS, "fitted_score_modes": list(FITTED_SCORE_MODES),
                  "rescoring_method": RESCORING_METHOD,
                  "share_unit": share_unit, "theta": ledger_doc.get("theta"),
                  "counted_rule": counted,
                  "recurrence_guard": {"actions": guard["actions"][:200], "n_actions": len(guard["actions"]),
                                       "ok": guard["ok"], "max_turn_draws_per_duel": clean_float(guard["max_turn_draws_per_duel"]),
                                       "max_group_draws": clean_float(guard["max_group_draws"]),
                                       "rule": "lower m (3->2->1) on strata over recurrence_turn_cap, then lower a group's "
                                               "share to recurrence_group_cap * n_turns / 1300 -- k before share"},
                  "v11": {"informational": True, "rule": "w = (M~ + eps)^gamma if S~ >= s_gate else 0",
                          "s_gate": s_gate, "counted": False},
                  "v2": {"informational": counted != "v2", "counted": counted == "v2", "eps": v2_eps, "gamma": v2_gamma,
                         "component_weights": v2_w, "corpus_means": {k: clean_float(v) for k, v in v2_means.items()},
                         "rule": "D_s = Σ_k ω_k · comp_k~ / corpus_mean_k over comps {action: 1 − soft A_match (token-Jaccard), "
                                 "forfeit: king forfeit rate, score: max(0, teacher own-action lift − king B), gap: Dbar+}; "
                                 "w_s = eps/N + (1 − eps) · D_s^gamma / Σ D^gamma (uniform floor = no forgetting); "
                                 "group share ∝ Σ w over slice keys; floors, cap, clamp, recurrence guard as v1",
                         "floor_property": "at the slice-key level the eps mass reproduces the LIVE share vector, so "
                                           "v2 = eps · live + (1 − eps) · divergence-driven",
                         "decision": "pending -- Jacob 2026-09-16 16:38 UTC: v2 becomes the counted rule for the fold-3 apply "
                                     "if it passes on today's and tomorrow's fold; otherwise v1.2 stays"},
                  "v12": {"informational": counted != "v1.2", "counted": counted == "v1.2", "s_gate": s_gate,
                          "rule": "M12~ = F~ + dplus_scale * Dbar+~; w = (M12~ + eps)^gamma if S~ >= s_gate else 0; "
                                  "F = king forfeit rate, Dbar+ = mean max(0, challenger - king) on gated near-king "
                                  "verdicts; both shrunk stratum -> cell -> group -> corpus",
                          "dplus_scale": clean_float(dplus_scale), "corpus_mean_forfeit": clean_float(corpus_f),
                          "corpus_mean_Dbar_plus": clean_float(corpus_dp),
                          "decision": "pending -- coordinator: v1.2 becomes the counted rule at fold 3 if it moves the "
                                      "vector toward the king groups as phase 9 did (2026-09-15 00:55 UTC)"},
                  "ledger_sha256": lsha, "manifest_sha256": msha, "corpus_epoch": epoch,
                  "joint_floor_applied": vec["joint_floor_applied"], "corpus_prior_M": rule.CORPUS_PRIOR_M,
                  "corpus_prior_S": clean_float(corpus_s), "groups": g_rows,
                  "block_floors": vec["blocks"],
                  "floors_check": rule.check_floors(vec["after_clamp"], static, block_floors=block_floors,
                                                    floor_frac=float(cfg["floor_frac_of_static"]),
                                                    floor_ct=float(cfg["floor_coding_terminal"]),
                                                    cap=float(cfg["group_cap"]),
                                                    supply={g: vec["reasons"].get(g) != "no_supply" for g in g_rows},
                                                    guard_cut=set(guard["groups_cut"]))}
    write_json(groups_doc, out / "groups.json")

    rec_doc = {
        "window_verdicts": cfg["recurrence_window_verdicts"],
        "caps": {"group_expected_draws_per_turn_per_duel": cfg["recurrence_group_cap"],
                 "single_turn_draws_per_duel": cfg["recurrence_turn_cap"]},
        "baseline_expected_draws_per_turn_per_duel": clean_float(rule.SLICE_N / tot_keys),
        "ledger": {g: {"draws_50": int(r.get("draws_50") or 0), "distinct_turns_50": int(r.get("distinct_turns_50") or 0),
                       "max_turn_draws_50": int(r.get("max_turn_draws_50") or 0)}
                   for g, r in sorted(roll["group"].items())},
        "ledger_strata_above_cap": sorted(
            [{"stratum": s, "draws_50": r["draws_50"], "max_turn_draws_50": r["max_turn_draws_50"]}
             for s, r in roll["stratum"].items() if r["max_turn_draws_50"] > 3],
            key=lambda x: (-x["max_turn_draws_50"], x["stratum"]))[:50],
        "projected_shadow": {"groups": {g: {k: clean_float(v) if isinstance(v, float) else v for k, v in d.items()}
                                        for g, d in proj_shadow["groups"].items()},
                             "max_turn_draws_per_duel": clean_float(proj_shadow["max_turn_draws_per_duel"]),
                             "max_turn_stratum": proj_shadow["max_turn_stratum"]},
        "projected_applied": {"groups": {g: {k: clean_float(v) if isinstance(v, float) else v for k, v in d.items()}
                                         for g, d in proj_applied["groups"].items()},
                              "max_turn_draws_per_duel": clean_float(proj_applied["max_turn_draws_per_duel"]),
                              "max_turn_stratum": proj_applied["max_turn_stratum"]},
    }
    write_json(rec_doc, out / "recurrence.json")

    by_src: dict[tuple[str, str], dict] = {}
    for rec in strata.values():
        k = (rec["source"], rec["harness"])
        d = by_src.setdefault(k, {"source": k[0], "harness": k[1], "group": rec["group"], "n_strata": 0,
                                  "n_turns": 0, "deficit_sum": 0.0, "draws_50": 0, "n_obs": 0})
        d["n_strata"] += 1
        d["n_turns"] += int(rec["n_turns"])
        d["deficit_sum"] += float(rec["M_t"]) * float(rec["S_t"])
        d["draws_50"] += int(rec["draws_50"])
        d["n_obs"] += int(rec["n_obs"])
    deficit_rows = []
    for k in sorted(by_src):
        d = by_src[k]
        d["deficit_mean"] = clean_float(d["deficit_sum"] / d["n_strata"])
        d["deficit_sum"] = clean_float(d["deficit_sum"])
        deficit_rows.append(d)
    deficit_rows.sort(key=lambda d: (-(d["deficit_mean"] or 0), d["source"], d["harness"]))
    write_json({"definition": "cell deficit = M~ * S~ averaged over the strata of (source, harness)",
                "rows": deficit_rows}, out / "deficit_by_source.json")

    top = sorted(strata.values(), key=lambda r: (-float(r["w_counted"]), r["stratum"]))[:10]
    rule_doc = {
        "rule_version": int(cfg["rule_version"]), "mode": mode, "knobs": knobs, "counted_rule": counted,
        "input_units": INPUT_UNITS, "fitted_score_modes": list(FITTED_SCORE_MODES), "rescoring_method": RESCORING_METHOD,
        "formula": "w_s = (M~_s + eps)^gamma * S~_s; share_g ∝ Σ w over the group's slice keys (a phase-9 bucket "
                   "weighs the mean w of the base strata it merges; share_unit = base_strata sums base strata instead); floors (coding+terminal ≥ floor_coding_terminal, "
                   "every group ≥ floor_frac_of_static × [mix]); cap group_cap; clamp ± max_share_shift vs the live "
                   "slice share; m_s = 1 + round(2 · rank_pct_within_group), ≤ n_turns, ≤ m_max",
        "theta": ledger_doc.get("theta"), "window": ledger_doc.get("window"),
        "ledger_sha256": lsha, "manifest_sha256": msha, "corpus_epoch": epoch,
        "weights_sha256": wsha, "n_strata": len(strata), "n_turns": len(index_rows),
        "probes_sha256": probes_sha, "n_probe_rows": probe_bytes.count(b"\n"),
        "v12": {"dplus_scale": clean_float(dplus_scale), "corpus_mean_forfeit": clean_float(corpus_f),
                "corpus_mean_Dbar_plus": clean_float(corpus_dp), "s_gate": s_gate},
        "v2": {"eps": v2_eps, "gamma": v2_gamma, "component_weights": v2_w,
               "corpus_means": {k: clean_float(v) for k, v in v2_means.items()},
               "uniform_floor_per_stratum": clean_float(v2_eps / max(1, len(strata)))},
        "corpus_prior_S": clean_float(corpus_s),
        "recompute": "python ops/curriculum/weights.py --ledger-json <ledger>.json --manifest-sha <manifest_sha256> "
                     "--probes teacher_probe.jsonl.gz --out <dir>  # prints weights_sha256",
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "top10_by_weight": [{"stratum": r["stratum"], "group": r["group"], "cell": r["cell"],
                             "w": clean_float(r["w"]), "M_t": clean_float(r["M_t"]), "S_t": clean_float(r["S_t"]),
                             "M": clean_float(r["M"]), "forfeit_rate": clean_float(r["forfeit_rate"]),
                             "forfeit_share": clean_float(r["forfeit_share"]),
                             "n_obs": r["n_obs"], "n_turns": r["n_turns"], "m_shadow": r["m_shadow"]} for r in top],
    }
    write_json(rule_doc, out / "rule.json")
    print(f"weights_sha256 {wsha}")
    log(f"weights: share_unit={share_unit}; shares " + ", ".join(
        f"{g} {100 * vec['after_clamp'][g]:.1f}% ({vec['reasons'][g]}; raw {100 * vec['raw'][g]:.1f}%, "
        f"now {100 * current.get(g, 0):.1f}%)" for g in sorted(vec["after_clamp"], key=lambda k: -vec["after_clamp"][k])))
    return rule_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger-json", required=True, help="the ledger's <sha>.json")
    ap.add_argument("--manifest-sha", default=None, help="corpus manifest sha (default: the live pointer)")
    ap.add_argument("--mode", default=None, choices=[None, "off", "shadow", "apply"])
    ap.add_argument("--index-cache", default=str(INDEX_CACHE))
    ap.add_argument("--data-base", default=DATA_BASE)
    ap.add_argument("--probes", default=None,
                    help="teacher-probe side table (.jsonl or .jsonl.gz); default: the live file")
    ap.add_argument("--out", default=str(STATE_DIR / "work" / "weights"))
    compute(ap.parse_args())


if __name__ == "__main__":
    main()
