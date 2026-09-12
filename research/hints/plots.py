#!/usr/bin/env python
"""PNG figures for the hinted-teacher report (matplotlib, headless).

  python plots.py --analysis RUN/analysis --out media_dir [--e4b RUN/analysis/e4b_turns.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COND_ORDER = ["H0", "ds_fact", "self_fact", "ds_plan", "ds_action", "pivot_action"]
LABEL = {"H0": "no hint", "ds_fact": "DeepSeek fact", "self_fact": "self fact", "ds_plan": "DeepSeek plan",
         "ds_action": "DeepSeek action", "pivot_action": "pivot judge"}


def load_tables(path: Path) -> list[dict]:
    return json.load(open(path / "tables.json"))


def fig_e1_refs(tables, out: Path) -> None:
    rows = {r["cond"]: r for r in tables if r["table"] == "E1_gate_off" and r["group"] == "all"}
    conds = [c for c in COND_ORDER if c in rows]
    x = range(len(conds))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.27
    ax.bar([i - w for i in x], [rows[c]["ref_yield"] for c in conds], w, label="valid refs / 3")
    ax.bar([i for i in x], [rows[c]["identical_frac"] for c in conds], w, label="identical-ref fraction")
    ax.bar([i + w for i in x], [rows[c]["r_dead_frac"] for c in conds], w, label="R-dead fraction")
    ax.axhline(0.30, color="red", ls="--", lw=1, label="identical-ref kill line 0.30")
    ax.set_xticks(list(x)); ax.set_xticklabels([LABEL[c] for c in conds], rotation=20)
    ax.set_ylim(0, 1); ax.set_title("E1 — reference yield and R-dead turns per condition (all 420 turns)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "e1_refs.png", dpi=130); plt.close(fig)


def fig_band(tc_path: Path, out: Path) -> None:
    pos = collections.defaultdict(list)
    for line in open(tc_path):
        t = json.loads(line)
        if t["t_pos0"] is not None and t["cond"] != "H0":
            pos[t["cond"]].append(max(-15, min(5, t["t_pos0"])))
    conds = [c for c in COND_ORDER if c in pos]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.violinplot([pos[c] for c in conds], showmedians=True)
    ax.axhspan(-1, 1, color="green", alpha=0.15, label="unhinted band (±1 width)")
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels([LABEL[c] for c in conds], rotation=20)
    ax.set_ylabel("hinted ref thought position (band widths from centre)")
    ax.set_title("E1 — where hinted reference thoughts sit relative to the unhinted G band")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "e1_band_position.png", dpi=130); plt.close(fig)


def fig_dz(tables, out: Path, key: str = "teacher_-king_liv") -> None:
    groups = ["all", "king_loop_onset", "king_pivot", "completion"]
    conds = [c for c in COND_ORDER if c != "H0"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(groups):
            r = next((r for r in tables if r["table"] == "E1_gate_off" and r["cond"] == c and r["group"] == g), None)
            if not r or r.get(f"dz_{key}") is None or math.isnan(r.get(f"dz_{key}", float("nan"))):
                continue
            xs.append(i + (j - len(conds) / 2) * w + w / 2); ys.append(r[f"dz_{key}"])
            lo.append(r[f"dz_{key}"] - r.get(f"dz_lo_{key}", r[f"dz_{key}"])); hi.append(r.get(f"dz_hi_{key}", r[f"dz_{key}"]) - r[f"dz_{key}"])
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3, label=LABEL[c])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.set_ylabel("Δz vs no hint (teacher held-out − king live), 95 % bootstrap")
    ax.set_title("E1 — change in paired separation when references are hinted")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "e1_dz.png", dpi=130); plt.close(fig)


def fig_e2(tables, out: Path) -> None:
    rows = [r for r in tables if r["table"] == "E2_mix" and r["group"] == "all"]
    by = collections.defaultdict(dict)
    for r in rows:
        if ":" in r["cond"]:
            c, mix = r["cond"].split(":")
            by[c][mix] = r
    h0 = next((r for r in tables if r["table"] == "E1_gate_off" and r["cond"] == "H0" and r["group"] == "all"), None)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mixes = ["0h/3u", "1h/2u", "2h/1u", "3h/0u"]
    for c, d in by.items():
        ys = [(h0["asd_teacher_heldout"] if h0 else float("nan"))] + [d.get(m, {}).get("asd_teacher_heldout", float("nan")) for m in mixes[1:]]
        axes[0].plot(mixes, ys, marker="o", label=LABEL.get(c, c))
        zs = [(h0["z_teacher_-king_liv"] if h0 else float("nan"))] + [d.get(m, {}).get("z_teacher_-king_liv", float("nan")) for m in mixes[1:]]
        axes[1].plot(mixes, zs, marker="o", label=LABEL.get(c, c))
    axes[0].set_title("E2 — R spread (sd of a_i, teacher held-out)"); axes[0].set_xlabel("hinted / unhinted refs")
    axes[1].set_title("E2 — paired z (teacher held-out − king live)"); axes[1].axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "e2_mix.png", dpi=130); plt.close(fig)


def fig_e4b(e4b_path: Path, out: Path) -> None:
    rows = [json.loads(l) for l in open(e4b_path)]
    arms = ["nohint", "ds_fact", "self_fact", "ds_plan", "pivot_action"]
    harn = sorted({r["harness"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        xs, ys, err = [], [], []
        for i, h in enumerate(["all"] + harn):
            sub = [r for r in rows if r["arm"] == a and (h == "all" or r["harness"] == h)]
            if not sub:
                continue
            k = sum(1 for r in sub if r["recovered_any"]); n = len(sub)
            p = k / n
            se = math.sqrt(p * (1 - p) / n) if n else 0
            xs.append(i + (j - len(arms) / 2) * w + w / 2); ys.append(p); err.append(1.96 * se)
        if xs:
            ax.bar(xs, ys, w, yerr=err, capsize=2, label=a)
    ax.set_xticks(range(len(harn) + 1)); ax.set_xticklabels(["all"] + harn)
    ax.set_ylabel("recovered in ≥1 of 2 continuations"); ax.set_ylim(0, 1)
    ax.set_title("E4b — recovery at states the unhinted teacher failed (baseline 0)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "e4b_recovery.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e4b")
    args = ap.parse_args()
    an = Path(args.analysis)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_tables(an)
    fig_e1_refs(tables, out)
    fig_band(an / "turn_conditions.jsonl", out)
    fig_dz(tables, out)
    fig_e2(tables, out)
    if args.e4b and Path(args.e4b).exists():
        fig_e4b(Path(args.e4b), out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
