"""Text-similarity checks behind the shadow-teacher result (all local).

Per panel miner, from its latest duel artifact:
  ref_sim  - mean over turns of MAX word-3-gram Jaccard between the miner's
             thought z_a and the era teacher's k reference thoughts on the
             same turn (parroting / template-matching the era teacher)
  self_sim - mean pairwise word-3-gram Jaccard across the miner's own
             thoughts on distinct turns (fixed template / filler signal)
  suffix_frac - fraction of thoughts sharing the modal trailing 60 chars
             (the reign-41 exploit was a fixed filler suffix)

Correlated against S_live, swe, and the shadow score (mean over the three
engy teachers from results/shadow_teacher_engy.jsonl).

Usage (from research/): python scripts/shadow_teacher_text_sim.py
"""

from __future__ import annotations

import gzip
import json
import statistics as st
from collections import Counter
from pathlib import Path

from shadow_teacher_engy import (
    artifact_for, load_panel, permutation_p, spearman,
)

SHADOW_JSONL = Path("results/shadow_teacher_engy.jsonl")
OUT_TXT = Path("results/shadow_teacher_text_sim.txt")
MAX_TURNS = 60


def trigrams(text: str) -> set[tuple[str, str, str]]:
    w = text.lower().split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def miner_features(art: Path) -> dict | None:
    d = json.load(gzip.open(art))
    refs_by_tid = d.get("teacher_refs") or {}
    thoughts: list[tuple[str, str]] = []  # (turn_id, z_a)
    for row in d.get("challenger_rows") or []:
        pairs = row.get("pairs") or []
        tid = row.get("turn_id")
        if pairs and tid and pairs[0].get("z_a"):
            thoughts.append((tid, pairs[0]["z_a"]))
    thoughts = thoughts[:MAX_TURNS]
    if len(thoughts) < 5:
        return None

    ref_sims = []
    for tid, z_a in thoughts:
        refs = refs_by_tid.get(tid) or []
        zs = [r.get("z") or "" for r in refs if r.get("z")]
        if not zs:
            continue
        g = trigrams(z_a)
        ref_sims.append(max(jaccard(g, trigrams(z)) for z in zs))

    grams = [trigrams(z) for _, z in thoughts]
    pair_sims = [jaccard(grams[i], grams[j])
                 for i in range(len(grams)) for j in range(i + 1, len(grams))]

    tails = Counter(z.strip()[-60:] for _, z in thoughts if z.strip())
    suffix_frac = tails.most_common(1)[0][1] / len(thoughts) if tails else 0.0

    return {
        "ref_sim": st.mean(ref_sims) if ref_sims else None,
        "self_sim": st.mean(pair_sims) if pair_sims else None,
        "suffix_frac": suffix_frac,
        "n_turns": len(thoughts),
    }


def shadow_scores() -> dict[str, float]:
    """repo -> mean shadow score over the three engy teachers."""
    by_repo: dict[str, list[float]] = {}
    for line in SHADOW_JSONL.read_text().splitlines():
        r = json.loads(line)
        if r.get("score") is not None:
            by_repo.setdefault(r["repo"], []).append(r["score"])
    return {k: st.mean(v) for k, v in by_repo.items()}


def corr_line(name: str, xs: list[float], ys: list[float]) -> str:
    rho = spearman(xs, ys)
    p = permutation_p(xs, ys, rho, trials=20_000)
    return f"{name:34s} n={len(xs):2d}  rho={rho:+.3f}  p={p:.4f}"


def main() -> None:
    panel = load_panel()
    shadows = shadow_scores()
    rows = []
    for row in panel:
        repo = row["repo"]
        art = artifact_for(repo)
        if art is None:
            continue
        feats = miner_features(art)
        if feats is None or feats["ref_sim"] is None:
            continue
        rows.append({"repo": repo, "s_live": row["s"], "swe": row["swe"],
                     "s_shadow": shadows.get(repo), **feats})

    lines = ["Text-similarity checks on the RT-7 panel (local artifacts)",
             "=" * 72]
    for r in sorted(rows, key=lambda r: -r["s_live"]):
        lines.append(
            f"{r['repo'][:42]:42s} S_live={r['s_live']:+.4f} "
            f"swe={r['swe']:.2f} ref_sim={r['ref_sim']:.3f} "
            f"self_sim={r['self_sim']:.3f} suffix={r['suffix_frac']:.2f}")
    lines.append("")

    def col(k):
        return [r[k] for r in rows]

    lines.append(corr_line("ref_sim vs S_live", col("ref_sim"), col("s_live")))
    lines.append(corr_line("ref_sim vs swe", col("ref_sim"), col("swe")))
    lines.append(corr_line("self_sim vs S_live",
                           col("self_sim"), col("s_live")))
    lines.append(corr_line("suffix_frac vs S_live",
                           col("suffix_frac"), col("s_live")))
    sub = [r for r in rows if r["s_shadow"] is not None]
    if len(sub) >= 5:
        lines.append(corr_line(
            "ref_sim vs S_shadow(mean)",
            [r["ref_sim"] for r in sub], [r["s_shadow"] for r in sub]))
        lines.append(corr_line(
            "self_sim vs S_shadow(mean)",
            [r["self_sim"] for r in sub], [r["s_shadow"] for r in sub]))

    text = "\n".join(lines)
    print(text)
    OUT_TXT.write_text(text + "\n")


if __name__ == "__main__":
    main()
