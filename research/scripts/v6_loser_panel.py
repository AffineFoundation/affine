"""Step 1 of the A-leg isomorphism test: pick the benched-LOSER panel (2026-09-10).

The winners-only panel (v6_action_leg_panel.py) cannot discriminate scoring
rules — every model in it crowned. To ask "does the duel margin track real
coding performance", we need rejected challengers with a bench score too.

This script lists every stored duel the challenger LOST, keeps the ones that
are usable (a real duel: ≥ --min-paired paired turns; a king whose bench
score is known; weights still fetchable), and picks a panel stratified by the
min(R,G) margin:

    near_miss   margin in (+0.0008, δ)     — the rule almost crowned them
    tie         margin in (−0.0030, +0.0008]
    mid         margin in (−0.0120, −0.0030]
    far         margin in (−0.0600, −0.0120], not degenerate (B pass ≥ 0.45)

One checkpoint per miner hotkey per stratum (diversity), most recent first
(schema-3 D, same era as the current kings). Availability: HF repos are
checked through the Hub API at the exact revision; r2:// prefixes through
head(manifest.json) with the eval read key (source ~/.affine-validator.env).

    source ~/.affine-validator.env
    python research/scripts/v6_loser_panel.py --per-stratum 3 \
        --out research/results/v6_loser_panel
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine import r2 as r2lib  # noqa: E402
from evalsrv import r2store  # noqa: E402

EVALS = REPO / "affine/state/evals"
BENCH_INDEX = REPO / "affine/state/benches/index.jsonl"
SUITE = "swe_rebench_lite_300"
DELTA = 0.002
STRATA = (
    ("near_miss", 0.0008, DELTA),
    ("tie", -0.0030, 0.0008),
    ("mid", -0.0120, -0.0030),
    ("far", -0.0600, -0.0120),
)


def bench_scores() -> dict[str, float]:
    out = {}
    for line in open(BENCH_INDEX):
        b = json.loads(line)
        if b["suite"] == SUITE and b.get("score") is not None:
            out[b["revision"]] = b["score"]
    return out


def hf_available(repo: str, revision: str) -> bool:
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}",
                      headers=headers, timeout=20, follow_redirects=True)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def r2_available(s3, repo: str) -> bool:
    # r2://<bucket>/<prefix>/
    rest = repo[len("r2://"):]
    bucket, _, prefix = rest.partition("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return bool(r2lib.object_exists(s3, bucket, prefix + "manifest.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=3)
    ap.add_argument("--min-paired", type=int, default=1100)
    ap.add_argument("--out", default="research/results/v6_loser_panel")
    ap.add_argument("--skip-availability", action="store_true")
    args = ap.parse_args()

    bench = bench_scores()
    evals = [json.loads(l) for l in open(EVALS / "index.jsonl")]
    losers = []
    for e in evals:
        if e["challenger_wins"] or e.get("margin") is None:
            continue
        d = json.load(gzip.open(EVALS / f"{e['challenge_id']}.json.gz"))
        v = d["verdict"]
        krev = d["request"]["king_revision"]
        c = v.get("challenger") or {}
        losers.append({
            "record": e["challenge_id"], "at": e["at"][:10], "hotkey": e.get("hotkey", ""),
            "repo": e["repo"], "revision": e["revision"], "margin": e["margin"], "z": e["z"],
            "rejection_reason": e.get("rejection_reason"),
            "n_paired": v["n_paired_turns"], "king_revision": krev,
            "bench_k": bench.get(krev), "b_pass": c.get("b_gate_pass_rate"),
            "forfeit_rate": c.get("forfeit_rate"), "median_len_z": c.get("median_len_z"),
            "schema3": bool(v["slice"].get("view_spec")),
        })
    print(f"losers: {len(losers)}", flush=True)

    usable = [l for l in losers
              if l["n_paired"] >= args.min_paired and l["bench_k"] is not None
              and not str(l["rejection_reason"] or "").startswith(("unpromptable", "protocol"))]
    print(f"usable (real duel, benched king): {len(usable)}", flush=True)

    # Stratify, newest first, one per hotkey per stratum, then check availability
    # only for the candidates we would actually take (plus a few spares).
    s3 = None
    if not args.skip_availability:
        s3 = r2store.client()
    picked: list[dict] = []
    report: list[str] = []
    for name, lo, hi in STRATA:
        pool = [l for l in usable if lo < l["margin"] <= hi]
        if name == "far":
            pool = [l for l in pool if (l["b_pass"] or 0) >= 0.45 and (l["forfeit_rate"] or 0) < 0.15]
        pool.sort(key=lambda l: (not l["schema3"], l["at"]), reverse=False)
        pool.sort(key=lambda l: l["at"], reverse=True)
        seen_hk: set[str] = set()
        taken = 0
        checked = 0
        report.append(f"== {name}: margin in ({lo:+.4f}, {hi:+.4f}]  pool {len(pool)} ==")
        for l in pool:
            if taken >= args.per_stratum:
                break
            if l["hotkey"] in seen_hk:
                continue
            checked += 1
            if args.skip_availability:
                ok = None
            elif l["repo"].startswith("r2://"):
                ok = r2_available(s3, l["repo"])
            else:
                ok = hf_available(l["repo"], l["revision"])
            l["available"] = ok
            flag = "ok " if ok else ("?? " if ok is None else "GONE")
            report.append(f"  {flag} {l['record']} {l['at']} margin {l['margin']:+.4f} z {l['z']:+.2f} "
                          f"n {l['n_paired']} Bpass {l['b_pass']:.2f} forfeit {(l['forfeit_rate'] or 0):.3f} "
                          f"king {l['king_revision'][:12]} (bench {l['bench_k']:.2f})  {l['repo'][:64]}")
            if ok is False:
                continue
            seen_hk.add(l["hotkey"])
            l["stratum"] = name
            picked.append(l)
            taken += 1
        report.append(f"  -> took {taken} (checked {checked})")
        report.append("")

    report.append(f"panel: {len(picked)} losers")
    report.append(f"{'stratum':10} {'record':11} {'margin':>8} {'z':>6} {'bench_k':>7} repo")
    for l in picked:
        report.append(f"{l['stratum']:10} {l['record']:11} {l['margin']:+8.4f} {l['z']:+6.2f} "
                      f"{l['bench_k']:7.2f} {l['repo']}")
    report.append("")
    report.append("bench commands (one per model, ~0.3–1.0 h each on the bench pod):")
    for l in picked:
        report.append(f"  python affine/scripts/bench_run.py --repo '{l['repo']}' --revision {l['revision']} "
                      f"--label loser-{l['record'][5:]} --suite {SUITE}")
    report.append("")
    report.append("then: python research/scripts/v6_action_leg_panel.py --records "
                  + " ".join(l["record"] for l in picked) + " --n 400 --out research/results/v6_action_leg_losers")
    text = "\n".join(report) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(text)
    out.with_suffix(".json").write_text(json.dumps({"suite": SUITE, "strata": STRATA, "panel": picked,
                                                    "n_losers": len(losers), "n_usable": len(usable)}, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
