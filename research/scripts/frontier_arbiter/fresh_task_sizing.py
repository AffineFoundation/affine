"""Size G2 — a crown confirmation on FRESH verifiable tasks (2026-09-21).

Question: if every crown candidate had to beat the sitting king on freshly
generated tasks with programmatic graders (both models through the full agent
loop, paired on the same tasks, win at 2·SE), how many tasks would we need,
what would it cost, and which generators exist today?

Inputs (read-only; nothing here spends):
  * ops/kingboard/state/rollouts.sqlite from the validator box — one row per
    datagen rollout (source, harness, seat, model digest, task_uid, outcome,
    wall_s, tokens). The king seat replays the same tasks the teacher ran, so
    it already IS a paired design across kings 11-20 + genesis + teacher.
  * https://affine.io/api/v1/matrix — env cells (solve rate + n) per row.
  * rollouts/rollouts/sources.toml (catalog kinds) — read by hand for §1.

Outputs: research/results/frontier_arbiter/fresh_task/report.{txt,json}

Statistics (paired design, one rollout per task per model):
  d_t = solved_A(t) − solved_B(t) ∈ {−1, 0, +1};  Δ = mean(d);
  q = P(d ≠ 0) (discordance);  var_d = q − Δ² ≈ q;
  SE = sqrt(var_d / n);  crown bar "win at 2·SE"  ⇔  Δ ≥ 2·SE
  n(2·SE) = 4·var_d / Δ²          (Δ sits exactly on the bar; ~50 % power)
  n(80 %) = (1.96 + 0.84)²·var_d / Δ² = 7.85·var_d / Δ²
  unpaired reference: var_d = 2·p(1−p)  (independent tasks per model)
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "research/results/frontier_arbiter/fresh_task"

GRADED = ("solved", "failed")
DELTAS_PTS = (3, 5, 10)
POD_USD_PER_H = 3.0          # Lium 1x H200, one TP1 replica (suite.toml lium_plan)
WINDOW_S = 2 * 3600          # the confirmation must fit next to the ~2 h card
UTIL = 0.7                   # tail / stragglers: usable share of pod-time

# Family = the kingboard group. Per-pod concurrency for the AGENT loop of the
# family's heaviest live source (sources.toml max_concurrency; the default
# production batch is 48 slots for sandbox-free chat sources).
FAMILY_CONC = {"coding": 32, "terminal": 32, "tool_use": 32, "general": 32, "math": 64, "nl2repo": 16}

# Sources whose generator can mint unseen instances TODAY (see INVENTORY):
# the "ready" portfolio a first G2 could draw from without new code.
READY_TODAY = {
    "tool_use": ["affine_tau2_synth", "affine_kb_synth"],
    "general": ["affine_rgym", "affine_rcore", "affine_prolog", "affine_uuidctf", "affine_deshuffle", "affine_verbatim", "affine_wikispeedia"],
    "coding": ["swesmith"],
    "terminal": ["terminal_lego", "affine_tmax"],  # POOL, not FRESH: ~23k unseen pre-minted tasks
}

# ---------------------------------------------------------------- §1 inventory
# Eligibility = can mint NEW instances on demand with a programmatic grader.
#   FRESH      generator exists and is seedable: new instance per seed/duel
#   POOL       large pre-generated pool (thousands, unseen tasks available for
#              months) but no on-demand generator on our side
#   FIXED      fixed benchmark / dataset list — NOT eligible (RT-7)
#   BUILD      recipe exists, code to write
INVENTORY = [
    # name, family, eligibility, generation mechanism, grader, supply / gen cost, runtime, in D / king seat
    ("swesmith", "coding", "POOL→FRESH", "SWE-smith bug injection (lm_rewrite, func_pm_* AST mutators, PR-revert) on a repo snapshot with a passing test suite; HF shards hold pre-minted instances; the mutators re-run on any cached image", "hidden tests FAIL→PASS (PASS→PASS kept)", "6,960 uids used of ~50k shards (py/go/java/ts/rs); fresh mint ≈ 1 container-min per candidate + a test run to validate", "docker per task (430 repo images cached on pods)", "yes (share 4.0, king seat all 7 harnesses)"),
    ("r2e_gym", "coding", "FIXED (pool) / FRESH via SWE-Gen", "R2E-Gym-Subset-Verified: commit-derived tasks with synthesized tests (R2E pipeline = backtranslated issue + LLM tests); HF is a fixed 3.9k list; the R2E/SWE-Gen recipe mints from any commit history", "repo tests (docker)", "3,876 uids on 9 repos — fixed; new mint needs the R2E pipeline (LLM test synthesis ≈ $0.05–0.2/task + validation run)", "docker per task", "yes (king seat)"),
    ("swelego", "coding", "FIXED", "SWE-Lego-Real-Data-Verified: curated real PR tasks (868 repos)", "repo tests", "1,625 uids — fixed HF list", "docker", "yes"),
    ("multiswe / scaleswe / swerebench_v2", "coding", "FIXED", "curated real-PR datasets", "repo tests", "fixed lists (1k–2.5k)", "docker", "yes"),
    ("feature-deletion (Cursor recipe)", "coding", "BUILD", "pick a repo image with a green test suite (swesmith / multiswe / tmax images already cached); pick a feature = set of functions whose deleting turns k ≥ 1 tests red while the rest stay green; delete implementation, keep signatures/docs; task = re-implement so the red tests pass; issue text generated from the tests' names + docstrings (or the removed docstring)", "the k red tests → green; rest of suite stays green (PASS→PASS)", "unbounded: every (repo, function-set) is a new instance; mint cost = 2 test runs (~1–3 container-min); no LLM needed for the minimal variant", "docker per task (same images as swesmith)", "no — ~300 lines: static call-graph + coverage map + deleter + verifiers taskset"),
    ("terminal_lego", "terminal", "POOL", "Prime's Terminal-Lego: LLM-generated terminal tasks (task.toml + tests + Dockerfile) — pre-minted HF checkout; the generator is Prime's, not ours", "Harbor verifier (test script)", "12,915 uids used of ~23k; ~10k still unseen by every king", "docker build per task (local_docker_build)", "yes (share 3.0, king seat)"),
    ("affine_tmax", "terminal", "POOL", "Prime TMax: 14,600 Harbor terminal tasks (public prime-tasks repo), pre-minted", "Harbor verifier", "14,600 dirs; ~13k unseen", "docker build per task (2 GB images, 8 concurrent)", "yes (share 1.0, king seat)"),
    ("terminal_bench_2", "terminal", "FIXED", "89 hand-written tasks (benchmark)", "Harbor verifier", "89 — fixed, public benchmark", "docker", "yes"),
    ("procedural terminal (BUILD)", "terminal", "BUILD", "seeded generators over our images: file-system forensics (uuidctf pattern), log-parsing with a hidden answer, build-fix with injected compile error, data-wrangling with a generated CSV and hidden checksum", "exact answer / test script", "unbounded; ≈ 0 mint cost (pure code)", "docker", "partly: affine_uuidctf IS this (3,000-seed pool, `standard` tier, king seat)"),
    ("affine_tau2_synth", "tool_use", "FRESH (generator in repo)", "τ²-synth: issue-combination × persona × variant over 10 synthetic customer-service domains; the domain generator (mikasenghaas/tau2-synth) is seedable — new issue combos / personas / db rows per seed; DeepSeek customer sim on Engy", "τ² final-state DB check + action set (deterministic)", "≈ 11,600 pre-minted; 753 uids used; new combos = code + a few $ of Engy user-sim per rollout ($0.001–0.003)", "no container (in-process orchestrator + user-sim API)", "yes (share 3.0, king seat, king floor 8/h)"),
    ("affine_kb_synth", "tool_use", "FRESH (same generator)", "τ²-synth domains with policy moved into a BM25 knowledge base (tau3 KB_search shape)", "τ² final-state check", "same pool as tau2_synth (284 uids used)", "none", "yes (share 2.0, king seat)"),
    ("affine_when2call", "tool_use", "FIXED", "NVIDIA When2Call train_pref (8,154 rows)", "gold tool name + args / class", "fixed list", "none", "yes"),
    ("affine_agent", "tool_use", "FIXED", "Prime general_agent toolbench (4,417 tasks, per-task DB)", "DB-hash", "fixed list", "host-served tool DB", "yes"),
    ("affine_eog / autobench", "general/tool_use", "FIXED", "EnterpriseOps-Gym 649 / AutomationBench 600 oracle tasks", "final-state SQL / assertions", "fixed lists", "service containers / in-process", "yes"),
    ("affine_rgym (reasoning-gym)", "general", "FRESH", "~100 procedural puzzle generators, identity = (generator, index, level); infinite by construction", "per-generator exact scorer", "unbounded; 0 mint cost; level 1 live (levels 2–3 hit the 16k cap)", "none (chat)", "yes (share 1.0, king seat)"),
    ("affine_rcore (reasoning-core)", "general", "FRESH", "formally verified planning / logic / regex / math generators", "reasoning_core verifier", "unbounded (20k rows cached)", "none", "yes"),
    ("affine_prolog / uuidctf / deshuffle / verbatim / wikispeedia", "general", "FRESH", "per-index seeded procedural pools (3,000–5,000 seeds; the seed space is the generator)", "exact / solved", "unbounded by seed; live pools 3k–5k", "swipl / sandbox / none", "yes (king seat)"),
    ("affine_math / i3math / logic / science / trivia / ifeval / pydantic / i3code / sql", "math/general", "FIXED", "HF train splits", "boxed / exact / tests / JSON schema", "fixed lists (2k–100k rows)", "none or docker", "yes"),
    ("affine-repo-qa-v1 (hub-survey proposal C)", "coding", "BUILD", "generated question about a cached repo image whose answer is extracted by static analysis (raises X? call sites of Y? default of Z?)", "exact match after normalisation", "unbounded over cached images; 0 LLM cost", "docker (any shell harness)", "no — ~150 lines"),
    ("affine-resume-v1 (proposal B)", "coding/terminal", "BUILD", "continue from a king failure state (replayed shell history) — grader = original tests", "original verifier", "bounded by king failures (thousands)", "docker", "no; not fresh-by-construction (states come from the king's own runs)"),
    ("affine-compaction-v1 (proposal A)", "chat", "BUILD", "summarise a real datagen transcript, TEXT ONLY", "IFEval-style checks (paths mentioned, no tool call, ≤ N words)", "unbounded over traces", "none", "no"),
    ("Prime hub: synlogic / enigmata / absolute-zero / kakuro / nonogram", "general", "FRESH (hub)", "procedural puzzle envs with verifiers", "exact", "infinite", "none", "no (survey rank 27+)"),
    ("Prime hub: swe-gen / SWE-smith-live style repo-task synth", "coding", "FRESH (hub, needs port)", "issue-synthesis on arbitrary GitHub repos (LLM-authored tests)", "synthesized tests", "unbounded; LLM cost per task", "docker", "no"),
]


def load_rows(db: Path):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(
        "select source, grp, harness, seat, model_label, digest12, task_uid, outcome, "
        "wall_s, temperature, prompt_tokens, completion_tokens, stop, timeout from rollouts")]


def reign_map(matrix: dict) -> dict[str, int]:
    m = {}
    for r in matrix["rows"]:
        if r.get("kind") in ("king", "genesis") and r.get("digest12"):
            m[r["digest12"]] = int(r.get("reign", 0))
    return m


def side_of(row, reigns) -> str | None:
    """Which model produced this rollout: 'T' teacher, 'K<reign>' king/genesis, else None."""
    if row["seat"] == "teacher":
        return "T"
    if row["seat"] == "king" and row["digest12"] in reigns:
        if row["temperature"] == 0.0:
            return None  # greedy king rows are not pooled by the matrix either
        return f"K{reigns[row['digest12']]}"
    return None


def paired_stats(a: dict[str, list[int]], b: dict[str, list[int]]):
    """Per-task solve means for two sides on shared tasks → Δ, discordance, sd."""
    shared = a.keys() & b.keys()
    if not shared:
        return None
    d = [st.fmean(a[t]) - st.fmean(b[t]) for t in shared]
    n = len(d)
    delta = st.fmean(d)
    var_d = st.pvariance(d) if n > 1 else 0.0
    q = sum(1 for x in d if abs(x) > 1e-9) / n
    pa = st.fmean(st.fmean(a[t]) for t in shared)
    pb = st.fmean(st.fmean(b[t]) for t in shared)
    return {"n": n, "delta": delta, "var_d": var_d, "sd_d": math.sqrt(var_d), "discordance": q, "p_a": pa, "p_b": pb}


def n_needed(var_d: float, delta_pts: float):
    d = delta_pts / 100.0
    return {"n_2se": math.ceil(4 * var_d / d**2), "n_80pct": math.ceil((1.96 + 0.8416) ** 2 * var_d / d**2)}


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/ft/rollouts.sqlite")
    ap.add_argument("--matrix", default="/tmp/ft/matrix.json")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--min-pair", type=int, default=30, help="min shared tasks for a king pair")
    args = ap.parse_args()

    matrix = json.load(open(args.matrix))
    reigns = reign_map(matrix)
    rows = load_rows(Path(args.db))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_group = {r["source"]: r["grp"] for r in rows if r["grp"]}
    for c in matrix["columns"]:
        if c["kind"] == "env":
            env_group[c["key"].split(":", 1)[1]] = c["group"]

    # ---------------------------------------------------------- per-task tables
    # solved[source][side][task_uid] = [0/1, ...] over sampled graded rollouts
    solved = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["outcome"] not in GRADED:
            continue
        side = side_of(r, reigns)
        if side is None:
            continue
        solved[r["source"]][side][r["task_uid"]].append(1 if r["outcome"] == "solved" else 0)

    kings = sorted({s for src in solved.values() for s in src if s.startswith("K") and s != "K0"},
                   key=lambda s: int(s[1:]))
    kings_11_20 = [k for k in kings if 11 <= int(k[1:]) <= 20]

    # ------------------------------------------------- §2a king-to-king (paired)
    per_source_pairs = {}
    fam_pair_acc = defaultdict(list)  # family -> list of pair stats (weighted by n)
    for src, sides in solved.items():
        fam = env_group.get(src, "?")
        ks = [k for k in kings_11_20 if k in sides]
        pairs = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ps = paired_stats(sides[ks[i]], sides[ks[j]])
                if ps and ps["n"] >= args.min_pair:
                    ps.update({"a": ks[i], "b": ks[j]})
                    pairs.append(ps)
                    fam_pair_acc[fam].append(ps)
        if pairs:
            per_source_pairs[src] = {"family": fam, "pairs": pairs}

    def pool(pairs):
        n = sum(p["n"] for p in pairs)
        if not n:
            return None
        q = sum(p["discordance"] * p["n"] for p in pairs) / n
        abs_delta = sum(abs(p["delta"]) * p["n"] for p in pairs) / n
        max_delta = max(abs(p["delta"]) for p in pairs)
        var_d = sum(p["var_d"] * p["n"] for p in pairs) / n
        return {"n_pairs": len(pairs), "n_task_pairs": n, "discordance": q, "var_d": var_d,
                "mean_abs_delta_pts": 100 * abs_delta, "max_abs_delta_pts": 100 * max_delta,
                "median_abs_delta_pts": 100 * st.median(abs(p["delta"]) for p in pairs)}

    family_power = {}
    for fam, pairs in sorted(fam_pair_acc.items()):
        pooled = pool(pairs)
        pooled["n_for_delta"] = {str(dp): n_needed(pooled["var_d"], dp) for dp in DELTAS_PTS}
        family_power[fam] = pooled

    # same, restricted to the generators that can mint unseen tasks today
    ready_power = {}
    all_ready_pairs = []
    for fam, srcs in READY_TODAY.items():
        pairs = [p for s in srcs if s in per_source_pairs for p in per_source_pairs[s]["pairs"]]
        all_ready_pairs += pairs
        if pairs:
            pooled = pool(pairs)
            pooled["sources"] = [s for s in srcs if s in per_source_pairs]
            pooled["n_for_delta"] = {str(dp): n_needed(pooled["var_d"], dp) for dp in DELTAS_PTS}
            ready_power[fam] = pooled
    if all_ready_pairs:
        pooled = pool(all_ready_pairs)
        pooled["n_for_delta"] = {str(dp): n_needed(pooled["var_d"], dp) for dp in DELTAS_PTS}
        ready_power["ALL_READY"] = pooled

    # ------------------------------------------ §2b unpaired solve rates & teacher gap
    fam_rates = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # fam -> side -> [solved, n]
    src_rates = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for src, sides in solved.items():
        fam = env_group.get(src, "?")
        for side, tasks in sides.items():
            for outs in tasks.values():
                fam_rates[fam][side][0] += sum(outs)
                fam_rates[fam][side][1] += len(outs)
                src_rates[src][side][0] += sum(outs)
                src_rates[src][side][1] += len(outs)

    def rate(sn):
        return 100 * sn[0] / sn[1] if sn[1] else None

    family_rates = {}
    for fam, sides in fam_rates.items():
        entry = {s: {"rate": rate(sn), "n": sn[1]} for s, sn in sides.items()}
        king_rates = [entry[k]["rate"] for k in kings_11_20 if k in entry and entry[k]["n"] >= 30]
        entry["_summary"] = {
            "teacher": entry.get("T", {}).get("rate"),
            "genesis": entry.get("K0", {}).get("rate"),
            "kings_11_20_mean": st.fmean(king_rates) if king_rates else None,
            "kings_11_20_spread": (max(king_rates) - min(king_rates)) if king_rates else None,
            "n_kings": len(king_rates),
        }
        family_rates[fam] = entry

    # teacher vs king paired, per family (pool every king with >= min_pair shared tasks)
    teacher_gap = {}
    for fam in sorted(fam_rates):
        acc = []
        for src, sides in solved.items():
            if env_group.get(src) != fam or "T" not in sides:
                continue
            for k in kings_11_20:
                if k in sides:
                    ps = paired_stats(sides["T"], sides[k])
                    if ps and ps["n"] >= args.min_pair:
                        ps.update({"a": "T", "b": k, "source": src})
                        acc.append(ps)
        if acc:
            n = sum(p["n"] for p in acc)
            teacher_gap[fam] = {
                "n_task_pairs": n, "n_pairs": len(acc),
                "teacher_minus_king_pts": 100 * sum(p["delta"] * p["n"] for p in acc) / n,
                "discordance": sum(p["discordance"] * p["n"] for p in acc) / n,
                "share_pairs_teacher_ahead": sum(1 for p in acc if p["delta"] > 0) / len(acc),
                "by_king": {k: 100 * sum(p["delta"] * p["n"] for p in acc if p["b"] == k) / max(1, sum(p["n"] for p in acc if p["b"] == k))
                            for k in kings_11_20 if any(p["b"] == k for p in acc)},
            }

    # matrix env cells: spread across kings 11-20 per group
    matrix_spread = {}
    krow = {r["reign"]: r for r in matrix["rows"] if r.get("kind") == "king"}
    trow = next(r for r in matrix["rows"] if r["kind"] == "teacher")
    for env, fam in env_group.items():
        key = f"env:{env}"
        vals = {}
        for rg, r in krow.items():
            c = r["cells"].get(key)
            if c and c.get("score") is not None and not c.get("low_n"):
                vals[rg] = c["score"]
        tc = trow["cells"].get(key, {})
        if len(vals) >= 2:
            matrix_spread[env] = {"family": fam, "n_kings": len(vals), "min": min(vals.values()),
                                  "max": max(vals.values()), "spread": max(vals.values()) - min(vals.values()),
                                  "sd": st.pstdev(vals.values()), "teacher": tc.get("score"),
                                  "kings": vals}

    # ------------------------------------------------------------- §3 cost/time
    cost = {}
    fam_wall = defaultdict(list)
    for r in rows:
        if r["outcome"] not in GRADED or r["seat"] != "king" or r["wall_s"] is None:
            continue
        fam = env_group.get(r["source"], "?")
        cost.setdefault(fam, defaultdict(list))
        cost[fam][r["source"]].append(r)
        fam_wall[fam].append(r)

    def cost_block(rs, conc):
        walls = [x["wall_s"] for x in rs]
        pt = [x["prompt_tokens"] for x in rs if x["prompt_tokens"]]
        ct = [x["completion_tokens"] for x in rs if x["completion_tokens"]]
        mean_w = st.fmean(walls)
        per_pod_per_h = conc * 3600 / mean_w * UTIL
        usd_per_rollout = POD_USD_PER_H / per_pod_per_h
        return {"n": len(rs), "wall_p50_s": pctl(walls, 0.5), "wall_p90_s": pctl(walls, 0.9), "wall_mean_s": mean_w,
                "prompt_tok_mean": st.fmean(pt) if pt else None, "completion_tok_mean": st.fmean(ct) if ct else None,
                "concurrency_per_pod": conc, "rollouts_per_pod_hour": per_pod_per_h,
                "usd_per_rollout_pod": usd_per_rollout}

    cost_out = {}
    for fam, srcs in cost.items():
        conc = FAMILY_CONC.get(fam, 16)
        fb = cost_block(fam_wall[fam], conc)
        fb["by_source"] = {s: cost_block(rs, conc) for s, rs in srcs.items() if len(rs) >= 30}
        # pods for N tasks × 2 models within the window
        plan = {}
        fp = family_power.get(fam)
        if fp:
            for dp in DELTAS_PTS:
                for key in ("n_2se", "n_80pct"):
                    n = fp["n_for_delta"][str(dp)][key]
                    cap_per_pod = fb["rollouts_per_pod_hour"] * WINDOW_S / 3600
                    pods_per_model = max(1, math.ceil(n / cap_per_pod))
                    plan[f"delta{dp}_{key}"] = {
                        "tasks": n, "pods_total": 2 * pods_per_model,
                        "usd": 2 * pods_per_model * POD_USD_PER_H * WINDOW_S / 3600,
                        "usd_rollouts_only": 2 * n * fb["usd_per_rollout_pod"],
                    }
        fb["plan_2h"] = plan
        cost_out[fam] = fb

    # ------------------------------------------------------------------ report
    result = {
        "generated_from": {"db": args.db, "matrix": args.matrix, "matrix_generated_at": matrix.get("generated_at")},
        "definitions": {
            "paired_delta": "mean over shared tasks of solved_A − solved_B (one sampled T=0.8 rollout per task per model unless the seat replayed)",
            "discordance": "share of shared tasks where the two models disagree (exactly one solved) = var of d when Δ≈0",
            "n_2se": "tasks for Δ to sit exactly on the 2·SE crown bar (≈50 % power)",
            "n_80pct": "tasks for 80 % power at the one-sided 2·SE bar",
            "rollouts_per_pod_hour": "concurrency × 3600 / mean wall × 0.7 utilisation, king agent loop on one 1×H200 pod",
            "usd_per_rollout_pod": "pod $/h ÷ rollouts per pod-hour (containers run on the pod; no sandbox fee)",
            "plan_2h": "pods for N tasks × 2 models inside 2 h and their rental cost; usd_rollouts_only = marginal compute if the pods are shared",
        },
        "kings": {"reigns_by_digest": reigns, "sides_used": kings_11_20},
        "inventory": [dict(zip(("name", "family", "eligibility", "generation", "grader", "supply", "runtime", "in_d_king_seat"), t)) for t in INVENTORY],
        "family_power": family_power,
        "ready_today_power": ready_power,
        "per_source_pairs": {s: {"family": v["family"], "pairs": v["pairs"]} for s, v in per_source_pairs.items()},
        "family_rates": family_rates,
        "teacher_gap_paired": teacher_gap,
        "matrix_env_spread": matrix_spread,
        "cost": cost_out,
        "constants": {"POD_USD_PER_H": POD_USD_PER_H, "WINDOW_S": WINDOW_S, "UTIL": UTIL, "FAMILY_CONC": FAMILY_CONC},
    }
    (out / "report.json").write_text(json.dumps(result, indent=1, default=str))
    (out / "report.txt").write_text(render(result))
    print(render(result))


def render(res) -> str:
    L = []
    P = L.append
    P("G2 fresh-task confirmation — sizing from live datagen rollouts (2026-09-21)")
    P(f"source: {res['generated_from']['db']} + matrix {res['generated_from']['matrix_generated_at']}")
    P("")
    P("== §1 generator inventory (FRESH = mints new instances on demand; POOL = large pre-minted pool; FIXED = not eligible) ==")
    for it in res["inventory"]:
        P(f"- {it['name']} [{it['family']}] {it['eligibility']}")
        P(f"    gen: {it['generation']}")
        P(f"    grader: {it['grader']} | supply: {it['supply']}")
        P(f"    runtime: {it['runtime']} | in D / king seat: {it['in_d_king_seat']}")
    P("")
    P("== §2a separation power — king-to-king PAIRED deltas (reigns 11-20, same tasks, sampled rollouts) ==")
    P(f"{'family':10} {'pairs':>5} {'taskpairs':>9} {'|Δ| med':>8} {'|Δ| mean':>8} {'|Δ| max':>8} {'discord':>8} {'var_d':>6} | tasks for Δ=3/5/10 pts: n(2SE) [n(80%)]")
    for fam, fp in res["family_power"].items():
        nf = fp["n_for_delta"]
        P(f"{fam:10} {fp['n_pairs']:>5} {fp['n_task_pairs']:>9} {fp['median_abs_delta_pts']:>8.1f} {fp['mean_abs_delta_pts']:>8.1f} {fp['max_abs_delta_pts']:>8.1f} {fp['discordance']:>8.3f} {fp['var_d']:>6.3f} | "
          + " / ".join(f"{nf[str(d)]['n_2se']} [{nf[str(d)]['n_80pct']}]" for d in DELTAS_PTS))
    P("")
    P("   restricted to generators that can mint UNSEEN tasks today (READY_TODAY):")
    for fam, fp in res["ready_today_power"].items():
        nf = fp["n_for_delta"]
        P(f"   {fam:10} {fp['n_pairs']:>5} {fp['n_task_pairs']:>9} {fp['median_abs_delta_pts']:>8.1f} {fp['mean_abs_delta_pts']:>8.1f} {fp['max_abs_delta_pts']:>8.1f} {fp['discordance']:>8.3f} {fp['var_d']:>6.3f} | "
          + " / ".join(f"{nf[str(d)]['n_2se']} [{nf[str(d)]['n_80pct']}]" for d in DELTAS_PTS)
          + (f"  ({', '.join(fp['sources'])})" if fp.get("sources") else ""))
    P("")
    P("   per-source pairs (n ≥ 30 shared tasks):")
    for src, v in sorted(res["per_source_pairs"].items(), key=lambda kv: (kv[1]["family"], kv[0])):
        ps = sorted(v["pairs"], key=lambda p: -p["n"])[:6]
        P(f"   {src:20} [{v['family']}] " + "; ".join(f"{p['a']}-{p['b']} n={p['n']} Δ={100*p['delta']:+.1f} q={p['discordance']:.2f}" for p in ps))
    P("")
    P("== §2b solve rates by family (sampled rollouts pooled over the family's envs; % solved) ==")
    P(f"{'family':10} {'teacher':>8} {'genesis':>8} {'kings μ':>8} {'kings spread':>12} {'n_kings':>7} | per king (rate/n)")
    for fam, e in sorted(res["family_rates"].items()):
        s = e["_summary"]
        ks = " ".join(f"{k}:{e[k]['rate']:.0f}/{e[k]['n']}" for k in res["kings"]["sides_used"] if k in e)
        f = lambda x: f"{x:8.1f}" if x is not None else f"{'-':>8}"
        P(f"{fam:10} {f(s['teacher'])} {f(s['genesis'])} {f(s['kings_11_20_mean'])} {f(s['kings_11_20_spread']):>12} {s['n_kings']:>7} | {ks}")
    P("")
    P("== §2c teacher vs king, PAIRED on shared tasks (teacher − king, points) ==")
    for fam, g in res["teacher_gap_paired"].items():
        bk = " ".join(f"{k}:{v:+.1f}" for k, v in g["by_king"].items())
        P(f"{fam:10} n={g['n_task_pairs']:>6} pairs={g['n_pairs']:>3} T−K={g['teacher_minus_king_pts']:+.1f} discord={g['discordance']:.2f} teacher ahead in {100*g['share_pairs_teacher_ahead']:.0f}% of pairs | {bk}")
    P("")
    P("== §2d matrix env cells: spread across kings 11-20 (unpaired solve rates, low_n cells excluded) ==")
    for env, m in sorted(res["matrix_env_spread"].items(), key=lambda kv: (kv[1]["family"], -kv[1]["spread"])):
        t = f"{m['teacher']:.0f}" if m["teacher"] is not None else "-"
        P(f"   {env:20} [{m['family']:8}] kings={m['n_kings']} min={m['min']:.0f} max={m['max']:.0f} spread={m['spread']:.0f} sd={m['sd']:.1f} teacher={t}")
    P("")
    P("== §3 cost / time (king agent loop, sampled graded rollouts; pod = 1×H200 $3/h, containers on the pod) ==")
    P(f"{'family':10} {'n':>6} {'wall p50':>8} {'p90':>7} {'mean':>7} {'ptok':>8} {'ctok':>7} {'conc':>4} {'ro/pod-h':>8} {'$/rollout':>9}")
    for fam, c in sorted(res["cost"].items()):
        P(f"{fam:10} {c['n']:>6} {c['wall_p50_s']:>8.0f} {c['wall_p90_s']:>7.0f} {c['wall_mean_s']:>7.0f} {c['prompt_tok_mean'] or 0:>8.0f} {c['completion_tok_mean'] or 0:>7.0f} {c['concurrency_per_pod']:>4} {c['rollouts_per_pod_hour']:>8.0f} {c['usd_per_rollout_pod']:>9.4f}")
        for s, b in sorted(c["by_source"].items(), key=lambda kv: -kv[1]["n"]):
            P(f"   {s:22} n={b['n']:>5} p50={b['wall_p50_s']:>5.0f}s p90={b['wall_p90_s']:>5.0f}s ptok={b['prompt_tok_mean'] or 0:>7.0f} ctok={b['completion_tok_mean'] or 0:>6.0f} ro/pod-h={b['rollouts_per_pod_hour']:>5.0f} $/ro={b['usd_per_rollout_pod']:.4f}")
    P("")
    P("   pods × 2 models inside 2 h (tasks per model → pods total, rental $ for the window, marginal compute $):")
    for fam, c in sorted(res["cost"].items()):
        if not c["plan_2h"]:
            continue
        parts = []
        for dp in DELTAS_PTS:
            a = c["plan_2h"][f"delta{dp}_n_2se"]
            b = c["plan_2h"][f"delta{dp}_n_80pct"]
            parts.append(f"Δ{dp}: {a['tasks']}t→{a['pods_total']}pods ${a['usd']:.0f} (${a['usd_rollouts_only']:.0f}) | 80%: {b['tasks']}t→{b['pods_total']}pods ${b['usd']:.0f} (${b['usd_rollouts_only']:.0f})")
        P(f"   {fam:10} " + " || ".join(parts))
    P("")
    P(MECHANISM)
    P("definitions: " + "; ".join(f"{k} = {v}" for k, v in res["definitions"].items()))
    return "\n".join(L) + "\n"


MECHANISM = """== §4 mechanism design ==
(a) gaming surface
  - reward hacking on synthetic tasks: the graders are the same tests the agent can read; a miner can special-case
    (edit tests, hard-code expected outputs, `pytest -p no:...`, delete the failing test). Mitigations already in
    our harnesses: tests re-copied from the task image before grading (SWE-smith / Harbor pattern), PASS→PASS set
    must stay green, network_allow=[] for chat-shaped sets, time/turn budgets. Feature-deletion adds: grade on the
    ORIGINAL test files (not the working tree), reject diffs that touch test paths, require the removed symbols'
    signatures to be present (a stub that returns the expected constant fails on ≥ 2 red tests by construction if
    the k tests exercise ≥ 2 distinct inputs — pick features with k ≥ 3).
  - generator predictability: a public generator with a private per-duel seed is FINE — training on the
    distribution IS the skill (rgym, tau2-synth personas, mutator families). NOT fine: finite pools that repeat
    (terminal_lego / tmax / swesmith shards: a miner can SFT the whole pool; ~23k+14k+50k public tasks). Rule:
    an instance may enter a confirmation at most once, ever (published task ids make this auditable), and the seed
    = block hash of the crown candidate's reveal, like the duel slice seed.
  - king-copy: an ε-copy ties (Δ≈0, discordance ≈ sampling noise only) → cannot win at 2·SE → no crown. Good.
    Side effect: under `must-win` a genuinely equal model never crowns either — that is the intent (δ analogue).
  - difficulty drift between duels: paired design cancels it (both models see the same instances); the published
    yardstick (teacher solve rate on the same instances) makes the drift visible across verdicts.
  - variance games: a challenger can raise its own per-task variance (T=0.8 sampling) — irrelevant for the paired
    Δ but raises SE; must-win therefore favours DETERMINISTIC-ish challengers. Fix: one rollout per task per model
    at T=0 for the confirmation, or k=2 and average.
  - harness games: the confirmation must run the challenger through the SAME harness set the datagen king seat
    uses (textbased / bash / pi / claude_code / terminus / toolcall), otherwise a miner tunes to one scaffold.
(b) composition with the dense meter
  - filter: dense duel must be won at 2·SE + δ on 1,300 turns (unchanged) — cheap, ~30 min.
  - gate (must not lose): confirmation Δ ≥ −2·SE. Blocks regressions only; a plateaued equal model still crowns
    on the dense meter alone → does not break the plateau. Cheapest in tasks (Δ=0 needs no power, only the SE).
  - must-win: Δ ≥ +2·SE (like the wvk-19 turn slice). Breaks the plateau: crowns only real capability gains of
    ≥ ~5 pts on ~500–850 tasks. Risk: the board freezes when nobody beats the king on tasks — acceptable if the
    king is at teacher level (then the freeze is the honest state); pair with the decaying-δ idea for liveness.
  - blend: rank = dense z + λ·task z. Re-opens Goodhart on the dense side (a big dense margin buys a task loss);
    not recommended.
  - recommendation: gate first (one crown cycle, no contract risk), then must-win once the ready families have
    published yardsticks.
(c) replayability: per verdict publish task ids + seeds + generator commit + task images' digests, both models'
    full traces (already the datagen trace envelope), grader outputs and the Δ / SE / n; anyone can re-run.
(d) 'frontier level': the teacher runs the same fresh instances (3rd model, +50 % cost) and its solve rate is
    published next to king/challenger; 'frontier' = challenger − teacher ≥ 0 at 2·SE on the paired tasks. Today the
    teacher is ahead of every king on every family (§2c: +6 math … +27 tool_use), so this yardstick has room.
"""


if __name__ == "__main__":
    main()
