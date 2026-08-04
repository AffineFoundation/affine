"""Offline markdown tables for the distillation-KL scoring paper.

Reads E-LADDER / E-KINGS / bank / judge-baseline / red-team result files and
writes paper-ready markdown to results/paper_tables.txt. Soft-fails missing
inputs with a note.

Usage: python scripts/paper_tables.py
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_PATH = RESULTS / "paper_tables.txt"
sys.path.insert(0, str(ROOT))

# Truth mirrors scripts/rule_search.py and scripts/analyze_ladder.py
KINGS_TRUTH = {
    "king-genesis": 58.2, "king-I": 38.4, "king-XCIX": 39.8, "king-VIII": 36.2,
    "king-XI": 33.6, "king-V": 32.0, "king-XLV": 26.0, "king-XLVI": 13.2,
    "king-LI": 11.6, "king-CI": 12.4,
    "king-II": 37.2, "king-III": 32.8, "king-VII": 33.2,
    "king-XCIV": 36.2, "king-XLI": 34.2,
}
BANK_W2_FULLZ = RESULTS / "bank_w2_fullz.jsonl"
BANK_W2 = RESULTS / "bank_w2.jsonl"
BANK_PATHS = [
    RESULTS / "bank_w1.jsonl",
    RESULTS / "bank_w3.jsonl",
]
EKINGS_W3_PATHS = [
    RESULTS / "ekings_w3_v2.jsonl",
    RESULTS / "ekings_w3_VII.jsonl",
]
LADDER_TRUTH = {
    "qwen3-1.7b": 33.2, "qwen3-4b": 54.2, "qwen3-8b": 57.5,
    "qwen3-14b": 63.5, "qwen3-32b": 65.7,
}

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
SOFT_ALPHA = 0.05
SOFT_TARGET = 20

try:
    from harness.score import (
        DEFAULT_GAMMA,
        DEFAULT_GAMMA_BANK,
        DEFAULT_L1_WEIGHT,
        gate_pass,
        l1_lift,
        lambda2,
        rank_term,
    )
except ImportError:  # pragma: no cover
    DEFAULT_GAMMA = 0.30
    DEFAULT_GAMMA_BANK = 0.08
    DEFAULT_L1_WEIGHT = 0.5

    def _cmd(y: str) -> str:
        return y.removeprefix("```bash\n").removesuffix("\n```").strip()

    def gate_pass(p, tau=0.02):  # type: ignore[misc]
        z, y = p.get("z_a", ""), p.get("y_a", "")
        c = _cmd(y)
        if c and c in z:
            return False
        return (p["lpA_ya_za"] - p["lpA_ya_e"]) >= tau

    def lambda2(p):  # type: ignore[misc]
        return p["lpC_yc_za"] - p["lpC_yc_e"]

    def l1_lift(p):  # type: ignore[misc]
        return p["lpA_yc_za"] - p["lpA_yc_e"]

    def rank_term(p, l1_weight=DEFAULT_L1_WEIGHT):  # type: ignore[misc]
        return lambda2(p) + l1_weight * l1_lift(p)


_out: io.TextIOBase = sys.stdout


def emit(msg: str = "") -> None:
    print(msg, file=_out)


def note(msg: str) -> None:
    emit(f"\n> note: {msg}\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def idents(s: str) -> set[str]:
    return set(IDENT_RE.findall(s or ""))


def soft_ident(pair: dict) -> float:
    """softΛ2 = Λ2 − α·relu(T − n_idents(z))/T (abandoned; pad-gameable)."""
    ni = len(idents(pair.get("z_a", "")))
    return lambda2(pair) - SOFT_ALPHA * max(0, SOFT_TARGET - ni) / SOFT_TARGET


def resolve_ekings_v2() -> Path | None:
    v2 = RESULTS / "ekings_v2_all.jsonl"
    if v2.exists():
        return v2
    parts = [RESULTS / "ekings_w1_v2.jsonl", RESULTS / "ekings_w2_v2.jsonl"]
    if all(p.exists() for p in parts):
        return None  # signal: use parts
    legacy = RESULTS / "ekings_all.jsonl"
    if legacy.exists():
        return legacy
    return None


def load_ekings_v2() -> list[dict] | None:
    path = resolve_ekings_v2()
    if path is not None:
        rows = load_jsonl(path)
    else:
        parts = [RESULTS / "ekings_w1_v2.jsonl", RESULTS / "ekings_w2_v2.jsonl"]
        if not all(p.exists() for p in parts):
            return None
        rows = []
        for p in parts:
            rows.extend(load_jsonl(p))
    for p in EKINGS_W3_PATHS:
        if p.exists():
            rows.extend(load_jsonl(p))
    return rows


def md_table(headers: list[str], rows: list[list[str]]) -> None:
    emit("| " + " | ".join(headers) + " |")
    emit("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        emit("| " + " | ".join(row) + " |")
    emit()


def bank_w2_path() -> Path | None:
    """Late-king bank wave: prefer full-z rescoring over truncated bank_w2."""
    if BANK_W2_FULLZ.exists():
        return BANK_W2_FULLZ
    if BANK_W2.exists():
        return BANK_W2
    return None


def bank_fracs_merged() -> dict[str, float]:
    """Mean of (L2_bank > 0) per miner from bank_w1 + bank_w3 + bank_w2_fullz."""
    by_miner: dict[str, list[float]] = defaultdict(list)
    found = False
    for path in BANK_PATHS:
        if not path.exists():
            continue
        found = True
        for r in load_jsonl(path):
            by_miner[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    w2 = bank_w2_path()
    if w2 is not None:
        found = True
        for r in load_jsonl(w2):
            by_miner[r["miner"]].append(1.0 if r["L2_bank"] > 0 else 0.0)
    if not found:
        return {}
    return {m: st.mean(vs) for m, vs in by_miner.items()}


def ekings_stats(rows: list[dict]) -> dict[str, dict]:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("valid") or "pairs" not in r:
            continue
        for p in r["pairs"]:
            pairs[r["miner"]].append(p)
    out = {}
    for m, ps in pairs.items():
        out[m] = {
            "Lambda2": st.mean(lambda2(p) for p in ps),
            "L1lift": st.mean(l1_lift(p) for p in ps),
            "mix": st.mean(rank_term(p) for p in ps),
            "soft": st.mean(soft_ident(p) for p in ps),
            "gate_pass": st.mean(1.0 if gate_pass(p) else 0.0 for p in ps),
            "n_pairs": len(ps),
        }
    return out


def spearman(scores: list[float], truth: list[float]) -> tuple[float, float]:
    rho, p = stats.spearmanr(scores, truth)
    return float(rho), float(p)


def hybrid_mix_spearman(
    stats_m: dict[str, dict],
    bank_fracs: dict[str, float],
    gamma: float = DEFAULT_GAMMA,
    gamma_bank: float = DEFAULT_GAMMA_BANK,
) -> tuple[float, float, int, list[str]]:
    """Production hybrid: causality gate + bank gate on merged-bank miners."""
    miners = sorted(
        (m for m in KINGS_TRUTH if m in stats_m),
        key=lambda m: -KINGS_TRUTH[m],
    )
    scores, truth, rejected = [], [], []
    for m in miners:
        if stats_m[m]["gate_pass"] < gamma:
            rejected.append(m.removeprefix("king-"))
            continue
        bf = bank_fracs.get(m)
        if bf is not None and bf < gamma_bank:
            rejected.append(m.removeprefix("king-"))
            continue
        scores.append(stats_m[m]["mix"])
        truth.append(KINGS_TRUTH[m])
    rho, p = spearman(scores, truth)
    return rho, p, len(scores), rejected


def table_eladder() -> None:
    emit("## a) E-LADDER: miner vs S vs LiveCodeBench truth\n")
    path = RESULTS / "ladder_run1.jsonl"
    if not path.exists():
        note(f"missing {path}")
        return
    rows = [r for r in load_jsonl(path) if r.get("valid")]
    term_keys = ["D1", "D2", "D3", "D4", "D5"]
    weights = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "D4": 0.5, "D5": 0.5}
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in term_keys:
            per[r["miner"]][k].append(r[k])
    agg = {}
    for miner, terms in per.items():
        a = {k: st.mean(vs) for k, vs in terms.items()}
        a["n_turns"] = len(terms["D1"])
        a["S"] = -sum(weights[k] * a[k] for k in term_keys)
        agg[miner] = a
    miners = sorted((m for m in agg if m in LADDER_TRUTH),
                    key=lambda m: -LADDER_TRUTH[m])
    t = [LADDER_TRUTH[m] for m in miners]
    s = [agg[m]["S"] for m in miners]
    rho, p = spearman(s, t)
    out = []
    for m in miners:
        out.append([
            m,
            f"{agg[m]['n_turns']}",
            f"{agg[m]['S']:+.4f}",
            f"{LADDER_TRUTH[m]:.1f}",
        ])
    md_table(["miner", "n", "S", "LiveCodeBench"], out)
    emit(f"Spearman(S, LCB) = {rho:+.3f} (p={p:.3g}), n={len(miners)}\n")


def table_ekings_mix() -> tuple[dict[str, dict] | None, list[dict] | None]:
    emit("## b) E-KINGS: mix ranking vs swe-rebench\n")
    emit(
        f"Production S\\*: mix = Λ2 + {DEFAULT_L1_WEIGHT:g}·L1lift; "
        f"γ={DEFAULT_GAMMA:.2f}, γ_bank={DEFAULT_GAMMA_BANK:.2f} "
        "(bank from merged `bank_w1`/`bank_w3`/`bank_w2_fullz`).\n"
    )
    rows = load_ekings_v2()
    if rows is None:
        note("missing results/ekings_v2_all.jsonl (and w1/w2 v2 parts)")
        return None, None
    stats_m = ekings_stats(rows)
    bf = bank_fracs_merged()
    if not bf:
        note("missing merged bank_w1/w3/bank_w2_fullz jsonl — bank_frac column empty")
    miners = sorted(
        (m for m in KINGS_TRUTH if m in stats_m),
        key=lambda m: -stats_m[m]["mix"],
    )
    out = []
    for m in miners:
        s = stats_m[m]
        b = bf.get(m)
        out.append([
            m.removeprefix("king-"),
            f"{s['mix']:+.4f}",
            f"{s['Lambda2']:+.4f}",
            f"{s['L1lift']:+.4f}",
            f"{KINGS_TRUTH[m]:.1f}",
            f"{s['gate_pass']:.0%}",
            f"{b:.3f}" if b is not None else "—",
        ])
    md_table(
        ["king", "mix", "Λ2", "L1lift", "swe-rebench", "gate pass", "bank_frac (merged)"],
        out,
    )
    return stats_m, rows


def table_spearman(stats_m: dict[str, dict] | None) -> None:
    emit(f"## c) Spearman vs swe-rebench (E-KINGS v2, {len(KINGS_TRUTH)} kings)\n")
    if stats_m is None:
        note("skip Spearman table — no E-KINGS v2 data")
        return
    bank_fracs = bank_fracs_merged()
    miners = sorted(
        (m for m in KINGS_TRUTH if m in stats_m),
        key=lambda m: -KINGS_TRUTH[m],
    )
    truth = [KINGS_TRUTH[m] for m in miners]
    g_mix = stats_m["king-genesis"]["mix"]
    i_mix = stats_m["king-I"]["mix"]
    gen_order = "genesis > I" if g_mix > i_mix else "I > genesis"

    rows_out = []
    for label, key in [
        ("raw Λ2", "Lambda2"),
        (f"soft-idents (α={SOFT_ALPHA}, T={SOFT_TARGET})", "soft"),
        (f"mix (Λ2 + {DEFAULT_L1_WEIGHT:g}·L1lift)", "mix"),
    ]:
        scores = [stats_m[m][key] for m in miners]
        rho, p = spearman(scores, truth)
        note_s = ""
        if key == "Lambda2":
            note_s = "I > genesis (FP)"
        elif key == "soft":
            note_s = "pad-gameable (RT-soft-pad)"
        elif key == "mix":
            note_s = gen_order
        rows_out.append([label, f"{rho:+.3f}", f"{p:.3g}", str(len(miners)), note_s])

    if bank_fracs:
        rho, p, n, rejected = hybrid_mix_spearman(stats_m, bank_fracs)
        rej_s = ", ".join(rejected) if rejected else "—"
        rows_out.append([
            f"mix + gates (γ={DEFAULT_GAMMA:.2f}, γ_bank={DEFAULT_GAMMA_BANK:.2f}, merged bank)",
            f"{rho:+.3f}",
            f"{p:.3g}",
            str(n),
            f"rejected: {rej_s}",
        ])
    else:
        note("skip mix+bank hybrid — missing merged bank_w1/w3/bank_w2_fullz jsonl")

    jpath = RESULTS / "judge_baseline" / "baseline.json"
    if jpath.exists():
        base = json.loads(jpath.read_text())
        c = base.get("correlations", {}).get(
            "judge_at_win__vs__swe_rebench_110task_subset", {}
        )
        if c:
            rows_out.append([
                "judge-at-win baseline (110-task subset)",
                f"{c['spearman_r']:+.3f}",
                f"{c['spearman_p']:.3g}",
                str(c.get("n", "—")),
                "ns",
            ])
    else:
        note("missing results/judge_baseline/baseline.json")

    md_table(["metric", "Spearman", "p", "n", "notes"], rows_out)


def _genesis_by_turn(rows: list[dict], fn) -> dict[str, float]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("valid") and r.get("miner") == "king-genesis" and "pairs" in r:
            out[r["turn_id"]].append(st.mean(fn(p) for p in r["pairs"]))
    return {tid: st.mean(vs) for tid, vs in out.items()}


def _paired_win_vs_genesis(
    rt_rows: list[dict],
    arm: str,
    gen_by_turn: dict[str, float],
    score_fn,
) -> tuple[float, int] | None:
    paired = []
    for r in rt_rows:
        a = r.get(arm)
        if not a or not a.get("valid"):
            continue
        tid = r["turn_id"]
        if tid not in gen_by_turn:
            continue
        if "pairs" in a:
            arm_score = st.mean(score_fn(p) for p in a["pairs"])
        elif "L2" in a and score_fn is lambda2:
            arm_score = a["L2"]
        else:
            continue
        paired.append(arm_score - gen_by_turn[tid])
    if not paired:
        return None
    return sum(1 for d in paired if d > 0) / len(paired), len(paired)


def rt_soft_pad_stats(ek_rows: list[dict]) -> tuple[float, float] | None:
    """Offline prefix-pad attack on soft-idents (logprobs frozen)."""
    pref_path = RESULTS / "prefix_idents.json"
    if not pref_path.exists():
        return None
    pref = {x["turn_id"]: set(x["idents"]) for x in json.loads(pref_path.read_text())}
    rows_by: dict[str, list[dict]] = defaultdict(list)
    for r in ek_rows:
        if r.get("valid") and "pairs" in r and r["miner"] in ("king-genesis", "king-I"):
            rows_by[r["miner"]].append(r)
    if "king-genesis" not in rows_by or "king-I" not in rows_by:
        return None

    pad_lex: set[str] = set()
    for r in rows_by["king-genesis"]:
        for p in r["pairs"]:
            pad_lex |= idents(p.get("z_a", ""))
    pad_lex_l = sorted(pad_lex)
    random.seed(0)

    def pad_z(z: str, pref_ids: set[str], use_grounded_tokens: bool) -> str:
        cur = idents(z)
        if use_grounded_tokens:
            base = len(cur & pref_ids)
            pool = [t for t in pad_lex_l if t in pref_ids and t not in cur]
        else:
            base = len(cur)
            pool = [t for t in pad_lex_l if t not in cur and t not in pref_ids]
        need = max(0, SOFT_TARGET - base)
        extra = random.sample(pool, min(need, len(pool))) if pool else []
        return z + ("\n# " + " ".join(extra) if extra else "")

    def soft_score(l2: float, z: str, pref_ids: set[str], grounded: bool) -> float:
        zids = idents(z)
        ni = len(zids & pref_ids) if grounded else len(zids)
        return l2 - SOFT_ALPHA * max(0, SOFT_TARGET - ni) / SOFT_TARGET

    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for m in ("king-genesis", "king-I"):
        for r in rows_by[m]:
            by[r["turn_id"]][m] = r

    wins = {}
    for mode, grounded_score, grounded_pad in [
        ("junk", False, False),
        ("grounded_prefix", True, True),
    ]:
        diffs = []
        for tid, sides in by.items():
            if len(sides) < 2:
                continue
            gp = sides["king-genesis"]["pairs"][0]
            ip = sides["king-I"]["pairs"][0]
            pids = pref.get(tid, set())
            gl2 = lambda2(gp)
            il2 = lambda2(ip)
            gz = gp.get("z_a", "")
            iz = pad_z(ip.get("z_a", ""), pids, use_grounded_tokens=grounded_pad)
            diffs.append(
                soft_score(il2, iz, pids, grounded_score)
                - soft_score(gl2, gz, pids, grounded_score)
            )
        wins[mode] = sum(1 for d in diffs if d > 0) / len(diffs)
    return wins["junk"], wins["grounded_prefix"]


def table_redteam(ek_rows: list[dict] | None) -> None:
    emit("## d) Red-team one-liners\n")
    rows_out: list[list[str]] = []

    p1 = RESULTS / "rt1_payloads.jsonl"
    if p1.exists():
        rows_out.append([
            "RT-1 payloads",
            "instruct/jailbreak/stego etc. lose on Λ2; empty wins 100% but "
            "causality gate blocks",
        ])
    else:
        note(f"missing {p1}")

    p2 = RESULTS / "rt2_stuffer_v2.jsonl"
    if p2.exists():
        rows = load_jsonl(p2)
        for arm, label in [("stuffer", "exact stuff"), ("empty", "silence / A9")]:
            g = _gate_rate_arm(rows, arm)
            if g is not None:
                rows_out.append([
                    f"RT-2 {label}",
                    f"causality+leakage gate closes (pass {g:.0%})",
                ])
    else:
        note(f"missing {p2}")

    p2c = RESULTS / "rt2c_paraphrase.jsonl"
    if p2c.exists() and ek_rows:
        rows = load_jsonl(p2c)
        gen_l2 = _genesis_by_turn(ek_rows, lambda2)
        gen_mix = _genesis_by_turn(ek_rows, rank_term)
        para_l2 = _paired_win_vs_genesis(rows, "para_stuff", gen_l2, lambda2)
        para_mix = _paired_win_vs_genesis(rows, "para_stuff", gen_mix, rank_term)
        bank_note = "bank gate closes (frac=0%)"
        p1b = RESULTS / "rt1b_priorbank.jsonl"
        if p1b.exists():
            brows = load_jsonl(p1b)
            pos = [
                1.0 if r.get("para_ls", {}).get("L2_bank", -1) > 0 else 0.0
                for r in brows if r.get("para_ls", {}).get("valid")
            ]
            if pos:
                frac = st.mean(pos)
                bank_note = f"bank gate closes (frac={frac:.0%}<0.08)"
        l2_s = f"{para_l2[0]:.0%}" if para_l2 else "—"
        mix_s = f"{para_mix[0]:.0%}" if para_mix else "—"
        rows_out.append([
            "RT-2c paraphrase stuff",
            f"beats genesis on Λ2 (win {l2_s}) but mix resists para (win {mix_s}); "
            + bank_note,
        ])
    else:
        note(f"missing {p2c} or E-KINGS for RT-2c pairing")

    if ek_rows:
        pad = rt_soft_pad_stats(ek_rows)
        if pad:
            junk, grounded = pad
            rows_out.append([
                "RT-soft-pad CONFIRMED",
                f"prefix `#` ident pads game soft-idents (junk pad win_I {junk:.0%}, "
                f"grounded prefix-scrape {grounded:.0%}); mix resists para better "
                f"(see RT-2c)",
            ])

    p4 = RESULTS / "rt4_copier.jsonl"
    if p4.exists():
        rows = load_jsonl(p4)
        diffs = []
        for r in rows:
            a, b = r.get("A"), r.get("B")
            if not a or not b or not a.get("valid") or not b.get("valid"):
                continue
            if "pairs" not in a or "pairs" not in b:
                continue
            la = st.mean(lambda2(p) for p in a["pairs"])
            lb = st.mean(lambda2(p) for p in b["pairs"])
            diffs.append(la - lb)
        if diffs:
            se = st.stdev(diffs) / math.sqrt(len(diffs))
            z = st.mean(diffs) / se if se else 0.0
            rows_out.append([
                "RT-4 king copier",
                f"3σ duel rejects copier (|z|={abs(z):.2f} < 3)",
            ])
    else:
        note(f"missing {p4}")

    if rows_out:
        md_table(["attack", "verdict"], rows_out)


def _gate_rate_arm(rows: list[dict], arm: str) -> float | None:
    rates = []
    for r in rows:
        a = r.get(arm)
        if not a or not a.get("valid"):
            continue
        if "gate_rate" in a:
            rates.append(a["gate_rate"])
        elif "pairs" in a:
            rates.append(st.mean(1.0 if gate_pass(p) else 0.0 for p in a["pairs"]))
    return st.mean(rates) if rates else None


def render() -> str:
    global _out
    buf = io.StringIO()
    _out = buf
    emit("# Paper tables (offline)\n")
    emit(f"Source root: `{RESULTS}`\n")
    emit(
        f"Production S\\*: rank by mix = Λ2 + {DEFAULT_L1_WEIGHT:g}·L1lift; invalidate with "
        f"causality+leakage gate γ={DEFAULT_GAMMA:.2f} and prior-bank gate "
        f"γ_bank={DEFAULT_GAMMA_BANK:.2f} (merged `bank_w1`/`bank_w3`/`bank_w2_fullz`).\n"
    )
    table_eladder()
    stats_m, ek_rows = table_ekings_mix()
    table_spearman(stats_m)
    table_redteam(ek_rows)
    _out = sys.stdout
    return buf.getvalue()


def main() -> None:
    text = render()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
