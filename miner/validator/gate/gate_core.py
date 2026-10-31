"""SN120 checkpoint promotion gate: sample draw + paired decision (pure functions, no I/O).

Jacob pt682u 2026-10-26 14:25 CR. Epoch N mines on head H and trains candidate C.
  S  = fresh private uniform sample of `sample_size` task indices from the env-probe pool (secret seed).
  T  = S minus every task index in epoch N's training batch (dropped from BOTH sides).
  GREEN iff paired z(C vs H on T) > z_min (>= when z_inclusive) AND cap-hit rate of C on T < cap_max. Else RED.
Per-task cells are probe cells {n, c, b, ...} (n samples, c correct, b budget/cap hits).
"""
from __future__ import annotations
import hashlib, math, random

VERSION = "sn120-promotion-gate-v1"
DEFAULTS = dict(sample_size=1000, z_min=-1.0, cap_max=0.20, pool=7496, env="affine_math")


def draw_sample(seed_hex: str, pool: int, n: int) -> list[int]:
    """Uniform without-replacement draw of n indices from range(pool), from a secret 32-byte hex seed."""
    if not (isinstance(seed_hex, str) and len(seed_hex) == 64):
        raise ValueError("seed must be 64 hex chars")
    rng = random.Random(int(hashlib.sha256(("%s:%s:gate-sample" % (VERSION, seed_hex)).encode()).hexdigest(), 16))
    return sorted(rng.sample(range(int(pool)), min(int(n), int(pool))))


def sample_commitment(seed_hex: str, indices: list[int]) -> str:
    """Public commitment (published at draw time); seed + indices are revealed only after the lag."""
    return hashlib.sha256(("%s|%s|%s" % (VERSION, seed_hex, ",".join(map(str, indices)))).encode()).hexdigest()


def _rate(cell):
    if not cell:
        return None
    n = cell.get("n") if isinstance(cell, dict) else cell[0]
    c = cell.get("c") if isinstance(cell, dict) else cell[1]
    if not n:
        return None
    return float(c or 0) / float(n)


def _caps(cell):
    if isinstance(cell, dict):
        return int(cell.get("b") or 0), int(cell.get("n") or 0)
    return int(cell[2] or 0), int(cell[0] or 0)


def paired(h_pt: dict, c_pt: dict, tasks) -> dict:
    """Paired comparison of C vs H on `tasks` (tasks lacking a graded cell on either side are skipped)."""
    n = gains = losses = 0
    sd = sd2 = h_sum = c_sum = 0.0
    cap_b = cap_n = 0
    for t in tasks:
        xh, xc = _rate(h_pt.get(t)), _rate(c_pt.get(t))
        if xh is None or xc is None:
            continue
        n += 1; d = xc - xh; sd += d; sd2 += d * d; h_sum += xh; c_sum += xc
        gains += xh < 0.5 <= xc; losses += xc < 0.5 <= xh
        b, m = _caps(c_pt.get(t)); cap_b += b; cap_n += m
    z = (sd / math.sqrt(sd2)) if sd2 > 0 else 0.0
    return dict(n=n, h_pass=(h_sum / n if n else None), c_pass=(c_sum / n if n else None),
                delta=((c_sum - h_sum) / n if n else None), gains=gains, losses=losses, z=z,
                c_cap_rate=(cap_b / cap_n if cap_n else None))


def decide(h_pt: dict, c_pt: dict, sample: list[int], batch_tasks, *, z_min=-1.0, cap_max=0.20,
           min_tasks=200, z_inclusive=False) -> dict:
    batch = set(int(t) for t in batch_tasks)
    T = [t for t in sample if t not in batch]
    p = paired(h_pt, c_pt, T)
    pre_post = paired(h_pt, c_pt, sorted(batch))          # analysis only, never gating
    reasons = []
    if p["n"] < min_tasks:
        reasons.append("insufficient_paired_tasks(%d<%d)" % (p["n"], min_tasks))
    # Jacob pt692u (2026-10-26 17:26 CR): pass unless clearly worse -> z >= z_min (inclusive) with z_min -1.5 from e154
    z_ok = (p["z"] >= z_min) if z_inclusive else (p["z"] > z_min)
    if not z_ok:
        reasons.append("paired_z %.3f %s %.2f" % (p["z"], "<" if z_inclusive else "<=", z_min))
    if p["c_cap_rate"] is None or not p["c_cap_rate"] < cap_max:
        reasons.append("cap_rate %s >= %.2f" % (None if p["c_cap_rate"] is None else round(p["c_cap_rate"], 4), cap_max))
    return dict(version=VERSION, decision=("red" if reasons else "green"), reasons=reasons,
                sample_size=len(sample), batch_overlap=len(sample) - len(T), evaluated=p,
                batch_pre_post=dict(pre_post, note="analysis only (not gating)"),
                z_min=z_min, z_inclusive=bool(z_inclusive), cap_max=cap_max, min_tasks=min_tasks)
