"""Project full eval artifacts into chart-safe summaries (no raw logprobs)."""

from __future__ import annotations

import math
import statistics as st
from typing import Any

from ..score import gate_pass, l1_lift, lambda2, rank_term


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _turn_points(rows: list[dict], side: str) -> list[dict]:
    """One aggregate point per turn for a scored side."""
    out: list[dict] = []
    for r in rows:
        if not r.get("valid") or "pairs" not in r:
            continue
        pairs = r["pairs"]
        if not pairs:
            continue
        try:
            lam = st.mean(lambda2(p) for p in pairs)
            lift = st.mean(l1_lift(p) for p in pairs)
            mix = st.mean(rank_term(p) for p in pairs)
            gpass = st.mean(1.0 if gate_pass(p) else 0.0 for p in pairs)
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "turn_id": r.get("turn_id"),
            "side": side,
            "lambda2": _finite(lam),
            "l1lift": _finite(lift),
            "mix": _finite(mix),
            "gate_ok": gpass >= 0.5,
            "gate_pass_rate": _finite(gpass),
            "n_pairs": len(pairs),
        })
    return out


def project_series(artifact: dict) -> dict:
    """Per-turn series for both sides — safe for the browser."""
    king_rows = artifact.get("king_rows") or []
    chall_rows = artifact.get("challenger_rows") or []
    king_pts = _turn_points(king_rows, "king")
    chall_pts = _turn_points(chall_rows, "challenger")
    # Paired delta when both sides scored the same turn.
    k_by = {p["turn_id"]: p for p in king_pts if p.get("turn_id")}
    paired = []
    for cp in chall_pts:
        tid = cp.get("turn_id")
        kp = k_by.get(tid)
        if not kp:
            continue
        cm, km = cp.get("mix"), kp.get("mix")
        paired.append({
            "turn_id": tid,
            "delta_mix": _finite((cm or 0) - (km or 0)) if cm is not None and km is not None else None,
            "challenger_mix": cm,
            "king_mix": km,
            "challenger_lambda2": cp.get("lambda2"),
            "king_lambda2": kp.get("lambda2"),
            "challenger_l1lift": cp.get("l1lift"),
            "king_l1lift": kp.get("l1lift"),
            "challenger_gate_ok": cp.get("gate_ok"),
            "king_gate_ok": kp.get("gate_ok"),
        })
    return {
        "slice": artifact.get("slice"),
        "rejection_reason": artifact.get("rejection_reason"),
        "king": king_pts,
        "challenger": chall_pts,
        "paired": paired,
        "n_king_turns": len(king_pts),
        "n_challenger_turns": len(chall_pts),
        "n_paired_turns": len(paired),
    }


def _pair_detail(p: dict) -> dict:
    """One scored pair with its published rollout text and derived terms.

    Everything here is already public in the Hippius artifact; this just
    serves one turn of it without the multi-MB gunzip on the client.
    """
    out = {
        "thought": p.get("z_a"),
        "action": p.get("y_a"),
        "lambda2": None,
        "l1lift": None,
        "mix": None,
        "gate_ok": None,
    }
    try:
        out["lambda2"] = _finite(lambda2(p))
        out["l1lift"] = _finite(l1_lift(p))
        out["mix"] = _finite(rank_term(p))
        out["gate_ok"] = bool(gate_pass(p))
    except (KeyError, TypeError, ValueError):
        pass
    # Raw logprob components for full replay verification.
    for k, v in p.items():
        if k.startswith("lp"):
            out[k] = _finite(v)
    return out


def _side_turn(rows: list[dict], turn_id: str) -> dict | None:
    for r in rows:
        if r.get("turn_id") != turn_id:
            continue
        return {
            "valid": r.get("valid"),
            "bank_frac": _finite(r.get("bank_frac")),
            "n_pairs": r.get("n_pairs"),
            "pairs": [_pair_detail(p) for p in (r.get("pairs") or [])],
        }
    return None


def project_turn_detail(artifact: dict, turn_id: str) -> dict | None:
    """Full rollout detail for one turn: both sides + teacher references."""
    chall = _side_turn(artifact.get("challenger_rows") or [], turn_id)
    king = _side_turn(artifact.get("king_rows") or [], turn_id)
    refs = (artifact.get("teacher_refs") or {}).get(turn_id) or []
    if chall is None and king is None and not refs:
        return None
    return {
        "turn_id": turn_id,
        "challenger": chall,
        "king": king,
        "teacher_refs": [{
            "thought": r.get("z"),
            "action": r.get("y"),
            "lp_own": _finite(r.get("lp_own")),
            "lp_empty": _finite(r.get("lp_empty")),
        } for r in refs],
    }


def project_duel_summary(history_row: dict | None, artifact: dict | None,
                         series: dict | None = None) -> dict:
    """Combine history verdict + optional artifact projection."""
    row = history_row or {}
    art = artifact or {}
    summary = {
        "challenge_id": row.get("challenge_id") or art.get("challenge_id"),
        "event": row.get("event"),
        "at": row.get("at"),
        "repo": row.get("repo"),
        "hotkey": row.get("hotkey"),
        "uid": row.get("uid"),
        "duration_s": row.get("duration_s"),
        "revision": row.get("revision"),
        "accepted": row.get("accepted"),
        "challenger_wins": row.get("challenger_wins"),
        "z": row.get("z"),
        "margin": row.get("margin"),
        "se": row.get("se"),
        "n_paired_turns": row.get("n_paired_turns"),
        "rejection_reason": row.get("rejection_reason") or art.get("rejection_reason"),
        "error_code": row.get("error_code"),
        "error_detail": row.get("error_detail"),
        "reign_number": row.get("reign_number"),
        "score": row.get("score"),
        "score_king": row.get("score_king"),
        "gates": row.get("gates"),
        "challenger": row.get("challenger"),
        "king": row.get("king"),
        "slice": art.get("slice"),
        "has_artifact": bool(artifact),
        "has_series": bool(series and (
            series.get("n_challenger_turns") or series.get("n_king_turns"))),
    }
    # Convenience blob for the failure slide-over.
    if (summary.get("event") == "failed"
            or summary.get("accepted") is False
            or summary.get("error_code")
            or summary.get("error_detail")
            or summary.get("rejection_reason")):
        summary["failure"] = {
            "code": summary.get("error_code") or summary.get("rejection_reason"),
            "detail": (summary.get("error_detail")
                       or summary.get("rejection_reason")
                       or ""),
            "at": summary.get("at"),
            "repo": summary.get("repo"),
            "hotkey": summary.get("hotkey"),
            "revision": summary.get("revision"),
        }
    return summary
