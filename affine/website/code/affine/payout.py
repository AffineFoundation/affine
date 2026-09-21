"""King payout window — who gets paid, and how much (operator directive
2026-09-14 11:09 UTC).

Rule in plain words: a crown is paid for at most `window_s` seconds
(72 h) after `crowned_at`. Every crown inside its window is paid one equal
share; a crown past its window earns nothing, even while it still holds the
throne for duels. Two paid crowns → 50/50, three → thirds. No paid crown →
the emission burns (`burn_uid`, the same fallback the unpaid genesis uses).

What counts as a crown: one row per reign in the stored lineage (the sitting
king + `king.previous`). Reigns that were revoked or reverted are removed
from the lineage by the revert paths (`State.revert_king`,
`ops/exploit-audit/auditd.py`), so they never appear here; a row flagged
`revoked` is excluded defensively. Genesis/seed rows have an empty hotkey
and take no share. A hotkey that holds several crowns inside the window
holds one share PER CROWN (summed per hotkey when the weights are set).

Pure functions only: no chain, no clock, no state file. `State` and the
dashboard readers both build their view through `annotate_lineage`, so the
validator's weights and the published paid set are the same computation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

# One share per paid crown; shares are expressed as a fraction of 1.0 and as
# basis points (1/10000) for the dashboard.
BPS = 10000


def rule_text(window_hours: float) -> str:
    """The payout rule in plain words, for api/v1/contract and the snapshot."""
    h = int(window_hours) if float(window_hours).is_integer() else window_hours
    return (f"A crown is paid for at most {h} hours after it is won. Every "
            f"crown still inside its {h}-hour window gets one equal share of "
            "the miner emission (2 paid crowns = 50/50, 3 = one third each). "
            "A crown older than that earns nothing, even while it still holds "
            "the throne for duels. No crown inside the window = the emission "
            "burns. Revoked reigns are never paid. A hotkey with several "
            "crowns inside the window holds one share per crown. The throne "
            "and duel rules are unchanged.")


def contract_block(window_hours: float, effective_at: str | None,
                   burn_uid: int) -> dict:
    """The rule as published in api/v1/contract (static part; the live paid
    set is in api/v1/snapshot `payout`)."""
    return {
        "rule": rule_text(window_hours),
        "window_hours": window_hours,
        "effective_at": effective_at or None,
        "share_per_paid_crown": "1 / n_paid_crowns",
        "burn_uid": int(burn_uid),
        "code": "affine/payout.py::annotate_lineage",
    }


def parse_iso(ts: str | None) -> datetime | None:
    """`datetime.isoformat()` strings as written by `state.now_iso()`; naive
    values are read as UTC. None/unparseable → None."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def paid_until(crowned_at: str | None, window_s: float) -> datetime | None:
    """The instant a crown stops being paid: `crowned_at + window_s`."""
    start = parse_iso(crowned_at)
    if start is None:
        return None
    return start + timedelta(seconds=float(window_s))


def lineage_rows(king: Mapping | None) -> list[dict]:
    """One row per stored reign, current first, then `previous` newest-first.

    Works on the plain dict shape of `state.json["king"]` (also what
    `dataclasses.asdict(King)` gives). NOT deduped by hotkey: each reign is
    its own crown and its own payout window.
    """
    if not king:
        return []
    rows = [{
        "reign_number": king.get("reign_number"),
        "repo": king.get("repo", ""),
        "revision": king.get("revision", ""),
        "hotkey": king.get("hotkey", "") or "",
        "crowned_at": king.get("crowned_at"),
        "block": king.get("block"),
        "score": king.get("score"),
        "current": True,
    }]
    for p in king.get("previous") or []:
        row = {
            "reign_number": p.get("reign_number"),
            "repo": p.get("repo", ""),
            "revision": p.get("revision", ""),
            "hotkey": p.get("hotkey", "") or "",
            "crowned_at": p.get("crowned_at"),
            "block": p.get("block"),
            "score": p.get("score"),
            "current": False,
        }
        if p.get("uid") is not None:
            row["uid"] = int(p["uid"])
        if p.get("revoked"):
            row["revoked"] = True
        rows.append(row)
    return rows


def annotate_lineage(rows: Iterable[Mapping], *, window_s: float,
                     now: datetime,
                     inaccessible: Iterable[str] = ()) -> list[dict]:
    """Stamp every lineage row with its payout status.

    Added keys per row:
      paid_until   ISO instant the crown's window closes (None if no crown time)
      expired      True once `now >= paid_until` (the boundary itself is unpaid)
      inaccessible True if the hotkey's model is provably gone/gated
      earning      True iff this crown is paid right now
      share        fraction of emissions for THIS crown (1/n_paid), else 0.0
      weight_bps   share in basis points (integer floor), else 0
    Rows are returned in the input order (current king first).
    """
    gone = set(inaccessible or ())
    out: list[dict] = []
    for r in rows:
        m = dict(r)
        hk = str(m.get("hotkey") or "")
        until = paid_until(m.get("crowned_at"), window_s)
        m["paid_until"] = until.isoformat() if until is not None else None
        m["expired"] = (until is None) or (now >= until)
        m["inaccessible"] = bool(hk) and hk in gone
        m["earning"] = (bool(hk) and not m.get("revoked")
                        and not m["expired"] and not m["inaccessible"])
        out.append(m)
    n_paid = sum(1 for m in out if m["earning"])
    share = (1.0 / n_paid) if n_paid else 0.0
    bps = (BPS // n_paid) if n_paid else 0
    for m in out:
        m["share"] = share if m["earning"] else 0.0
        m["weight_bps"] = bps if m["earning"] else 0
    return out


def paid_crowns(annotated: Iterable[Mapping]) -> list[dict]:
    """The paid set: every crown with `earning == True`, input order kept."""
    return [dict(m) for m in annotated if m.get("earning")]


def shares_by_hotkey(annotated: Iterable[Mapping]) -> dict[str, float]:
    """Sum the per-crown shares per hotkey (a hotkey with two paid crowns
    gets 2/n). Insertion order = first appearance (current king first)."""
    out: dict[str, float] = {}
    for m in annotated:
        if not m.get("earning"):
            continue
        hk = str(m.get("hotkey") or "")
        out[hk] = out.get(hk, 0.0) + float(m.get("share") or 0.0)
    return out


def uid_weights(shares: Mapping[str, float], uid_of: Mapping[str, int],
                burn_uid: int) -> tuple[list[int], list[float]]:
    """Map hotkey shares onto registered uids for `set_weights`.

    Unregistered hotkeys (not in `uid_of`) are dropped and the remaining
    shares are renormalised to sum to 1 — the same "unregistered members
    are skipped" behaviour the rolling chain had. Two hotkeys on one uid
    cannot happen on Bittensor, but the shares are summed per uid anyway.
    Nothing payable → `([burn_uid], [1.0])`.
    """
    acc: dict[int, float] = {}
    for hk, share in shares.items():
        uid = uid_of.get(hk)
        if uid is None or share <= 0:
            continue
        acc[int(uid)] = acc.get(int(uid), 0.0) + float(share)
    total = sum(acc.values())
    if not acc or total <= 0:
        return [int(burn_uid)], [1.0]
    uids = list(acc.keys())
    return uids, [acc[u] / total for u in uids]


def describe(annotated: Iterable[Mapping]) -> str:
    """One log line: `#12 5DJ91T4Z… 1/1 until 2026-09-15T21:48Z`, or `burn`."""
    paid = paid_crowns(annotated)
    if not paid:
        return "burn (no crown inside the payout window)"
    n = len(paid)
    parts = []
    for m in paid:
        hk = str(m.get("hotkey") or "")
        until = str(m.get("paid_until") or "")[:16]
        parts.append(f"#{m.get('reign_number')} {hk[:8]}… 1/{n} until {until}Z")
    return "; ".join(parts)
