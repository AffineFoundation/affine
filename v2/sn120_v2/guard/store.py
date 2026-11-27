"""Where a guard decision is kept: a private record and a public-safe row.

Two files, and the split between them is the whole point:

  verdicts.jsonl  PRIVATE. The judge's full answer, its reasoning, its own
                  answer, a proposed fixed key, the king hint, the soft flags,
                  token usage. Never published, never mirrored, never put in a
                  rotation input. Knowing the judge's answer to a live task
                  would be a scoring leak.
  guard.jsonl     PUBLIC and sealed. One canonical row per decision, with the
                  six fields the rotation rule reads and nothing else.

Both are append-only. The public ledger is a rotation input, so it is written
under the reserve's exclusive ``ingest_lock``, the same lock that serialises
ingest against a reload at open.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..canonical import canonical_json
from ..envs.reserve import GUARD, ingest_lock
from ..errors import ConfigError
from ..rotation import GUARD_PASS, GUARD_REJECT, GUARD_ROW_FIELDS, parse_guard_ledger
from .judge import Verdict

if TYPE_CHECKING:  # checks.py needs the math grader; the store must not
    from .checks import CheckResult

VERDICTS = "verdicts.jsonl"  # private; beside the guard ledger but never published


def utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def public_row(task_id: str, content_sha: str, passed: bool, category: str,
               guard_version: str, added_at: str | None = None) -> dict[str, str]:
    """The public-safe ledger row. Exactly ``GUARD_ROW_FIELDS``; no reasoning, no answer, no hint."""
    row = {
        "id": task_id,
        "content_sha": content_sha,
        "verdict": GUARD_PASS if passed else GUARD_REJECT,
        "category": (category or ("ok" if passed else "reject")).strip().lower(),
        "guard_version": guard_version,
        "added_at": added_at or utc_date(),
    }
    if set(row) != GUARD_ROW_FIELDS:
        raise ConfigError("a guard ledger row must carry exactly the public fields")
    return row


@dataclass(frozen=True)
class Decision:
    """One finished guard decision: what the public ledger records, and the private detail behind it."""

    task_id: str
    content_sha: str
    passed: bool
    category: str
    guard_version: str
    checks: CheckResult | None = None
    verdict: Verdict | None = None
    hint: str = ""
    added_at: str = ""

    def row(self) -> dict[str, str]:
        return public_row(self.task_id, self.content_sha, self.passed, self.category,
                          self.guard_version, self.added_at or None)

    def private(self) -> dict:
        """The private record. This is the only place the judge's own words are kept."""
        out: dict = {
            "id": self.task_id,
            "content_sha": self.content_sha,
            "verdict": GUARD_PASS if self.passed else GUARD_REJECT,
            "category": self.category,
            "guard_version": self.guard_version,
            "added_at": self.added_at or utc_date(),
            "flags": list(self.checks.flags) if self.checks else [],
            "rejects": list(self.checks.rejects) if self.checks else [],
            "hint": self.hint,
        }
        if self.verdict is not None:
            out["judge"] = {
                "verdict": self.verdict.verdict,
                "confidence": self.verdict.confidence,
                "reason": self.verdict.reason,
                "my_answer": self.verdict.my_answer,
                "fix_key": self.verdict.fix_key,
                "undecided": self.verdict.undecided,
                "tries": self.verdict.tries,
                "usage": self.verdict.usage,
                "cost_usd": round(self.verdict.cost_usd, 6),
            }
        return out


def guard_path(root: Path) -> Path:
    return Path(root) / GUARD


def verdicts_path(root: Path) -> Path:
    return Path(root) / VERDICTS


def append_private(root: Path, decisions: list[Decision]) -> int:
    """Append private records. Not a rotation input, so it needs no lock beyond the file append."""
    if not decisions:
        return 0
    path = verdicts_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.private(), sort_keys=True, ensure_ascii=False).encode() + b"\n")
    return len(decisions)


def append_ledger(root: Path, decisions: list[Decision]) -> tuple[int, int]:
    """Append public rows under the reserve's exclusive ingest lock.

    Idempotent per ``(id, content_sha)``: a pair the ledger already decides is
    skipped, so a rerun after a crash adds nothing. Returns (appended, skipped).
    """
    if not decisions:
        return 0, 0
    root = Path(root)
    with ingest_lock(root, exclusive=True):
        path = guard_path(root)
        existing = path.read_bytes() if path.exists() else b""
        ledger = parse_guard_ledger(existing)
        seen = {(task_id, sha) for mapping in (ledger.passes, ledger.rejects) for task_id, sha in mapping.items()}
        lines, appended, skipped = [], 0, 0
        for decision in decisions:
            row = decision.row()
            key = (row["id"], row["content_sha"])
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            lines.append(canonical_json(row) + b"\n")
            appended += 1
        if lines:
            body = existing + b"".join(lines)
            parse_guard_ledger(body)  # refuse to write bytes the rule itself would reject
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(b"".join(lines))
        return appended, skipped


def record(root: Path, decisions: list[Decision]) -> tuple[int, int]:
    """Write the private records first, then the public rows.

    In that order on purpose: the public ledger is what admits a task, so it is
    never ahead of the private evidence that explains it.
    """
    append_private(root, decisions)
    return append_ledger(root, decisions)
