"""SQLite projection of history.jsonl + bench_history.jsonl."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from .readers import history_row_from_raw

log = logging.getLogger("affine.dash.index")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT,
  event TEXT,
  challenge_id TEXT,
  repo TEXT,
  hotkey TEXT,
  accepted INTEGER,
  z REAL,
  score REAL,
  score_king REAL,
  rejection_reason TEXT,
  error_code TEXT,
  payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_at ON history(at);
CREATE INDEX IF NOT EXISTS idx_history_event ON history(event);
CREATE INDEX IF NOT EXISTS idx_history_cid ON history(challenge_id);

CREATE TABLE IF NOT EXISTS bench (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  finished_at TEXT,
  repo TEXT,
  revision TEXT,
  hotkey TEXT,
  suite TEXT,
  label TEXT,
  ok INTEGER,
  score REAL,
  payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bench_repo ON bench(repo);
"""


class DashIndex:
    """Append-only ingest from jsonl files via byte-offset watermarks."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.db_path = state_dir / "dash.sqlite"
        self.history_path = state_dir / "history.jsonl"
        self.bench_path = state_dir / "bench_history.jsonl"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _get_meta(self, key: str, default: str = "0") -> str:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))

    def ingest(self) -> None:
        with self._lock:
            self._ingest_history()
            self._ingest_bench()

    def _ingest_history(self) -> None:
        path = self.history_path
        if not path.exists():
            return
        offset = int(self._get_meta("history_offset", "0"))
        size = path.stat().st_size
        if offset > size:
            # File truncated/replaced — rebuild.
            self._conn.execute("DELETE FROM history")
            offset = 0
        if offset == size:
            return
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read()
            new_offset = f.tell()
        # Incomplete final line stays unconsumed.
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n")
            if cut < 0:
                return
            raw = raw[: cut + 1]
            new_offset = offset + len(raw)
        rows = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            slim = history_row_from_raw(rec)
            rows.append((
                slim.get("at"), slim.get("event"), slim.get("challenge_id"),
                slim.get("repo"), slim.get("hotkey"),
                (1 if slim.get("accepted") is True
                 else 0 if slim.get("accepted") is False else None),
                slim.get("z"), slim.get("score"), slim.get("score_king"),
                slim.get("rejection_reason"), slim.get("error_code"),
                json.dumps(slim, default=str),
            ))
        if rows:
            self._conn.executemany(
                "INSERT INTO history(at, event, challenge_id, repo, hotkey, "
                "accepted, z, score, score_king, rejection_reason, error_code, "
                "payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                rows)
        self._set_meta("history_offset", str(new_offset))

    def _ingest_bench(self) -> None:
        path = self.bench_path
        if not path.exists():
            return
        offset = int(self._get_meta("bench_offset", "0"))
        size = path.stat().st_size
        if offset > size:
            self._conn.execute("DELETE FROM bench")
            offset = 0
        if offset == size:
            return
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read()
            new_offset = f.tell()
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n")
            if cut < 0:
                return
            raw = raw[: cut + 1]
            new_offset = offset + len(raw)
        rows = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = rec.get("result") or {}
            rows.append((
                rec.get("finished_at"), rec.get("repo"), rec.get("revision"),
                rec.get("hotkey"), rec.get("suite"), rec.get("label"),
                1 if result.get("ok") else 0,
                result.get("score"),
                json.dumps(rec, default=str),
            ))
        if rows:
            self._conn.executemany(
                "INSERT INTO bench(finished_at, repo, revision, hotkey, suite, "
                "label, ok, score, payload) VALUES (?,?,?,?,?,?,?,?,?)",
                rows)
        self._set_meta("bench_offset", str(new_offset))

    def query_history(self, *, limit: int = 100, cursor: int | None = None,
                      q: str = "", event: str = "") -> dict:
        limit = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list = []
        if cursor is not None:
            clauses.append("id < ?")
            params.append(int(cursor))
        if event:
            clauses.append("event = ?")
            params.append(event)
        if q:
            like = f"%{q.lower()}%"
            clauses.append(
                "(LOWER(COALESCE(repo,'')) LIKE ? OR "
                "LOWER(COALESCE(hotkey,'')) LIKE ? OR "
                "LOWER(COALESCE(challenge_id,'')) LIKE ? OR "
                "LOWER(COALESCE(rejection_reason,'')) LIKE ? OR "
                "LOWER(COALESCE(error_code,'')) LIKE ?)")
            params.extend([like, like, like, like, like])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (f"SELECT id, payload FROM history{where} "
               f"ORDER BY id DESC LIMIT ?")
        params.append(limit)
        with self._lock:
            self.ingest()
            rows = self._conn.execute(sql, params).fetchall()
        items = []
        next_cursor = None
        for row in rows:
            try:
                items.append(json.loads(row["payload"]))
            except json.JSONDecodeError:
                continue
            next_cursor = row["id"]
        return {"items": items, "next_cursor": next_cursor}

    def get_history_by_challenge(self, challenge_id: str) -> dict | None:
        with self._lock:
            self.ingest()
            row = self._conn.execute(
                "SELECT payload FROM history WHERE challenge_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (challenge_id,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            return None

    def benchmarks_payload(self, active_jobs: list[dict] | None = None) -> dict:
        with self._lock:
            self.ingest()
            rows = self._conn.execute(
                "SELECT payload FROM bench ORDER BY id ASC").fetchall()
        models: dict[str, dict] = {}
        for row in rows:
            try:
                rec = json.loads(row["payload"])
            except json.JSONDecodeError:
                continue
            result = rec.get("result") or {}
            # One value per model (keyed by repo): its latest successful
            # score. Failures never reach the panel — see dashboard.py.
            if not result.get("ok") or result.get("score") is None:
                continue
            m = models.setdefault(rec.get("repo"), {
                "model_repo": rec.get("repo"),
                "revision": rec.get("revision"),
                "label": rec.get("label", ""),
                "hotkey": rec.get("hotkey", ""),
                "suites": {},
            })
            m["revision"] = rec.get("revision")  # chronological: last ok wins
            m["suites"][rec["suite"]] = {
                "score": result.get("score"),
                "ok": True,
                "n_sims": result.get("n_sims"),
                "finished_at": rec.get("finished_at"),
            }
        active = [{
            "job_id": j["job_id"], "model_repo": j["repo"], "suite": j["suite"],
            "state": j["state"], "queued_at": j["queued_at"],
        } for j in (active_jobs or [])]
        return {"models": list(models.values()), "active": active}
